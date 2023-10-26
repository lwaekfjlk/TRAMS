import math
import time
import logging
import wandb
import os
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from torch.nn.parallel import DistributedDataParallel
from contextlib import nullcontext
from data_utils import Corpus, format_wt103, format_text8, format_enwik8
from utils.src import Trainer


class LTDecoderTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

    def train(self, args, model, optimizer, scheduler, train_iter, val_iter, device):
        model = model.module if hasattr(model, "module") else model

        accu_lens = 0
        accu_lm_loss = 0
        accu_aux_loss = 0
        total_lens = 0
        mems = None
        attn_outputs_mems= None
        for iter, (input_ids, labels, seq_len) in enumerate(train_iter):
            model.train()
            model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
            if args.distributed and (iter+1) % args.grad_acc_steps != 0:
                accele_context = model.no_sync
            else:
                accele_context = nullcontext

            with accele_context():
                outputs = model(input_ids, labels=labels, mems=mems, attn_outputs_mems=attn_outputs_mems, output_attn_outputs=True)
                loss = outputs.losses.mean()
                aux_loss = None
                mems = outputs.mems
                attn_outputs_mems = outputs.attn_outputs_mems
                accu_lens += seq_len
                accu_lm_loss += seq_len * loss.item()
                if aux_loss is not None:
                    accu_aux_loss += seq_len * aux_loss.item()
                    loss = loss + args.aux_coef * aux_loss
                loss = loss / args.grad_acc_steps
                loss.backward()
                batch_size = labels.size(0)
                total_lens += batch_size * args.tgt_len
                
                '''
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(name)
                '''

            if (iter + 1) % args.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_lm_loss = self.avg_rank(accu_lm_loss / accu_lens)
                train_aux_loss = self.avg_rank(accu_aux_loss / accu_lens)

                self.step += 1
                lr = scheduler.get_last_lr()[0]
                accu_lm_loss = 0
                accu_aux_loss = 0
                accu_lens = 0

                self.wandb_log({'logged_lr': lr, 'step': self.step})
                self.wandb_log({'train_lm_loss': train_lm_loss,'step': self.step})
                if args.compressive_mem_mode == 'conv_compressive':
                    self.wandb_log({'train_aux_loss': train_aux_loss, 'step': self.step})

                if self.step % args.evaluate_steps == 0:
                    earlystopping = self.evaluate(args, model, optimizer, val_iter, device)
                    if earlystopping:
                        break

        total_lens *= torch.cuda.device_count()
        self.log('training lens: {}'.format(total_lens))
        return


    def evaluate(self, args, model, optimizer, loader, device):
        model = model.module if hasattr(model, "module") else model
        model.eval()
        if args.mem_len == 0:
            model.reset_length(
                args.eval_tgt_len,
                args.ext_len+args.tgt_len-args.eval_tgt_len, 
                args.mem_len
            )
        else:
            model.reset_length(
                args.eval_tgt_len,
                args.ext_len, 
                args.mem_len+args.tgt_len-args.eval_tgt_len
            )
        accu_loss = 0
        accu_lens = 0
        total_lens = 0
        mems = None
        attn_outputs_mems= None
        mem_ids = torch.empty((args.eval_batch_size, 0), dtype=torch.long, device=device)
        with torch.no_grad():
            eval_iter = tqdm(loader, total=len(loader), desc='evaluating:')
            for input_ids, labels, seq_len in eval_iter:
                outputs = model(input_ids, labels=labels, mems=mems, attn_outputs_mems=attn_outputs_mems, output_attn_outputs=False)
                loss = outputs.losses.mean()
                mems = outputs.mems
                attn_outputs_mems = outputs.attn_outputs_mems
                accu_loss += seq_len * loss.item()
                accu_lens += seq_len
                batch_size = labels.size(0)
                total_lens += batch_size * args.eval_tgt_len
                mem_ids = torch.cat([mem_ids, input_ids], dim=1)
                mem_ids = mem_ids[:, -args.mem_len:]

        total_lens *= torch.cuda.device_count()
        loss = self.avg_rank(accu_loss / accu_lens)
        ppl = math.exp(loss)
        bpc = loss / math.log(2)
        self.log('evaluate lens: {}'.format(total_lens))
        self.log('val loss: {:.3f}'.format(loss))
        self.log('val ppl: {:.3f}'.format(ppl))
        if args.dataset in ['enwik8', 'text8']:
            self.log('val loss: {:.3f} val bpc: {:.3f}'.format(loss, bpc))
            self.wandb_log({'val bpc': bpc, 'step': self.step})
        else:
            self.log('val loss: {:.3f} val ppl: {:.3f}'.format(loss, ppl))
            self.wandb_log({'val ppl': ppl, 'step': self.step})

        earlystopping = self.judge_earlystopping(loss, model, optimizer, metric_direction='small')
        return earlystopping


    def test(self, args, model, vocab, loader, device):
        model = self.load_model_ft(os.path.join(args.pretrained_model_name))
        model = model.module if hasattr(model, "module") else model
        model.eval()
        model.reset_length(args.eval_tgt_len, args.ext_len, args.mem_len)

        accu_loss = 0
        accu_lens = 0
        total_lens = 0
        mems = None
        mem_ids = torch.empty((args.eval_batch_size, 0), dtype=torch.long, device=device)
        stat_attns = []
        stat_ppls = []
        with torch.no_grad():
            test_iter = tqdm(loader, total=len(loader), desc='testing:')
            for input_ids, labels, seq_len in test_iter:
                outputs = model(input_ids, labels=labels, mems=mems)
                loss = outputs.losses.mean()
                mems = outputs.mems
                accu_loss += seq_len * loss.item()
                accu_lens += seq_len
                batch_size = labels.size(0)
                total_lens += batch_size * args.eval_tgt_len
                
                mem_ids = torch.cat([mem_ids, input_ids], dim=1)
                mem_ids = mem_ids[:, -args.mem_len:]

        if args.stat_attention:
            self.stat_attention_final(args, stat_attns)
        if args.stat_ppl:
            self.stat_ppl_final(args, stat_ppls)
        total_lens *= torch.cuda.device_count()
        loss = self.avg_rank(accu_loss / accu_lens)
        ppl = math.exp(loss)
        bpc = loss / math.log(2)
        self.log('test lens: {}'.format(total_lens))
        self.log('test loss: {:.3f}'.format(loss))
        if args.dataset in ['enwik8', 'text8']:
            self.log('test loss: {:.3f} test bpc: {:.3f}'.format(loss, bpc))
        else:
            self.log('test loss: {:.3f} test ppl: {:.3f}'.format(loss, ppl))
        return


    def inference(self, args, model, vocab, loader, device):
        model = self.load_model_ft(os.path.join(args.model_save_dir, args.dataset, args.model_checkpoint_name))
        model = model.module if hasattr(model, "module") else model
        model.eval()

        if args.dataset == 'wt103':
            unk_id = vocab.get_idx('<unk>')
        with torch.no_grad():
            for input_ids, labels, seq_len in loader:
                input_ids = input_ids[0]
                prompt_ids = input_ids[:args.prompt_len]
                reference = vocab.get_symbols(input_ids)
                prediction = vocab.get_symbols(prompt_ids)
                for i in range(seq_len):
                    if i == 0:
                        input_ids = prompt_ids.view(1, -1)
                        outputs = model(input_ids)
                    else:
                        outputs = model(input_ids, mems=mems)
                    log_prob, mems = outputs[0], outputs[1]
                    prob = torch.exp(log_prob[0, -1, :])
                    if args.dataset == 'wt103':
                        prob[unk_id].data.fill_(0.)

                    # sample from the top-k tokens
                    top_prob, top_index = torch.topk(prob, k=5)
                    token = torch.multinomial(top_prob, 1)
                    token = top_index[token]

                    input_ids = token.detach().view(1, 1)
                    symbol = vocab.get_sym(token.item())
                    prediction.append(symbol)

                self.log('='*25 + '[PRED]' + '='*25)
                if args.dataset == 'wt103':
                    self.log(format_wt103(prediction))
                elif args.dataset == 'text8':
                    self.log(format_text8(prediction))
                elif args.dataset == 'enwik8':
                    self.log(format_enwik8(prediction))
                self.log('='*25 + '[REF]' + '='*25)
                if args.dataset == 'wt103':
                    self.log(format_wt103(reference))
                elif args.dataset == 'text8':
                    self.log(format_text8(reference))
                elif args.dataset == 'enwik8':
                    self.log(format_enwik8(reference))




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--topk_num', type=int, default=50)
    parser.add_argument('--remain_mem_num', type=int, default=150)
    parser.add_argument('--timestamp', type=str, default=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(round(time.time()*1000))/1000)))
    parser.add_argument('--use_checkpoint', action="store_true")
    parser.add_argument('--tgt_len', type=int, default=150)
    parser.add_argument('--eval_tgt_len', type=int, default=10)
    parser.add_argument('--ext_len', type=int, default=0)
    parser.add_argument('--mem_len', type=int, default=1200)
    parser.add_argument('--clamp_len', type=int, default=1000)
    parser.add_argument('--batch_chunk', type=int, default=1)
    parser.add_argument("--training_mode", action="store_true")
    parser.add_argument("--inference_mode", action="store_true")
    parser.add_argument("--testing_mode", action="store_true")
    parser.add_argument("--seed_id", type=int, default=12355)
    parser.add_argument("--model_size", type=int, default=1024)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--evaluate_steps", type=int, default=200)
    parser.add_argument("--max_training_steps", type=int, default=40000)
    parser.add_argument("--scheduler", type=str, default='noam')
    parser.add_argument("--grad_acc_steps", type=int, default=4)
    parser.add_argument("--max_norm", type=float, default=0.25)
    parser.add_argument("--train_batch_size", type=int, default=30)
    parser.add_argument("--eval_batch_size", type=int, default=30)
    parser.add_argument("--num_of_epoch", type=int, default=100)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_saved_checkpoint_name", type=str, default=None)
    parser.add_argument("--d_model", type=int, default=410)
    parser.add_argument("--n_layer", type=int, default=16)
    parser.add_argument("--n_head", type=int, default=10)
    parser.add_argument("--d_head", type=int, default=41)
    parser.add_argument("--sample_softmax", type=int, default=-1)
    parser.add_argument("--eos_token_id", type=int, default=0)
    parser.add_argument("--d_inner", type=int, default=2100)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dropatt", type=float, default=0.0)
    parser.add_argument("--pre_lnorm", action="store_true")
    parser.add_argument("--same_length", action="store_true")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--untie_r", action="store_true")
    parser.add_argument("--div_val", type=int, default=1)
    parser.add_argument("--attn_type", type=int, default=0)
    parser.add_argument("--init_range", type=float, default=0.1)
    parser.add_argument("--proj_init_std", type=float, default=0.01)
    parser.add_argument("--init_std", type=float, default=0.02)
    parser.add_argument("--layer_norm_epsilon", type=float, default=1e-05)
    parser.add_argument("--prompt_len", type=int, default=10)
    parser.add_argument("--compressive_mem_mode", type=str, default="origin")
    parser.add_argument("--compressive_method", type=str, default="min_norm")
    parser.add_argument("--uncompressive_mem_len", type=int, default=100)
    parser.add_argument("--mem_len_after_compression", type=int, default=None)
    parser.add_argument("--collect_v_norm", action="store_true")
    parser.add_argument("--norm_type", type=str, default="inf")
    parser.add_argument("--test_kv_sim", action="store_true")
    parser.add_argument("--cmem_ratio", type=int, default=4)
    parser.add_argument("--cmem_len", type=int, default=50)
    parser.add_argument("--visualize_attention", action="store_true")
    parser.add_argument("--visualize_hidden_states", action="store_true")
    parser.add_argument("--stat_attention", action="store_true")
    parser.add_argument("--stat_ppl", action="store_true")
    parser.add_argument("--aux_coef", type=float, default=0.1)
    parser.add_argument("--dataset_dir", type=str, default="./data/")
    parser.add_argument('--dataset', type=str, default='enwik8', choices=['ptb', 'wt2', 'wt103', 'enwik8', 'text8'])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--model_save_dir", type=str, default="./model/")
    parser.add_argument("--optimizer_save_dir", type=str, default="./model/")
    parser.add_argument("--tmp_save_dir", type=str, default="./tmp/")
    parser.add_argument("--max_generation_len", type=int, default=1000)
    parser.add_argument("--model_checkpoint_name", type=str, 
        default="best_TransfoXL_finetune_model_{}.ckpt".format('41'))
    parser.add_argument("--optimizer_checkpoint_name", type=str, 
        default="best_TransfoXL_finetune_optimizer_{}.ckpt".format('41'))
    parser.add_argument("--cache_dir", type=str , 
        default="../../LTDecoder_data/cache/dataset_cache")
    parser.add_argument("--untokenized_train_dataset_without_cache", type=str, 
        default="/toy_dataset3/train_dataset.jsonl")
    parser.add_argument("--untokenized_dev_dataset_without_cache", type=str, 
        default="/toy_dataset3/dev_dataset.jsonl")
    parser.add_argument("--untokenized_test_dataset_without_cache", type=str, 
        default="/toy_dataset3/test_dataset.jsonl")
    parser.add_argument("--pretrained_model_name", type=str, default=None)
    args = parser.parse_args() 


    trainer = LTDecoderTrainer(args)
    if args.training_mode:
        trainer.training_mode()
    elif args.inference_mode:
        trainer.inference_mode()
    elif args.testing_mode:
        start_time = time.time()
        trainer.testing_mode()
        end_time = time.time()
        print("Testing time: {}".format(end_time - start_time))
    else:
        raise ValueError('Wrong mode')

