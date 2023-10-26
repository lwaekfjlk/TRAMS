import os
import logging
import wandb
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from utils.modeling_transfo_xl import TransfoXLLMHeadModel, TransfoXLConfig
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils import get_lm_corpus
from earlystopping import EarlyStopper


class Trainer(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.set_tool()
        self.set_dist()
        self.set_seed()

        self.train_iter, self.valid_iter, self.test_iter = self.prepare_data()
        self.model = self.get_model(use_checkpoint=self.args.use_checkpoint)
        self.optimizer = Adam(params=self.model.parameters(), lr=self.args.lr)
        self.scheduler = self.get_scheduler()
        self.earlystopper = EarlyStopper(args, self.logger)


    def avg_rank(self, scalar):
        if self.args.local_rank == -1:
            return scalar
        scalar_t = torch.tensor(
            scalar, 
            dtype=torch.float, 
            device=self.device
        ) / torch.distributed.get_world_size()
        torch.distributed.all_reduce(
            scalar_t, 
            op=torch.distributed.ReduceOp.SUM
        )
        return scalar_t.item()


    def set_tool(self):
        if self.args.local_rank in [-1, 0]:
            os.environ['WANDB_API_KEY'] = '972035264241fb0f6cc3cab51a5d82f47ca713db'
            #wandb.init(project="LTDecoder", name=self.args.timestamp, config=self.args, dir='./tmp')
            wandb.init(mode='disabled')
        self.logger = logging.getLogger(__file__)


    def set_dist(self):
        self.args.distributed = self.args.local_rank != -1
        logging.basicConfig(
            level=logging.INFO 
            if self.args.local_rank in [-1, 0] 
            else logging.WARN
        )
        if self.args.distributed:
            self.device = torch.device("cuda", self.args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", 
                init_method="env://"
            )
        else:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )


    def set_seed(self):
        if self.args.distributed:
            rank = torch.distributed.get_rank()
            torch.manual_seed(self.args.seed_id + rank_id)
            torch.cuda.manual_seed(self.args.seed_id + rank_id)
            torch.cuda.manual_seed_all(self.args.seed_id + rank_id)
        else:
            torch.manual_seed(self.args.seed_id)
            torch.cuda.manual_seed(self.args.seed_id)
            torch.cuda.manual_seed_all(self.args.seed_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def log(self, str):
        if self.args.local_rank in [-1, 0]:
            self.logger.info(str)


    def wandb_log(self, dict):
        if self.args.local_rank in [-1, 0]:
            wandb.log(dict)


    def judge_earlystopping(self, metric, model, optimizer, metric_direction='small'):
        if self.args.local_rank in [-1, 0]:
            self.earlystopper(metric, model, optimizer, metric_direction)
            return self.earlystopper.early_stop
        else:
            return


    def get_config(self):
    # adaptive softmax / embedding
        cutoffs, tie_projs = [], [False]
        if self.args.adaptive:
            assert self.args.dataset in ['wt103']
            if self.args.dataset == 'wt103':
                cutoffs = [20000, 40000, 200000]
                tie_projs += [True] * len(cutoffs)

        config = TransfoXLConfig(
            vocab_size=self.args.vocab_size,
            d_model=self.args.d_model,
            d_embed=self.args.d_model,
            n_head=self.args.n_head,
            d_head=self.args.d_head,
            d_inner=self.args.d_inner,
            div_val=self.args.div_val,
            pre_lnorm=self.args.pre_lnorm,
            n_layer=self.args.n_layer,
            tgt_len=self.args.tgt_len,
            mem_len=self.args.mem_len,
            ext_len=self.args.ext_len,
            clamp_len=self.args.clamp_len,
            same_length=self.args.same_length,
            attn_type=self.args.attn_type,
            sample_softmax=self.args.sample_softmax,
            adaptive=self.args.adaptive,
            dropout=self.args.dropout,
            dropatt=self.args.dropatt,
            untie_r=self.args.untie_r,
            init_range=self.args.init_range,
            proj_init_std=self.args.proj_init_std,
            init_std=self.args.init_std,
            layer_norm_epsilon=self.args.layer_norm_epsilon,
            eos_token_id=self.vocab.get_idx('<eos>'),
            cutoffs=cutoffs,
            tie_projs=tie_projs,
        )
        return config


    def get_model(self, use_checkpoint=False):
        config = self.get_config()

        if use_checkpoint:
            model = TransfoXLLMHeadModel(
                config=config, 
                args=self.args
            ).to(self.device)
            model.load_state_dict(torch.load(self.args.pretrained_model_name), strict=False)
        else:
            model = TransfoXLLMHeadModel(config=config, args=self.args).to(self.device)

        if self.args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)
            model = DistributedDataParallel(model, device_ids=[self.args.local_rank], output_device=self.args.local_rank)
        return model


    def load_model_ft(self, name):
        config = self.get_config()
        model = TransfoXLLMHeadModel(
            config=config, 
            args=self.args
        ).to(self.device)
        # TODO (haofeiyu): current text8 and enwik8 has problems with adaptive
        # 2022/11/25 actually train text8 and enwik8 with adaptive
        model.load_state_dict(torch.load(name), strict=False)
        return model


    def get_scheduler(self):
        if self.args.scheduler == "noam":
            def noam_lambda(step):
                step = max(step, 1)
                coef = self.args.model_size ** (-0.5) * min(
                    step ** (-0.5), 
                    step * self.args.warmup_steps ** (-1.5)
                )
                return coef
            self.log(
                '====used GPU number: {}====='.format(torch.cuda.device_count())
            )
            self.args.warmup_steps = min(
                len(self.train_iter)//self.args.grad_acc_steps+1, 
                self.args.warmup_steps
            )
            scheduler = LambdaLR(
                self.optimizer, 
                lr_lambda=noam_lambda
            )
        elif self.args.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer, 
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.max_training_steps,
            )
        elif self.args.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.max_training_steps,
            )
        else:
            scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        return scheduler


    def prepare_data(self):
        self.log('Preparing data...')
        self.corpus = get_lm_corpus(self.args.dataset_dir, self.args.dataset)
        if self.args.dataset in ['enwik8', 'text8']:
            self.corpus.vocab.add_symbol('<eos>')
        self.vocab = self.corpus.vocab
        self.args.vocab_size = len(self.vocab.idx2sym)

        train_iter = self.corpus.get_iterator('train', self.args.train_batch_size, self.args.tgt_len,
            device=self.device, ext_len=self.args.ext_len)
        val_iter = self.corpus.get_iterator('valid', self.args.eval_batch_size, self.args.eval_tgt_len,
            device=self.device, ext_len=self.args.ext_len)
        if self.args.testing_mode or self.args.training_mode:
            test_iter = self.corpus.get_iterator('test', self.args.eval_batch_size, self.args.eval_tgt_len,
                device=self.device, ext_len=self.args.ext_len)
        elif self.args.inference_mode:
            test_iter = self.corpus.get_iterator('test', 1, self.args.max_generation_len,
                device=self.device, ext_len=self.args.ext_len)
        return train_iter, val_iter, test_iter


    def training_mode(self):
        self.step = 0
        for epoch in range(self.args.num_of_epoch):
            self.train(self.args, self.model, self.optimizer, self.scheduler, self.train_iter, self.valid_iter, self.device)
        return


    def testing_mode(self):
        self.test(self.args, self.model, self.vocab, self.test_iter, self.device)
        return


    def inference_mode(self):
        self.inference(self.args, self.model, self.vocab, self.test_iter, self.device)
        return

    
    def train(self, args, model, optimizer, scheduler, train_iter, valid_iter, device):
        pass 


    def inference(self, args, model, vocab, loader, device):
        pass


    def test(self, args, model, loader, device):
        pass 

