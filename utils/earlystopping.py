import logging
import numpy as np
import torch
import math
import os

class EarlyStopper(object):
    def __init__(self, args, logger, patience=20, verbose=True, delta=0):
        self.args = args
        self.logger = logger
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_min = np.Inf
        self.delta = delta

    def __call__(self, metric, model, optimizer, metric_direction):
        if metric_direction == 'small':
            score = -metric
        elif metric_direction == 'large':
            score = metric
        else:
            raise ValueError('no suitable metric direction')

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info('EarlyStopping patience: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, metric, model, optimizer):
        if self.args.dataset in ['enwik8', 'text8']:
            self.logger.info('BPC decreased from {:6f} --> {:6f}. Saving model ...'.format(self.metric_min/math.log(2), metric/math.log(2)))
        else:
            self.logger.info('PPL decreased from {:6f} --> {:6f}. Saving model ...'.format(math.exp(self.metric_min), math.exp(metric)))

        if self.args.task is None:
            name = self.args.dataset
        else:
            name = self.args.task
        torch.save(model.state_dict(), os.path.join(self.args.model_save_dir, name, self.args.model_checkpoint_name))
        torch.save(optimizer.state_dict(), os.path.join(self.args.optimizer_save_dir, name, self.args.optimizer_checkpoint_name))
        self.metric_min = metric