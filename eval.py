import hps
from data_utils import DataLoader
from model import LSTMseq2seq

import torch
import os
import numpy as np
import h5py
import shutil
import time


class Evaluation():

    def __init__(self, mode, bach_size=hps.batch_size):
        self.mode = mode
        self.in_len = hps.in_len
        self.out_len = hps.out_len
        self.model = None
        self.data = None
        self.batch_size = bach_size

    def load_model(self, ckpt_path):
        state = torch.load(ckpt_path['model'])
        model = LSTMseq2seq(hps.n_cell, hps.n_unit)
        model = torch.nn.DataParallel(model, dim=1)
        model.load_state_dict(state)
        model.float()
        model.cuda()
        return model

    def infer(self, data_path, save_path=None, model=None, ckpt_path=None):
        if model is not None:
            self.model = model
        else:
            if ckpt_path is None:
                ckpt_path = os.path.join(hps.model_path, 'model_best.pth.tar')
            self.model = self.load_model(ckpt_path)
        self.data = DataLoader(self.mode, data_path, self.batch_size)
        if save_path is None:
            save_path = os.path.join(hps.model_path, 'pred_temp.h5')
        self.data.new_epoch()

        x, y = self.data.get_batch()
        pred = [[], [], [], []]
        while x is not None and y is not None:
            x = torch.autograd.Variable(torch.from_numpy(x)).cuda()
            out = self.model(x, self.mode, self.out_len)
            cnt = 0
            for field in out:
                pred[cnt].append(np.concatenate(np.swapaxes(out[field].data.cpu().numpy(), 0, 1), 0))
                cnt += 1
            x, y = self.data.get_batch()

        dst = h5py.File(save_path, 'w')
        for cnt in range(len(pred)):
            tosave = np.concatenate(pred[cnt], 0)
            dst.create_dataset(hps.dst_fields[cnt], data=tosave)
        dst.close()
        return save_path

    def eval(self, gt_path, pred_path):
        beg = self.data.beg
        end = self.data.end

        gt_h5f = h5py.File(gt_path, 'r')
        pred_h5f = h5py.File(pred_path, 'r')
        end = pred_h5f['asu'].shape[0] + beg
        pass

        losses = {
            'asu': self.get_loss('cross_entropy', gt_h5f['asu'][beg:end], pred_h5f['asu']),
            'size': self.get_loss('mean_square', gt_h5f['size'][beg:end], pred_h5f['size']),
            'opcode': self.get_loss('cross_entropy', gt_h5f['opcode'][beg:end], pred_h5f['opcode']),
            'time_diff': self.get_loss('mean_square', gt_h5f['time_diff'][beg:end], pred_h5f['time_diff'])
        }
        losses['all'] = np.sum([losses[i] for i in losses])
        accus = self.get_accu(gt_h5f, pred_h5f, beg, end)
        return losses, accus

    def get_loss(self, metric, gt, pred):
        gt = np.array(gt, dtype=np.float)
        pred = np.array(pred, dtype=np.float)
        if metric == 'cross_entropy':
            mid = -(1.0/gt.shape[0]) *\
                  np.sum(np.multiply(gt, np.log(pred + 1e-6)) + np.multiply((1.0-gt), np.log(1.0-pred + 1e-6)))
        elif metric == 'mean_square':
            mid = np.mean(np.square(np.subtract(pred, gt)))
        else:
            ValueError('%s not implemented yet!' % metric)
        return mid

    def get_accu(self, gt, pred, beg, end):
        def compute_accu(pred, gt):
            pred_label = np.argmax(np.array(pred, np.float), -1)
            gt_label = np.argmax(np.array(gt, np.float), -1)
            total = gt_label.size
            correct = np.sum((pred_label == gt_label))
            return np.divide(correct, total)

        accus = {
            'asu': compute_accu(
                pred['asu'],
                gt['asu'][beg:end]
            ),
            'size': np.mean(
                np.array(pred['size']) -
                np.array(gt['size'][beg:end])
            ),
            'opcode': compute_accu(
                pred['opcode'],
                gt['opcode'][beg:end]
            ),
            'time_diff': np.mean(
                np.array(pred['time_diff']) -
                np.array(gt['time_diff'][beg:end])
            )
        }
        return accus