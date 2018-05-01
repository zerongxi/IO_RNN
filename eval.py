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
        self.data = DataLoader(data_path, self.mode, self.batch_size)
        if save_path is None:
            save_path = os.path.join(hps.model_path, 'pred_temp.h5')
        self.data.new_epoch()

        x, y = self.data.get_batch()
        prediction = []
        while x is not None and y is not None:
            x = torch.autograd.Variable(torch.from_numpy(x)).cuda()
            out = self.model(x, self.mode, self.out_len)
            out = out.data.cpu().numpy()
            pred = []
            for field in out:
                pred.append(np.concatenate(np.swapaxes(out[field], 0, 1), 0))
            prediction.append(pred)
            x, y = self.data.get_batch()
        prediction = np.swapaxes(prediction, 0, 1)
        dst = h5py.File(save_path, 'w')
        for cnt in range(len(prediction)):
            tosave = np.concatenate(prediction[cnt], 0)
            dst.create_dataset(hps.dst_fields[cnt], data=tosave)
        dst.close()
        return save_path

    def eval(self, gt_path, pred_path):
        beg = self.data.beg
        end = self.data.end

        gt_h5f = h5py.File(gt_path, 'r')
        pred_h5f = h5py.File(pred_path, 'r')

        losses = {
            'asu': self.get_loss('cross_entropy', gt_h5f['asu'][beg:end], pred_h5f['asu'][beg:end]),
            'size': self.get_loss('mean_square', gt_h5f['size'][beg:end], pred_h5f['size'][beg:end]),
            'opcode': self.get_loss('cross_entropy', gt_h5f['opcode'][beg:end], pred_h5f['opcode'][beg:end]),
            'time_diff': self.get_loss('mean_square', gt_h5f['time_diff'][beg:end], pred_h5f['time_diff'][beg:end])
        }
        losses['all'] = np.sum([losses[i] for i in losses])
        accus = self.get_accu()
        return losses, accus

    def get_loss(self, metric, gt, pred):
        if metric == 'cross_entropy':
            mid = -(1.0/gt.shape[0]) * np.sum(gt*np.log(pred) + (1-gt)*np.log(1-pred))
        elif metric == 'mean_square':
            mid = np.mean(np.square(np.subtract(pred, gt)))
        else:
            ValueError('%s not implemented yet!' % metric)
        return mid

    def get_accu(self, gt, pred):
        def compute_accu(pred, gt):
            _, pred_label = torch.max(pred, 2)
            _, gt_label = torch.max(gt, 2)
            total = gt_label.nelement()
            correct = (pred_label == gt_label).sum().item()
            return torch.div(correct / total)
        self.data_shape = self.model.data_shape
        accus = {
            'asu': compute_accu(
                pred[:, :, :self.data_shape[2][0]],
                gt[:, :, :self.data_shape[2][0]]
            ),
            'size': torch.mean(
                pred[:, :, self.data_shape[2][0]:self.data_shape[2][1]] -
                gt[:, :, self.data_shape[2][0]:self.data_shape[2][1]]
            ),
            'opcode': compute_accu(
                pred[:, :, self.data_shape[2][1]:self.data_shape[2][2]],
                gt[:, :, self.data_shape[2][1]:self.data_shape[2][2]]
            ),
            'time_diff': torch.mean(
                pred[:, :, self.data_shape[2][2]:] -
                gt[:, :, self.data_shape[2][2]:]
            )
        }
        return accus