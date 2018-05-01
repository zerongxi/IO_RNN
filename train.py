import torch
import os
import shutil
from datetime import datetime
import numpy as np

import hps
from model import LSTMseq2seq
from data_utils import DataLoader
from eval import Evaluation


class Train:

    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.train_data = DataLoader('train', self.data_path)
        self.model = LSTMseq2seq(hps.n_cell, hps.n_unit, hps.dropout)
        self.data_shape = self.model.data_shape
        self.best_val_loss = 100.0
        self.early_stopping = 0
        optimizer = getattr(torch.optim, hps.optimizer)
        self.optimizer = optimizer(self.model.parameters(), hps.lr)
        self.mse_fn = torch.nn.MSELoss()
        self.cross_entropy_fn = torch.nn.CrossEntropyLoss()
        self.model = torch.nn.DataParallel(self.model, dim=1)
        self.epoch = 0

        state = self.load_checkpoint()
        if state is not None:
            self.epoch = state['epoch'] + 1
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_val_loss = state['best_val_loss']
            self.early_stopping = state['early_stopping']

        self.model.float()
        self.model.cuda()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, hps.lr_decay, self.epoch - 1)
        pass

    def save_checkpoint(self, state, is_best):
        file_path = os.path.join(
            self.model_path,
            'model_epoch%s.pth.tar' % (str(state['epoch']).zfill(3))
        )
        torch.save(state, file_path)

        if is_best:
            shutil.copyfile(
                file_path,
                os.path.join(self.model_path, 'model_best.pth.tar')
            )
        return file_path

    def load_checkpoint(self):
        state = None
        if os.path.exists(self.model_path):
            saved_model = [f for f in os.listdir(self.model_path) if f.startswith('model_epoch')]
            if len(saved_model) > 0:
                saved_model.sort()
                state = torch.load(os.path.join(self.model_path, saved_model[-1]))
        else:
            os.makedirs(self.model_path)
        return state

    def get_loss(self, out, y):
        for i in y:
            out[i] = out[i].view(-1, out[i].shape[-1])
            y[i] = y[i].view(-1, y[i].shape[-1])
            if i == 'asu' or i == 'opcode':
                _, y[i] = torch.max(y[i], -1)
        pass
        losses = [
            self.cross_entropy_fn(out['asu'], y['asu']),
            self.mse_fn(out['size'], y['size']),
            self.cross_entropy_fn(out['opcode'], y['opcode']),
            self.mse_fn(out['time_diff'], y['time_diff'])
        ]
        return losses

    def get_accu(self, out, y):
        def compute_accu(pred, gt):
            _, pred_label = torch.max(pred, -1)
            _, gt_label = torch.max(gt, -1)
            total = gt_label.nelement()
            correct = (pred_label == gt_label).sum().item()
            return np.divide(correct, total)
        accus = {
            'asu': compute_accu(out['asu'], y['asu']),
            'size': torch.mean(out['size'] - y['size']),
            'opcode': compute_accu(out['opcode'], y['opcode']),
            'time_diff': torch.mean(out['time_diff'] - y['time_diff'])
        }
        return accus

    def on_epoch(self):

        batch_log = os.path.join(self.model_path, 'batch_log.txt')
        epoch_log = os.path.join(self.model_path, 'epoch_log.txt')
        epoch_t = datetime.now()
        self.scheduler.step()
        self.train_data.new_epoch()
        batch = 0
        train_loss = np.zeros((4,), np.float)
        x, y = self.train_data.get_batch()
        while x is not None and y is not None:
            batch_t = datetime.now()
            self.model.zero_grad()
            x = torch.autograd.Variable(torch.from_numpy(x)).cuda()
            for i in y:
                y[i] = torch.autograd.Variable(torch.from_numpy(y[i])).cuda()
            out = self.model(x, 'train')
            accus = self.get_accu(out, y)
            losses = self.get_loss(out, y)
            loss = sum(losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), hps.clip_norm)
            self.optimizer.step()
            train_loss += np.array([l.data.cpu().numpy() for l in losses])
            batch += 1
            x, y = self.train_data.get_batch()

            if batch % hps.print_step == 0:
                log_line = 'epoch%3d  batch%4d  total_loss:%f  asu_loss:%f  size_loss:%f  opcode_loss:%f  ' \
                        'time_diff_loss:%f  asu_accu:%f  opcode_accu:%f  batch_time:%s' % \
                           (self.epoch, batch, loss, losses[0], losses[1], losses[2], losses[3],
                            accus['asu'], accus['opcode'], datetime.now() - batch_t)
                print(log_line)
                '''
                with open(batch_log, 'a') as f:
                    f.write(log_line + '\n')
                '''

        train_loss /= batch

        evaluate = Evaluation('val')
        val_pred_path = evaluate.infer(self.data_path, model=self.model)
        val_losses, val_accus = evaluate.eval(self.data_path, val_pred_path)
        del evaluate

        is_best = False
        if val_losses['all'] < self.best_val_loss:
            self.best_val_loss = val_losses['all']
            is_best = True

        if is_best:
            self.early_stopping = 0
        else:
            self.early_stopping += 1

        state = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_losses,
            'best_val_loss': self.best_val_loss,
            'early_stopping': self.early_stopping
        }
        self.save_checkpoint(state, is_best)

        train_loss = np.sum(train_loss)
        log_line = 'epoch%3d train_loss:%f, val_loss:%f, asu_loss:%f, size_loss:%f, opcode_loss:%f, ' \
        'time_diff_loss:%f, asu_accu:%f, opcode:%f, cost_time: %s' %\
                   (self.epoch, train_loss, val_losses['all'], val_losses['asu'], val_losses['size'], val_losses['opcode'],
                    val_losses['time_diff'], val_accus['asu'], val_accus['opcode'], datetime.now() - epoch_t)
        print(log_line)
        with open(epoch_log, 'a') as f:
            f.writelines(log_line + '\n')

    def train(self):
        while self.epoch < hps.n_epoch:
            if self.early_stopping > hps.early_stopping:
                break
            self.on_epoch()
            self.epoch += 1
        pass


if __name__ == '__main__':
    train = Train(hps.data_path, hps.model_path)
    train.train()
    pass
