import h5py
from threading import Thread
from queue import Queue
from random import shuffle
import numpy as np
import time

import hps


class DataLoader:

    def __init__(self, mode, data_path=hps.data_path, batch_size=hps.batch_size, capacity=100, in_len=hps.in_len, out_len=hps.out_len):
        self.batch_size = batch_size
        self.mode = mode
        self.capacity = capacity
        self.in_len = in_len
        self.out_len = out_len
        self.queue = None
        self.read_thread = None

        h5f = h5py.File(data_path, 'r')
        self.data = {
            'asu': np.array(h5f['asu']),
            'size': np.array(h5f['size']),
            'opcode': np.array(h5f['opcode']),
            'time_diff': np.array(h5f['time_diff'])
        }
        h5f.close()

        n_sample = self.data['asu'].shape[0]
        if mode == 'train':
            self.beg = 0
            self.end = int(round(n_sample * hps.tr_ratio))
        elif mode == 'val' or mode == 'validation':
            self.beg = int(round(n_sample * hps.tr_ratio)) - hps.in_len
            self.end = int(round(n_sample * (hps.tr_ratio + hps.va_ratio)))
        elif mode == 'test':
            self.beg = int(round(n_sample * (hps.tr_ratio + hps.va_ratio))) - hps.in_len
            self.end = n_sample
        else:
            ValueError('Mode is not valid!')
        self.n_sample = self.end - self.beg
        for field in self.data:
            self.data[field] = self.data[field][self.beg:self.end]

    def get_datashape(self):
        n_asu = self.data['asu'].shape[1]
        n_size = 1
        n_opcode = self.data['opcode'].shape[1]
        n_time_diff = 1
        data_shape = {
            'n_asu': n_asu,
            'n_size': n_size,
            'n_opcode': n_opcode,
            'n_time_diff': n_time_diff
        }
        data_shape = [data_shape, sum([data_shape[i] for i in data_shape])]
        data_shape.append(np.cumsum([data_shape[0][i] for i in data_shape[0]]).astype(np.uint8))
        print(data_shape)
        return data_shape

    def get_batch(self, retry=0):
        if self.queue.empty():
            if retry > 10:
                x, y = None, None
            else:
                time.sleep(1.0)
                x, y = self.get_batch(retry+1)
        else:
            x, y = self.queue.get()
        return x, y

    def new_epoch(self):
        self.queue = Queue()
        if self.mode == 'train':
            self.read_thread = Thread(
                target=self.put_train_batch,
            )
        else:
            self.read_thread = Thread(
                target=self.put_test_batch,
            )
        self.read_thread.start()
        return

    def put_train_batch(self):
        sleep = 1.0
        n_asu = self.data['asu'].shape[1]
        n_size = 1
        n_opcode = self.data['opcode'].shape[1]
        n_time_diff = 1
        x_len = self.in_len + self.out_len - 1
        y_len = x_len

        cnt_sample = 0
        complete = False
        while True:
            if self.queue.qsize() > self.capacity:
                time.sleep(sleep)
                continue
            x = np.zeros((x_len, self.batch_size, n_asu + n_size + n_opcode + n_time_diff), np.float32)
            y = {
                'asu': np.zeros((y_len, self.batch_size, n_asu), np.int64),
                'size': np.zeros((y_len, self.batch_size, n_size), np.float32),
                'opcode': np.zeros((y_len, self.batch_size, n_opcode), np.int64),
                'time_diff': np.zeros((y_len, self.batch_size, n_time_diff), np.float32)
            }
            for cnt_inbatch in range(self.batch_size):
                x_beg = cnt_sample
                x_end = cnt_sample + x_len
                y_beg = cnt_sample + 1
                y_end = cnt_sample + 1 + y_len
                if y_end > self.n_sample:
                    complete = True
                    break
                x[:, cnt_inbatch, :n_asu] = self.data['asu'][x_beg:x_end]
                x[:, cnt_inbatch, n_asu:n_asu+n_size] = self.data['size'][x_beg:x_end]
                x[:, cnt_inbatch, n_asu+n_size:n_asu+n_size+n_opcode] = self.data['opcode'][x_beg:x_end]
                x[:, cnt_inbatch, n_asu+n_size+n_opcode:] = self.data['time_diff'][x_beg:x_end]
                y['asu'][:, cnt_inbatch, :] = self.data['asu'][y_beg:y_end]
                y['size'][:, cnt_inbatch, :] = self.data['size'][y_beg:y_end]
                y['opcode'][:, cnt_inbatch, :] = self.data['opcode'][y_beg:y_end]
                y['time_diff'][:, cnt_inbatch, :] = self.data['time_diff'][y_beg:y_end]
                cnt_sample += hps.step_size

            if complete:
                break
            self.queue.put([x, y])
        pass

    def put_test_batch(self):
        sleep = 1.0
        n_asu = self.data['asu'].shape[1]
        n_size = 1
        n_opcode = self.data['opcode'].shape[1]
        n_time_diff = 1
        x_len = self.in_len
        y_len = self.out_len

        cnt_sample = 0
        complete = False
        while True:
            if self.queue.qsize() > self.capacity:
                time.sleep(sleep)
                continue
            x = np.zeros((x_len, self.batch_size, n_asu + n_size + n_opcode + n_time_diff), np.float32)
            y = {
                'asu': np.zeros((y_len, self.batch_size, n_asu), np.float32),
                'size': np.zeros((y_len, self.batch_size, n_size), np.float32),
                'opcode': np.zeros((y_len, self.batch_size, n_opcode), np.float32),
                'time_diff': np.zeros((y_len, self.batch_size, n_time_diff), np.float32)
            }
            for cnt_inbatch in range(self.batch_size):
                x_beg = cnt_sample
                x_end = cnt_sample + x_len
                y_beg = cnt_sample + x_len
                y_end = cnt_sample + x_len + y_len
                if x_end > self.n_sample:
                    complete = True
                    break
                x[:, cnt_inbatch, :n_asu] = self.data['asu'][x_beg:x_end]
                x[:, cnt_inbatch, n_asu:n_asu+n_size] = self.data['size'][x_beg:x_end]
                x[:, cnt_inbatch, n_asu+n_size:n_asu+n_size+n_opcode] = self.data['opcode'][x_beg:x_end]
                x[:, cnt_inbatch, n_asu+n_size+n_opcode:] = self.data['time_diff'][x_beg:x_end]
                y['asu'][:, cnt_inbatch, :] = self.data['asu'][y_beg:y_end]
                y['size'][:, cnt_inbatch, :] = self.data['size'][y_beg:y_end]
                y['opcode'][:, cnt_inbatch, :] = self.data['opcode'][y_beg:y_end]
                y['time_diff'][:, cnt_inbatch, :] = self.data['time_diff'][y_beg:y_end]
                cnt_sample += y_len
            if complete:
                break
            self.queue.put([x, y])
        pass


def unscale_data(raw):
    raw['size'] = np.subtract(np.exp2(raw['size']))
    return raw