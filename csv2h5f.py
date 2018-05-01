import pandas as pd
import h5py
import numpy as np
import random


import hps


def csv2h5f(src_path, dst_path):
    src = pd.read_csv(src_path)
    n_sample = src.shape[0]
    n_asu = np.max(src['asu']) + 1
    asu = np.zeros((n_sample, n_asu), np.bool)

    size = np.zeros((n_sample, 1), np.float32)

    opcode = np.zeros((n_sample, 2), np.bool) # 0st for read and 1st for write

    time_diff = np.zeros((n_sample, 1), np.float32)

    for row in src.iterrows():
        index = row[0]
        data = row[1]

        asu[index][data['asu']] = True
        size[index][0] = np.log2(data['size'] / hps.block_size + 1)
        if data['opcode'] == 'r':
            opcode[index][0] = True
        elif data['opcode'] == 'w':
            opcode[index][1] = True
        else:
            print('Index: %d  Warning: neither write or read!' % index)
        if index == 0:
            time_diff[index][0] = data['timestamp']
        else:
            time_diff[index][0] = data['timestamp'] - src['timestamp'][index-1]

        if random.uniform(0.0, 1.0) > 0.001:
            continue
        print('Total: %d, complete %8d' % (n_sample, index))

    dst = h5py.File(dst_path, 'w')
    dst.create_dataset('asu', data=asu, compression='lzf')
    dst.create_dataset('size', data=size, compression='lzf')
    dst.create_dataset('opcode', data=opcode, compression='lzf')
    dst.create_dataset('time_diff', data=time_diff, compression='lzf')
    dst.close()
    pass


def add_header_for_csv(src_path):
    line = ','.join(hps.src_fields) + '\n'
    with open(src_path, 'r+') as f:
        content = f.read()
        f.seek(0)
        f.write(line + content)
    return


if __name__ == '__main__':
    src_path = './data/Financial1.csv'
    dst_path = './data/Financial1.h5'
    # add_header_for_csv(src_path)
    csv2h5f(src_path, dst_path)
