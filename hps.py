import os

# data
data_path = './data/dataset.h5'
src_fields = ['asu', 'lba', 'size', 'opcode', 'timestamp']
dst_fields = ['asu', 'size', 'opcode', 'time_diff']
block_size = 512


# rnn
batch_size = 1024
in_len = 50
out_len = 10
n_cell = 4
n_unit = 128
dropout = 0.5
model_path = './model'
if not os.path.exists(model_path):
    os.makedirs(model_path)


# train
tr_ratio = 0.7
va_ratio = 0.1
te_ratio = 1.0 - tr_ratio - va_ratio
optimizer = 'Adam'
lr = 1e-2
lr_decay = 0.98
print_step = 100
step_size = 3
n_epoch = 200
early_stopping = 100
clip_norm = 5.0

debug = None #50000
