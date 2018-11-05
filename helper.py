import numpy as np
from config.configurable import Configurable
import math
import json

class Helper(Configurable):
    def __init__(self, data_filepath, length_filepath, mode):
        # mode: source/target_input/target_output
        super(Helper, self).__init__('data')
        self.data = np.load(data_filepath)
        self.length = np.load(length_filepath)
        self.mode = mode
        self.maximum_step = math.ceil(self.data.shape[0]/self.get_config('train', 'batch_size'))
        self.cursor = 0

    def next_batch(self):
        # generate a mini-batch data
        # if mode is 'target_input', we need to add a <START> token at the begining of target sequence
        
        batch_size = self.get_config('train', 'batch_size')
        batch = self.data[self.cursor: min(self.data.shape[0], batch_size+self.cursor)]
        length = self.length[self.cursor: min(self.length.shape[0], batch_size+self.cursor)]
        self.cursor += batch_size
        if batch.shape[0] < batch_size:
            supplement = self.data[: batch_size-batch.shape[0]]
            supplement_length = self.length[: batch_size-batch.shape[0]]
            batch = np.concatenate([batch, supplement], axis=0)
            length = np.concatenate([length, supplement_length], axis=0)
            self.cursor -= self.data.shape[0]

        if self.mode == 'target_input':
            batch[:, -1] = self.config['start_id']
            batch = np.roll(batch, shift=1, axis=1)

        return batch, length

    def reset_cursor(self):
        self.cursor = 0
