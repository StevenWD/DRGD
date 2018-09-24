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

        if self.mode == 'source':
            batch = np.transpose(batch)
            mask = np.zeros((batch_size, self.config['source_max_seq_length']), dtype=np.float32)

        if self.mode == 'target_input':
            char_dict = json.load(open(self.base_dir+'data/LCSTS/target_char_dict.json', 'r'))
            batch[:, -1] = char_dict['<START>']
            batch = np.roll(batch, shift=1, axis=1)

        if 'target' in self.mode:
            mask = np.zeros((batch_size, self.config['target_max_seq_length']), dtype=np.float32)

        for index in range(batch_size):
            mask[index, :length[index]] = 1

        return batch, mask

    def reset_cursor(self):
        self.cursor = 0
