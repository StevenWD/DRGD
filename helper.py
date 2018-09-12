import numpy as np
from config.configurable import Configurable
import math
import json

class Helper(Configurable):
    def __init__(self, data_filepath, mode):
        # mode: source/target_input/target_output
        super(Helper, self).__init__('data')
        self.data = np.load(data_filepath)
        self.mode = mode
        self.maximum_step = math.ceil(self.data.shape[0]/self.get_config('train', 'batch_size'))
        self.cursor = 0

    def next_batch(self):
        batch = self.data[self.cursor: min(self.data.shape[0], self.get_config('train', 'batch_size')+self.cursor)]
        self.cursor += self.get_config('train', 'batch_size')
        if batch.shape[0] < self.get_config('train', 'batch_size'):
            supplement = self.data[: self.get_config('train', 'batch_size')-batch.shape[0]]
            batch = np.concatenate([batch, supplement], axis=0)
            self.cursor -= self.data.shape[0]

        if self.mode == 'target_input':
            char_dict = json.load(open(self.base_dir+'data/LCSTS/char_dict.json', 'r'))
            batch[:, -1] = char_dict['<START>']
            batch = np.roll(batch, shift=1, axis=1)

        return batch
