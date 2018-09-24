import tensorflow as tf
from config.configurable import Configurable
from collections import namedtuple
from pydoc import locate
from copy import deepcopy


class Encoder(Configurable):
    def __init__(self):
        super(Encoder, self).__init__('encoder')
        self.cell_fw = self.build_cell(self.config['cell_classname'], 'cell_fw')
        self.cell_bw = self.build_cell(self.config['cell_classname'], 'cell_bw')

    def encode(self, inputs, mask):
        # inputs shape: (batch, max_seq_length, emd_dim)
        
        batch_size = inputs.shape[1].value
        
        initial_state = self.cell_fw.zero_state(batch_size, tf.float32)
        
        fw_state = initial_state
        bw_state = initial_state
        hidden_states = list()
        for step in range(self.get_config(section='data', key='source_max_seq_length')):
            _, fw_state = self.cell_fw(inputs[step], fw_state)
            _, bw_state = self.cell_bw(inputs[step], bw_state)
            fw_state = tf.multiply(fw_state, tf.expand_dims(mask[:, step], 1))
            bw_state = tf.multiply(bw_state, tf.expand_dims(mask[:, step], 1))
            state = tf.add(fw_state, bw_state)
            hidden_states.append(state)
        
        hidden_states = tf.stack(values=hidden_states, axis=0)

        return hidden_states 

    def build_cell(self, cell_classname, cell_name):
        cell_class = locate(cell_classname)
        return cell_class(num_units=self.config['cell']['num_units'],
                name=cell_name,
                **self.config['cell']['cell_params'])
