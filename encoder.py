import tensorflow as tf
from config.configurable import Configurable
from collections import namedtuple
from pydoc import locate
from copy import deepcopy


class Encoder(Configurable):
    def __init__(self):
        super(Encoder, self).__init__('encoder')
    
    def encode(self, inputs):
        # inputs shape: (batch, max_seq_length, emd_dim)

        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE) as scope:
            cell_fw = self.build_cell(self.config['cell_classname'], 'cell_fw')
            cell_bw = self.build_cell(self.config['cell_classname'], 'cell_bw')

        batch_size = inputs.shape[0].value
        initial_state = tf.Variable(cell_fw.zero_state(batch_size, tf.float32), trainable=False)
        
        fw_state = initial_state
        bw_state = initial_state
        hidden_states = list()
        for step in range(self.get_config(section='data', key='source_max_seq_length')):
            _, fw_state = cell_fw(inputs[:, step, :], fw_state)
            _, bw_state = cell_bw(inputs[:, step, :], bw_state)
            state = tf.expand_dims(tf.add(fw_state, bw_state), axis=0)
            hidden_states.append(state)
        
        hidden_states = tf.concat(values=hidden_states, axis=0)
        # print(hidden_states.shape)

        return hidden_states

        # _, output_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
        #         cell_bw=cell_bw,
        #         inputs=inputs,
        #         time_major=True,
        #         initial_state_fw=initial_state,
        #         initial_state_bw=initial_state)

        # print(output_state)
        # return output_state

    def build_cell(self, cell_classname, cell_name):
        cell_class = locate(cell_classname)
        return cell_class(num_units=self.config['cell']['num_units'],
                name=cell_name,
                **self.config['cell']['cell_params'])
