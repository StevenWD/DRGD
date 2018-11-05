import tensorflow as tf
from config.configurable import Configurable
from collections import namedtuple
from pydoc import locate

EncoderOutput = namedtuple(
    "EncoderOutput",
    "outputs final_state attention_values attention_values_length")

class Encoder(Configurable):
    def __init__(self):
        super(Encoder, self).__init__('encoder')
        self.cell_fw = self.build_cell(self.config['cell_classname'], 'cell_fw')
        self.cell_bw = self.build_cell(self.config['cell_classname'], 'cell_bw')

    def encode(self, inputs, length):
        # inputs shape: (batch_size, max_len, emd_dim)
        # mask shape: (batch_size, max_len)
        (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cell_fw,
            cell_bw=self.cell_bw,
            inputs=inputs,
            sequence_length=length,
            scope='encoder',
            dtype=tf.float32)

        outputs = tf.concat([output_fw, output_bw], axis=2)
        final_state = tf.concat([output_state_fw, output_state_bw], axis=1)

        return EncoderOutput(outputs=outputs,
            final_state=final_state,
            attention_values=outputs,
            attention_values_length=length)

    def build_cell(self, cell_classname, cell_name):
        cell_class = locate(cell_classname)
        return cell_class(num_units=self.config['cell']['num_units'],
                name=cell_name,
                **self.config['cell']['cell_params'])
