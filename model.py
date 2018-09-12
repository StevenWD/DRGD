import tensorflow as tf
from pydoc import locate
from config.configurable import Configurable
from encoder import Encoder
from decoder import Decoder
from loss import KL_loss, cross_entropy_sequence_loss
from train_op import build_train_op

class DRGD(Configurable):
    def __init__(self, mode=tf.contrib.learn.ModeKeys.TRAIN, name='DRGD'):
        super(DRGD, self).__init__('model')
        self.mode = mode
        self.name = name

    def build(self):
        with tf.device('/gpu:1'):
            with tf.name_scope('input') as scope:
                self.source_placeholder = tf.placeholder(dtype=tf.int32, shape=(self.get_config('train', 'batch_size'), self.get_config('data', 'source_max_seq_length')), name='source_placeholder')
                self.target_input_placeholder = tf.placeholder(dtype=tf.int32, shape=(self.get_config('train', 'batch_size'), self.get_config('data', 'target_max_seq_length')), name='target_input_placeholder')
                self.target_output_placeholder = tf.placeholder(dtype=tf.int32, shape=(self.get_config('train', 'batch_size'), self.get_config('data', 'target_max_seq_length')), name='target_output_placeholder')
                self.blank_input_placeholder = tf.placeholder(dtype=tf.float32, shape=(self.get_config('train', 'batch_size'), self.get_config('data', 'emd_dim')), name='blank_input_placeholder')

        with tf.device('/gpu:0'):
            with tf.name_scope('encoder') as scope:
                self.encoder = Encoder()

            with tf.name_scope('decoder') as scope:
                self.decoder = Decoder(self.mode)
