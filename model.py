import tensorflow as tf
from pydoc import locate
from config.configurable import Configurable
from encoder import Encoder
from decoder import Decoder
from train_op import build_train_op

class DRGD(Configurable):
    def __init__(self, mode=tf.contrib.learn.ModeKeys.TRAIN, name='DRGD'):
        super(DRGD, self).__init__('model')
        self.mode = mode
        self.name = name

    def build(self):
        with tf.name_scope('placeholders') as scope:
            self.source_placeholder = tf.placeholder(dtype=tf.int32, shape=(self.get_config('train', 'batch_size'), self.get_config('data', 'source_max_seq_length')), name='source_placeholder')
            self.target_input_placeholder = tf.placeholder(dtype=tf.int32, shape=(self.get_config('train', 'batch_size'), self.get_config('data', 'target_max_seq_length')), name='target_input_placeholder')
            self.target_output_placeholder = tf.placeholder(dtype=tf.int32, shape=(self.get_config('train', 'batch_size'), self.get_config('data', 'target_max_seq_length')), name='target_output_placeholder')
            self.source_length = tf.placeholder(dtype=tf.int32, shape=(self.get_config('train', 'batch_size')))
            self.target_length = tf.placeholder(dtype=tf.int32, shape=(self.get_config('train', 'batch_size')))

        with tf.name_scope('encoder') as scope:
            self.encoder = Encoder()

        with tf.name_scope('decoder') as scope:
            self.decoder = Decoder()
