import numpy as np
import tensorflow as tf
from loss import KL_loss, cross_entropy_sequence_loss
from model import DRGD
from train_op import build_train_op

model = DRGD()
model.build()

encoder = model.encoder
decoder = model.decoder
emd_weight = np.load(model.get_config('data', 'embedding_filepath'))
embedding = tf.get_variable(initializer=emd_weight, name='embedding')

source_sequence = np.random.randn(256, model.get_config('data', 'source_max_seq_length'), model.get_config('data', 'emd_dim'))
target_sequence = np.random.randn(256, model.get_config('data', 'target_max_seq_length'), model.get_config('data', 'emd_dim'))
target_input = np.random.randint(low=0, high=10000, size=(256, model.get_config('data', 'target_max_seq_length')))
epsilon = np.random.randn(256, model.get_config('data', 'target_max_seq_length'), model.get_config('decoder', 'variable_size'))

blank_input = np.random.randn(256, model.get_config('data', 'emd_dim')).astype('float32')

source_tensor = tf.convert_to_tensor(source_sequence, dtype=tf.float32)
target_tensor = tf.convert_to_tensor(target_sequence, dtype=tf.float32)
epsilon_tensor = tf.convert_to_tensor(epsilon, dtype=tf.float32)

hidden_states = encoder.encode(source_tensor)
# logit, mu, sigma = decoder.decode(target_tensor, hidden_states, epsilon_tensor)
output = decoder.beam_search(hidden_states, blank_input, embedding)
# print(y, mu, sigma)
print(logit)

# kl_loss = KL_loss(mu, sigma)
# ce_loss = cross_entropy_sequence_loss(logit, target_input)
# loss = tf.add(kl_loss, ce_loss)
# 
# train_op = build_train_op(loss)
