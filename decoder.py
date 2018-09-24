import tensorflow as tf
from config.configurable import Configurable
from pydoc import locate
import numpy as np

def batch_gather(tensor, indices):
    """Gather in batch from a tensor of arbitrary size.

    In pseudocode this module will produce the following:
    output[i] = tf.gather(tensor[i], indices[i])

    Args:
      tensor: Tensor of arbitrary size.
      indices: Vector of indices.
    Returns:
      output: A tensor of gathered values.
    """
    shape = list(tensor.shape)
    flat_first = tf.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
    indices = tf.convert_to_tensor(indices)
    offset_shape = [shape[0]] + [1] * (indices.shape.ndims - 1)
    offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
    output = tf.gather(flat_first, indices + offset)
    return output

class Decoder(Configurable):
    def __init__(self, mode):
        super(Decoder, self).__init__('decoder')
        self.mode = mode
        self.cell_1 = self.build_cell(self.config['cell_classname'], 'cell_1')
        self.cell_2 = self.build_cell(self.config['cell_classname'], 'cell_2')

    # def decode(self, inputs, encoder_hidden_states, epsilon):
    def decode(self, inputs, encoder_hidden_states, source_mask, target_mask):
        # inputs shape: (batch_size, target_max_seq_length, emd_dim)
        # encoder_hidden_states shape: (source_max_seq_length, batch_size, cell.output_size)
        # epsilon shape: (batch_size, target_max_seq_length, variable_size)
        # source_mask: (batch_size, source_max_seq_length)
        # target_mask: (batch_size, target_max_seq_length)

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
            W_att = tf.get_variable(name='W_att', shape=[self.config['cell']['num_units']*2, self.config['cell']['num_units']], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b_att = tf.get_variable(name='b_att', shape=[self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            W_dy_zh = tf.get_variable(name='W_dy_zh', shape=[self.config['variable_size'], self.config['cell']['num_units']], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            W_dz_hh = tf.get_variable(name='W_dz_hh', shape=[self.config['cell']['num_units'], self.config['cell']['num_units']], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_dy_h = tf.get_variable(name='b_dy_h', shape=[self.config['cell']['num_units']], dtype=tf.float32, initializer=tf.zeros_initializer())


            W_d_hy = tf.get_variable(name='W_d_hy', shape=[self.config['cell']['num_units'], self.get_config('data', 'target_word_num')], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_d_hy = tf.get_variable(name='b_d_hy', shape=[self.get_config('data', 'target_word_num')], dtype=tf.float32, initializer=tf.zeros_initializer())

        initial_state = tf.reduce_mean(input_tensor=encoder_hidden_states, axis=0)
        # initial_state = encoder_hidden_states
        # z = tf.random_normal(shape=[self.get_config('train', 'batch_size'), self.config['variable_size']], dtype=tf.float32)
        # z = tf.random_normal(shape=[inputs.get_shape()[0].value, self.config['variable_size']], dtype=tf.float32)
        state_1 = initial_state
        batch_size = inputs.shape[0]

        logit_s = list()
        # mu_s = list()
        # sigma_s = list()
        for step in range(self.get_config(section='data', key='target_max_seq_length')):
            _, new_state_1 = self.cell_1(inputs=inputs[:, step, :], state=state_1)
            state_1 = tf.add(tf.multiply(new_state_1, tf.expand_dims(target_mask[:, step], -1)), tf.multiply(state_1, 1.0-tf.expand_dims(target_mask[:, step], -1)))
            a_ij = self.compute_attention_weight(state_1, encoder_hidden_states, source_mask)
            c_t = tf.reduce_sum(tf.multiply(encoder_hidden_states, a_ij), axis=0)
            _, new_state_2 = self.cell_2(inputs=c_t, state=state_1)
            state_2 = tf.add(tf.multiply(new_state_2, tf.expand_dims(target_mask[:, step], -1)), tf.multiply(state_1, 1.0-tf.expand_dims(target_mask[:, step], -1)))
            # _, state_2 = cell_2(inputs=inputs[:, step, :], state=state_2)
            # z, mu, sigma = self.vae(inputs[:, step, :], z, state_2, epsilon[:, step, :])
            # h_y = tf.tanh(tf.add(tf.add(tf.matmul(z, W_dy_zh), tf.matmul(h, W_dz_hh)), b_dy_h))
            # logit = tf.expand_dims(output_2, axis=1)
            logit = tf.add(tf.matmul(state_2, W_d_hy), b_d_hy)
            # # print(logit.shape)
            # # y = tf.reshape(tf.argmax(p, axis=-1), [-1, 1])
            logit_s.append(tf.expand_dims(logit, axis=1))
            # mu_s.append(tf.expand_dims(mu, axis=1))
            # sigma_s.append(tf.expand_dims(sigma, axis=1))
        return tf.concat(logit_s, axis=1)
        # return tf.concat(logit_s, axis=1), tf.concat(mu_s, axis=1), tf.concat(sigma_s, axis=1)

    def build_cell(self, cell_classname, cell_name):
        cell_class = locate(cell_classname)
        return cell_class(num_units=self.config['cell']['num_units'],
                name=cell_name,
                **self.config['cell']['cell_params'])

    def beam_search(self, encoder_hidden_states, blank_input, embedding, source_mask, end_id):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
            W_att = tf.get_variable(name='W_att', shape=[self.config['cell']['num_units']*2, self.config['cell']['num_units']], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b_att = tf.get_variable(name='b_att', shape=[self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            W_dy_zh = tf.get_variable(name='W_dy_zh', shape=[self.config['variable_size'], self.config['cell']['num_units']], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            W_dz_hh = tf.get_variable(name='W_dz_hh', shape=[self.config['cell']['num_units'], self.config['cell']['num_units']], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_dy_h = tf.get_variable(name='b_dy_h', shape=[self.config['cell']['num_units']], dtype=tf.float32, initializer=tf.zeros_initializer())


            W_d_hy = tf.get_variable(name='W_d_hy', shape=[self.config['cell']['num_units'], self.get_config('data', 'target_word_num')], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_d_hy = tf.get_variable(name='b_d_hy', shape=[self.get_config('data', 'target_word_num')], dtype=tf.float32, initializer=tf.zeros_initializer())

        initial_state = tf.reduce_mean(input_tensor=encoder_hidden_states, axis=0)
        batch_size = initial_state.shape[0]
        sum_log_probs = tf.reshape(tf.squeeze(tf.log([[1.] + [0.] * (self.config['beam_search_width'] - 1)])), (1, -1, 1))

        state_1 = tf.stack([initial_state for i in range(self.config['beam_search_width'])], axis=1)
        # state_2 = tf.stack([initial_state for i in range(self.config['beam_search_width'])], axis=1)

        last_output = tf.stack([blank_input for i in range(self.config['beam_search_width'])], axis=1)
        target_mask = tf.ones([batch_size, self.config['beam_search_width']], dtype=tf.float32)

        y_s = list()
        for step in range(self.get_config(section='data', key='target_max_seq_length')):

            beam_state_1 = list()
            beam_state_2 = list()
            output = list()
            for search_step in range(self.config['beam_search_width']):
                _, new_state_1 = self.cell_1(inputs=last_output[:, search_step, :], state=state_1[:, search_step, :])
                new_state_1 = tf.add(tf.multiply(new_state_1, tf.expand_dims(target_mask[:, search_step], -1)), tf.multiply(state_1[:, search_step, :], 1.0-tf.expand_dims(target_mask[:, search_step], -1)))
                a_ij = self.compute_attention_weight(new_state_1, encoder_hidden_states, source_mask)
                c_t = tf.reduce_sum(tf.multiply(encoder_hidden_states, a_ij), axis=0)
                _, new_state_2 = self.cell_2(inputs=c_t, state=new_state_1)
                new_state_2 = tf.add(tf.multiply(new_state_2, tf.expand_dims(target_mask[:, search_step], -1)), tf.multiply(new_state_1, 1.0-tf.expand_dims(target_mask[:, search_step], -1)))
            
                logit = tf.add(tf.matmul(new_state_2, W_d_hy), b_d_hy)
                probs = tf.nn.log_softmax(logit)
                output.append(probs)
                beam_state_1.append(new_state_1)
                beam_state_2.append(new_state_2)

            output = tf.stack(output, axis=1)
            beam_state_1 = tf.stack(beam_state_1, axis=1)
            beam_state_2 = tf.stack(beam_state_2, axis=1)

            sum_log_probs = tf.add(sum_log_probs, tf.multiply(output, tf.expand_dims(target_mask, -1)))
            log_probs = tf.reshape(sum_log_probs, (batch_size, -1))

            value, index = tf.nn.top_k(input=log_probs, k=self.config['beam_search_width'], sorted=True)
            sum_log_probs = tf.expand_dims(value, axis=-1)

            num_classes = tf.multiply(tf.ones_like(index), self.get_config('data', 'target_word_num'))
            top_index = tf.mod(index, num_classes)
            top_state = tf.floor_div(index, num_classes)
            last_output = tf.nn.embedding_lookup(embedding, top_index)

            state_1 = batch_gather(beam_state_1, top_state)
            state_2 = batch_gather(beam_state_2, top_state)
            target_mask = tf.to_float(tf.not_equal(top_index, end_id))
            y_s.append(top_index[:, 0])
        y_s = tf.stack(y_s, axis=1)
        return y_s
 

    def compute_attention_weight(self, state, hidden_states, source_mask):
        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE) as scope:
            W_d = tf.get_variable(name='W_d', shape=[self.config['cell']['num_units'], self.config['cell']['num_units']], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            W_e = tf.get_variable(name='W_e', shape=[self.config['cell']['num_units'], self.config['cell']['num_units']], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b_a = tf.get_variable(name='b_a', shape=[self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            v = tf.get_variable(name='v', shape=[self.config['cell']['num_units'], 1], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

        r_1 = tf.einsum('ijk,kl->ijl', hidden_states, W_e) # 121*256*500, 500*500
        r_2 = tf.matmul(state, W_d) # 256*500, 500*500 = 256*500
        a_1 = tf.add(r_1, r_2) # 121*256*500
        a_2 = tf.add(a_1, b_a) # 121*256*500
        t = tf.multiply(tf.tanh(a_2), tf.expand_dims(tf.transpose(source_mask), axis=-1)) # 121*256*500
        e_ij = tf.squeeze(tf.einsum('ijk,kl->ijl', t, v)) # 121*256
        e_ij = tf.multiply(e_ij, tf.transpose(source_mask))
        a_ij = tf.expand_dims(tf.nn.softmax(e_ij, axis=0), axis=-1)
        return a_ij

    def vae(self, y_p, z_p, h_p, eps):
        with tf.variable_scope('vae', reuse=tf.AUTO_REUSE) as scope:
            W_ez_yh = tf.get_variable('W_ez_yh', shape=[self.get_config('data', 'emd_dim'), self.config['cell']['num_units']], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            W_ez_zh = tf.get_variable('W_ez_zh', shape=[self.config['variable_size'], self.config['cell']['num_units']], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            W_ez_hh = tf.get_variable('W_ez_hh', shape=[self.config['cell']['num_units'], self.config['cell']['num_units']], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b_ez_h = tf.get_variable(name='b_ez_h', shape=[self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            
            with tf.variable_scope('mu', reuse=tf.AUTO_REUSE) as scope:
                W_ez_mu = tf.get_variable('W_ez_mu', shape=[self.config['cell']['num_units'], self.config['variable_size']], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                b_ez_mu = tf.get_variable(name='b_ez_mu', shape=[self.config['variable_size']], initializer=tf.zeros_initializer(), dtype=tf.float32)

            with tf.variable_scope('sigma', reuse=tf.AUTO_REUSE) as scope:
                W_ez_sigma = tf.get_variable('W_ez_sigma', shape=[self.config['cell']['num_units'], self.config['variable_size']], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                b_ez_sigma = tf.get_variable(name='b_ez_sigma', shape=[self.config['variable_size']], initializer=tf.zeros_initializer(), dtype=tf.float32)

        yh = tf.squeeze(tf.matmul(y_p, W_ez_yh)) # 1*350, 350*500 = 1*500->500
        zh = tf.squeeze(tf.matmul(z_p, W_ez_zh)) # 1*500, 500*500 = 1*500->500
        hh = tf.squeeze(tf.matmul(h_p, W_ez_hh)) # 1*500, 500*500 = 1*500->500
        h_ez = tf.sigmoid(tf.add(tf.add_n([yh, zh, hh]), b_ez_h))

        mu = tf.add(tf.squeeze(tf.matmul(h_ez, W_ez_mu)), b_ez_mu)
        sigma = tf.add(tf.squeeze(tf.matmul(h_ez, W_ez_sigma)), b_ez_sigma)

        z = tf.add(mu, tf.multiply(sigma, eps))

        return z, mu, sigma

