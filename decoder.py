import tensorflow as tf
from config.configurable import Configurable
from pydoc import locate

class Decoder(Configurable):
    def __init__(self, mode):
        super(Decoder, self).__init__('decoder')
        self.mode = mode

    def decode(self, inputs, encoder_hidden_states, epsilon):
        # inputs shape: (batch_size, target_max_seq_length, emd_dim)
        # encoder_hidden_states shape: (source_max_seq_length, batch_size, cell.output_size)
        # epsilon shape: (batch_size, target_max_seq_length, variable_size)

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
            cell_1 = self.build_cell(self.config['cell_classname'], 'cell_1')
            cell_2 = self.build_cell(self.config['cell_classname'], 'cell_2')
            W_att = tf.get_variable(name='W_att', shape=[self.config['cell']['num_units']*2, self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            b_att = tf.get_variable(name='b_att', shape=[self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            W_dy_zh = tf.get_variable(name='W_dy_zh', shape=[self.config['variable_size'], self.config['cell']['num_units']], dtype=tf.float32, initializer=tf.zeros_initializer())
            W_dz_hh = tf.get_variable(name='W_dz_hh', shape=[self.config['cell']['num_units'], self.config['cell']['num_units']], dtype=tf.float32, initializer=tf.zeros_initializer())
            b_dy_h = tf.get_variable(name='b_dy_h', shape=[self.config['cell']['num_units']], dtype=tf.float32, initializer=tf.zeros_initializer())


            W_d_hy = tf.get_variable(name='W_d_hy', shape=[self.config['cell']['num_units'], self.get_config('data', 'target_word_num')], dtype=tf.float32, initializer=tf.zeros_initializer())
            b_d_hy = tf.get_variable(name='b_d_hy', shape=[self.get_config('data', 'target_word_num')], dtype=tf.float32, initializer=tf.zeros_initializer())

        initial_state = tf.reduce_mean(input_tensor=encoder_hidden_states,
                axis=0)
        # z = tf.random_normal(shape=[self.get_config('train', 'batch_size'), self.config['variable_size']], dtype=tf.float32)
        z = tf.random_normal(shape=[inputs.get_shape()[0].value, self.config['variable_size']], dtype=tf.float32)
        state_1 = initial_state
        state_2 = initial_state

        logit_s = list()
        mu_s = list()
        sigma_s = list()
        for step in range(self.get_config(section='data', key='target_max_seq_length')):
            # print(inputs[:, step, :])
            output_1, state_1 = cell_1(inputs=inputs[:, step, :], state=state_1)
            a_ij = self.compute_attention_weight(state_1, encoder_hidden_states)
            c_t = tf.reduce_sum(tf.multiply(encoder_hidden_states, a_ij), axis=0)
            t = tf.add(tf.matmul(tf.concat([c_t, state_2], axis=1), W_att), b_att)
            h = tf.tanh(t)
            output_2, state_2 = cell_2(inputs=inputs[:, step, :], state=h)
            # print(output_2.shape)
            # _, state_2 = cell_2(inputs=inputs[:, step, :], state=state_2)
            z, mu, sigma = self.vae(inputs[:, step, :], z, state_2, epsilon[:, step, :])
            h_y = tf.tanh(tf.add(tf.add(tf.matmul(z, W_dy_zh), tf.matmul(h, W_dz_hh)), b_dy_h))
            # logit = tf.expand_dims(output_2, axis=1)
            logit = tf.expand_dims(tf.add(tf.matmul(state_2, W_d_hy), b_d_hy), axis=1)
            # print(logit.shape)
            # y = tf.reshape(tf.argmax(p, axis=-1), [-1, 1])
            mu = tf.expand_dims(mu, axis=1)
            sigma = tf.expand_dims(sigma, axis=1)
            # y_s.append(y)
            logit_s.append(logit)
            mu_s.append(mu)
            sigma_s.append(sigma)
        return tf.concat(logit_s, axis=1), tf.concat(mu_s, axis=1), tf.concat(sigma_s, axis=1)
        # return tf.concat(logit_s, axis=1)

    def build_cell(self, cell_classname, cell_name):
        cell_class = locate(cell_classname)
        return cell_class(num_units=self.config['cell']['num_units'],
                name=cell_name,
                **self.config['cell']['cell_params'])

    def beam_search(self, encoder_hidden_states, blank_input, embedding):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
            cell_1 = self.build_cell(self.config['cell_classname'], 'cell_1')
            cell_2 = self.build_cell(self.config['cell_classname'], 'cell_2')
            W_att = tf.get_variable(name='W_att', shape=[self.config['cell']['num_units']*2, self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            b_att = tf.get_variable(name='b_att', shape=[self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            W_dy_zh = tf.get_variable(name='W_dy_zh', shape=[self.config['variable_size'], self.config['cell']['num_units']], dtype=tf.float32, initializer=tf.zeros_initializer())
            W_dz_hh = tf.get_variable(name='W_dz_hh', shape=[self.config['cell']['num_units'], self.config['cell']['num_units']], dtype=tf.float32, initializer=tf.zeros_initializer())
            b_dy_h = tf.get_variable(name='b_dy_h', shape=[self.config['cell']['num_units']], dtype=tf.float32, initializer=tf.zeros_initializer())


            W_d_hy = tf.get_variable(name='W_d_hy', shape=[self.config['cell']['num_units'], self.get_config('data', 'target_word_num')], dtype=tf.float32, initializer=tf.zeros_initializer())
            b_d_hy = tf.get_variable(name='b_d_hy', shape=[self.get_config('data', 'target_word_num')], dtype=tf.float32, initializer=tf.zeros_initializer())

        initial_state = tf.reduce_mean(input_tensor=encoder_hidden_states, axis=0)
        
        state_1 = initial_state
        state_2 = initial_state

        last_output = list()
        for i in range(self.config['beam_search_width']):
            last_output.append(tf.expand_dims(blank_input, axis=1))
        last_output = tf.concat(last_output, axis=1)
        
        y_s = list()
        for step in range(self.get_config(section='data', key='target_max_seq_length')):

            local_value_index = list()
            for search_step in range(self.config['beam_search_width']):
                _, state_1 = cell_1(inputs=last_output[:, search_step, :], state=state_1)
                a_ij = self.compute_attention_weight(state_1, encoder_hidden_states)
                c_t = tf.reduce_sum(tf.multiply(encoder_hidden_states, a_ij), axis=0)
                t = tf.add(tf.matmul(tf.concat([c_t, state_2], axis=1), W_att), b_att)
                h = tf.tanh(t)
                _, state_2 = cell_2(inputs=last_output[:, search_step, :], state=h)
            
                logit = tf.add(tf.matmul(state_2, W_d_hy), b_d_hy)
                value, index = tf.nn.top_k(input=logit, k=self.config['beam_search_width'])
                local_value_index.append(tf.concat(values=[tf.expand_dims(input=value, axis=1), tf.cast(tf.expand_dims(input=index, axis=1), dtype=tf.float32)], axis=1))
           
            local_value_index = tf.concat(local_value_index, axis=2)
            _, top_value_index = tf.nn.top_k(input=local_value_index, k=self.config['beam_search_width'])
            top_index = tf.squeeze(top_value_index[:, 0, :])
            last_output = tf.nn.embedding_lookup(embedding, top_index)
            y_s.append(tf.expand_dims(top_index[:, 0], axis=1))
        y_s = tf.concat(y_s, axis=1)
        return y_s
 

    def compute_attention_weight(self, state, hidden_states):
        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE) as scope:
            W_d = tf.get_variable(name='W_d', shape=[self.config['cell']['num_units'], self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            W_e = tf.get_variable(name='W_e', shape=[self.config['cell']['num_units'], self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            b_a = tf.get_variable(name='b_a', shape=[self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            v = tf.get_variable(name='v', shape=[self.config['cell']['num_units'], 1], initializer=tf.zeros_initializer(), dtype=tf.float32)

        r_1 = tf.einsum('ijk,kl->ijl', hidden_states, W_e) # 100*256*500, 500*500
        r_2 = tf.matmul(state, W_d) # 256*500, 500*500
        a_1 = tf.add(r_1, r_2) # 100*256*500
        a_2 = tf.add(a_1, b_a) # 100*256*500
        t = tf.tanh(a_2)
        e_ij = tf.squeeze(tf.einsum('ijk,kl->ijl', t, v))
        a_ij = tf.expand_dims(tf.nn.softmax(e_ij, axis=0), axis=-1)
        return a_ij

    def vae(self, y_p, z_p, h_p, eps):
        with tf.variable_scope('vae', reuse=tf.AUTO_REUSE) as scope:
            W_ez_yh = tf.get_variable('W_ez_yh', shape=[self.get_config('data', 'emd_dim'), self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            W_ez_zh = tf.get_variable('W_ez_zh', shape=[self.config['variable_size'], self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            W_ez_hh = tf.get_variable('W_ez_hh', shape=[self.config['cell']['num_units'], self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            b_ez_h = tf.get_variable(name='b_ez_h', shape=[self.config['cell']['num_units']], initializer=tf.zeros_initializer(), dtype=tf.float32)
            
            with tf.variable_scope('mu', reuse=tf.AUTO_REUSE) as scope:
                W_ez_mu = tf.get_variable('W_ez_mu', shape=[self.config['cell']['num_units'], self.config['variable_size']], initializer=tf.zeros_initializer(), dtype=tf.float32)
                b_ez_mu = tf.get_variable(name='b_ez_mu', shape=[self.config['variable_size']], initializer=tf.zeros_initializer(), dtype=tf.float32)

            with tf.variable_scope('sigma', reuse=tf.AUTO_REUSE) as scope:
                W_ez_sigma = tf.get_variable('W_ez_sigma', shape=[self.config['cell']['num_units'], self.config['variable_size']], initializer=tf.zeros_initializer(), dtype=tf.float32)
                b_ez_sigma = tf.get_variable(name='b_ez_sigma', shape=[self.config['variable_size']], initializer=tf.zeros_initializer(), dtype=tf.float32)

        yh = tf.squeeze(tf.matmul(y_p, W_ez_yh))
        zh = tf.matmul(z_p, W_ez_zh)
        hh = tf.matmul(h_p, W_ez_hh)
        h_ez = tf.sigmoid(tf.add(tf.add_n([yh, zh, hh]), b_ez_h))

        mu = tf.add(tf.matmul(h_ez, W_ez_mu), b_ez_mu)
        sigma = tf.add(tf.matmul(h_ez, W_ez_sigma), b_ez_sigma)

        z = tf.add(mu, tf.multiply(sigma, eps))

        return z, mu, sigma

