import tensorflow as tf
from config.configurable import Configurable
from collections import namedtuple
from pydoc import locate
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq


BeamSearchDecoderState = namedtuple("BeamSearchDecoderState", 
    ("cell_state", "log_probs", "finished", "lengths")
)


class Decoder(Configurable):
    def __init__(self):
        super(Decoder, self).__init__('decoder')
        self.cell_1 = self.build_cell(self.config['cell_classname'], 'cell_1') # GRU cell 1
        self.cell_2 = self.build_cell(self.config['cell_classname'], 'cell_2') # GRU cell 2
        self.init_z = tf.zeros((self.get_config('train', 'batch_size'), self.config['variable_size'])) # initlize value for z

    def decode_onestep(self, inputs, encoder_output, state_1, state_2, z):
        batch_size = self.get_config('train', 'batch_size')
        variable_size = self.config['variable_size']
        hidden_dim = self.config['cell']['num_units']
        word_num = self.get_config('data', 'target_word_num')
        emd_dim = self.get_config('data', 'emd_dim')
        source_max_seq_length = self.get_config('data', 'source_max_seq_length')

        source_mask = tf.sequence_mask(lengths=encoder_output.attention_values_length, maxlen=source_max_seq_length, dtype=tf.bool)

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
            W_dy_zh = tf.get_variable(name='W_dy_zh', shape=[variable_size, hidden_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            W_dz_hh = tf.get_variable(name='W_dz_hh', shape=[hidden_dim, hidden_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_dy_h = tf.get_variable(name='b_dy_h', shape=[hidden_dim], dtype=tf.float32, initializer=tf.zeros_initializer())

            W_d_hy = tf.get_variable(name='W_d_hy', shape=[hidden_dim, word_num], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_d_hy = tf.get_variable(name='b_d_hy', shape=[word_num], dtype=tf.float32, initializer=tf.zeros_initializer())

            with tf.variable_scope('vae', reuse=tf.AUTO_REUSE):
                # encoder
                W_ez_yh = tf.get_variable('W_ez_yh', shape=[emd_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                W_ez_zh = tf.get_variable('W_ez_zh', shape=[variable_size, hidden_dim], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                W_ez_hh = tf.get_variable('W_ez_hh', shape=[hidden_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                b_ez_h = tf.get_variable(name='b_ez_h', shape=[hidden_dim], initializer=tf.zeros_initializer(), dtype=tf.float32)
                # mean
                W_ez_hm = tf.get_variable('W_ez_mu', shape=[hidden_dim, variable_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                b_ez_m = tf.get_variable(name='b_ez_mu', shape=[variable_size], initializer=tf.zeros_initializer(), dtype=tf.float32)
                # var
                W_h_s = tf.get_variable('W_ez_sigma', shape=[hidden_dim, variable_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                b_ez_s = tf.get_variable(name='b_ez_sigma', shape=[variable_size], initializer=tf.zeros_initializer(), dtype=tf.float32)

        last_state_1 = state_1
        _, state_1 = self.cell_1(inputs=inputs, state=state_1)
        a_ij = self.compute_attention_weight(state_1, encoder_output.attention_values, source_mask)
        c_t = tf.reduce_sum(tf.multiply(encoder_output.attention_values, a_ij), axis=1)

        _, state_2 = self.cell_2(inputs=tf.concat([c_t, inputs], axis=-1), state=state_2)
        
        h_ez_t = tf.nn.sigmoid(tf.add(tf.add_n([tf.matmul(inputs, W_ez_yh), tf.matmul(z, W_ez_zh), tf.matmul(last_state_1, W_ez_hh)]), b_ez_h)) 
        mean_t = tf.nn.xw_plus_b(h_ez_t, W_ez_hm, b_ez_m)
        var_t = tf.exp(tf.nn.xw_plus_b(h_ez_t, W_h_s, b_ez_s))
        sigma = tf.sqrt(var_t)

        eps = tf.random_normal((batch_size, variable_size))
        z_t = tf.add(mean_t, tf.multiply(sigma, eps))

        h_dy_t = tf.tanh(tf.add(tf.add_n([tf.matmul(z_t, W_dy_zh), tf.matmul(state_2, W_dz_hh)]), b_dy_h))
        logit_t = tf.nn.xw_plus_b(h_dy_t, W_d_hy, b_d_hy)

        KL_t = tf.to_float(-0.5) * tf.reduce_sum(tf.to_float(1.0)+tf.log(var_t)-tf.square(mean_t)-var_t, axis=-1)

        return state_1, state_2, z_t, logit_t, KL_t

    def decode(self, inputs, length, encoder_output):
        target_max_seq_length = self.get_config('data', 'target_max_seq_length')
        batch_size = self.get_config('train', 'batch_size')

        state_1 = tf.reduce_mean(encoder_output.outputs, axis=1)
        state_2 = tf.reduce_mean(encoder_output.outputs, axis=1)
        z = self.init_z
        logits = list()
        KL = list()
        for step in range(target_max_seq_length):
            state_1, state_2, z, logit, KL_t = self.decode_onestep(inputs[:, step, :], encoder_output, state_1, state_2, z)
            logits.append(logit)
            KL.append(KL_t)
        logits = tf.stack(logits, axis=1)
        KL = tf.stack(KL, axis=1)

        return logits, KL

    def beam_search(self, encoder_output, embedding):
        vocab_size = self.get_config('data', 'target_word_num')
        end_id = self.get_config('data', 'end_id')
        start_id = self.get_config('data', 'start_id')
        beam_width = self.config['beam_search_width']
        batch_size = self.get_config('train', 'batch_size')
        target_max_seq_length = self.get_config('data', 'target_max_seq_length')
        length_penalty_weight = self.config['length_penalty_weight']
        hidden_dim = self.config['cell']['num_units']
        batch_size_beam_width = batch_size * beam_width

        start_tokens = tf.ones([batch_size, beam_width], tf.int32) * start_id
        start_inputs = tf.nn.embedding_lookup(embedding, start_tokens)
        inputs = start_inputs

        finished = tf.one_hot(
            indices=tf.zeros(batch_size, tf.int32),
            depth=beam_width,
            on_value=False,
            off_value=True,
            dtype=tf.bool)
        log_probs = tf.one_hot(
            indices=tf.zeros(batch_size, tf.int32),
            depth=beam_width,
            on_value=tf.convert_to_tensor(0.0, tf.float32),
            off_value=tf.convert_to_tensor(-np.Inf, tf.float32),
            dtype=tf.float32)

        tile_state = tf.tile(tf.expand_dims(tf.reduce_mean(encoder_output.outputs, axis=1), 1), [1, beam_width, 1])
        tile_z = tf.tile(tf.expand_dims(self.init_z, 1), [1, beam_width, 1])

        beam_state = BeamSearchDecoderState(
            cell_state=(tile_state, tile_state, tile_z),
            log_probs=log_probs,
            finished=finished,
            lengths=tf.zeros([batch_size, beam_width], dtype=tf.int32))

        y_s = tf.zeros([batch_size, beam_width, 0], dtype=tf.int32)

        for step in range(target_max_seq_length):
            state_1, state_2, z = beam_state.cell_state

            new_state_1 = list()
            new_state_2 = list()
            new_z = list()
            logits = list()
            for search_step in range(beam_width):
                state_1_t, state_2_t, z_t, logits_t, _ = self.decode_onestep(inputs[:, search_step, :], encoder_output, state_1[:, search_step, :], state_2[:, search_step, :], z[:, search_step, :])
                new_state_1.append(state_1_t)
                new_state_2.append(state_2_t)
                new_z.append(z_t)
                logits.append(logits_t)

            state_1 = tf.stack(new_state_1, 1)
            state_2 = tf.stack(new_state_2, 1)
            z = tf.stack(new_z, 1)
            logits = tf.stack(logits, 1)

            prediction_lengths = beam_state.lengths
            previously_finished = beam_state.finished

            step_log_probs = tf.nn.log_softmax(logits, axis=-1)
            step_log_probs = self.mask_probs(step_log_probs, end_id, previously_finished)
            total_probs = tf.expand_dims(beam_state.log_probs, 2) + step_log_probs

            lengths_to_add = tf.one_hot(
                indices=tf.fill([batch_size, beam_width], end_id),
                depth=vocab_size,
                on_value=tf.to_int32(0),
                off_value=tf.to_int32(1),
                dtype=tf.int32)
            add_mask = tf.to_int32(tf.logical_not(previously_finished))
            lengths_to_add = lengths_to_add * tf.expand_dims(add_mask, 2)
            new_prediction_lengths = lengths_to_add + tf.expand_dims(prediction_lengths, 2)

            scores = self.get_scores(log_probs=total_probs,
                sequence_lengths=new_prediction_lengths,
                length_penalty_weight=length_penalty_weight)

            scores_flat = tf.reshape(scores, (batch_size, -1))
            next_beam_scores, word_indices = tf.nn.top_k(scores_flat, k=beam_width, sorted=True)
            next_beam_scores = tf.reshape(next_beam_scores, (batch_size, beam_width))
            word_indices = tf.reshape(word_indices, (batch_size, beam_width))

            next_beam_probs = tf.batch_gather(tf.reshape(total_probs, (batch_size, -1)), word_indices)

            next_word_ids = tf.mod(word_indices, vocab_size)
            next_beam_ids = tf.to_int32(tf.div(word_indices, vocab_size))

            previously_finished = tf.batch_gather(previously_finished, next_beam_ids)
            next_finished = tf.logical_or(previously_finished,
                tf.equal(next_word_ids, end_id))

            lengths_to_add = tf.to_int32(tf.logical_not(previously_finished))
            next_prediction_len = tf.batch_gather(beam_state.lengths, next_beam_ids)
            next_prediction_len = next_prediction_len + lengths_to_add

            next_state_1 = tf.batch_gather(state_1, next_beam_ids)
            next_state_2 = tf.batch_gather(state_2, next_beam_ids)
            next_z = tf.batch_gather(z, next_beam_ids)

            beam_state = BeamSearchDecoderState(
                cell_state=(next_state_1, next_state_2, next_z),
                log_probs=next_beam_probs,
                lengths=next_prediction_len,
                finished=next_finished)

            inputs = tf.cond(
                tf.reduce_all(next_finished),
                lambda: start_inputs, lambda: tf.nn.embedding_lookup(embedding, next_word_ids))

            y_s = tf.concat([tf.batch_gather(y_s, next_beam_ids), tf.expand_dims(next_word_ids, axis=2)], axis=2)
        return y_s[:, 0, :]


    def mask_probs(self, probs, eos_token, finished):
        vocab_size = probs.shape[-1]
        finished_row = tf.one_hot(
            indices=eos_token,
            depth=vocab_size,
            dtype=tf.float32,
            on_value=tf.convert_to_tensor(0., dtype=tf.float32),
            off_value=tf.float32.min)

        finished_probs = tf.tile(
            tf.reshape(finished_row, [1, 1, -1]),
            tf.concat([finished.shape, [1]], 0))

        finished_mask = tf.tile(
            tf.expand_dims(finished, 2), [1, 1, vocab_size])

        return tf.where(finished_mask, finished_probs, probs)

    def get_scores(self, log_probs, sequence_lengths, length_penalty_weight):
        length_penalty_ = self.length_penalty(sequence_lengths=sequence_lengths, penalty_factor=length_penalty_weight)

        return log_probs / length_penalty_


    def length_penalty(self, sequence_lengths, penalty_factor):
        penalty_factor = tf.to_float(penalty_factor)
        return tf.div((tf.to_float(5.0) + tf.to_float(sequence_lengths))**penalty_factor, (tf.to_float(1.0 + 5.0))**penalty_factor)

    def build_cell(self, cell_classname, cell_name):
        cell_class = locate(cell_classname)
        return cell_class(num_units=self.config['cell']['num_units'],
                name=cell_name,
                **self.config['cell']['cell_params'])

    def compute_attention_weight(self, state, hidden_states, source_mask):
        encoder_hidden_hid = self.get_config('encoder', 'cell')['num_units']
        decoder_hidden_hid = self.config['cell']['num_units']

        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE) as scope:
            W_d_hh = tf.get_variable(name='W_d_hh', shape=[encoder_hidden_hid*2, decoder_hidden_hid], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            W_e_hh = tf.get_variable(name='W_e_hh', shape=[decoder_hidden_hid, decoder_hidden_hid], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b_a = tf.get_variable(name='b_a', shape=[decoder_hidden_hid], initializer=tf.zeros_initializer(), dtype=tf.float32)
            v = tf.get_variable(name='v', shape=[decoder_hidden_hid], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

        r_1 = tf.einsum('ijk,kl->ijl', hidden_states, W_e_hh)
        r_2 = tf.expand_dims(tf.matmul(state, W_d_hh), axis=1)
        a = tf.add(r_1, r_2)
        t = tf.tanh(tf.add(a, b_a))
        e_ij = tf.reduce_sum(tf.multiply(v, t), axis=-1)

        mask_value = tf.log(tf.to_float(0.0))
        e_ij_mask = mask_value * tf.ones_like(e_ij)

        e_ij = tf.where(source_mask, e_ij, e_ij_mask) # replace the attention weight on <PAD> token with -inf, then after softmax it can be zero

        a_ij = tf.expand_dims(tf.nn.softmax(e_ij, axis=1), axis=-1)
        return a_ij
