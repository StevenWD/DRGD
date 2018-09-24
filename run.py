import tensorflow as tf
import numpy as np
from model import DRGD
from train_op import build_train_op
from loss import KL_loss, cross_entropy_sequence_loss
import os
from metrics import compute_metric_score
import json

tf.logging.set_verbosity(tf.logging.INFO)

# def build_mask(target_length, max_length):
#     mask = np.zeros((target_length.shape[0], max_length), dtype='float32')
#     for i in range(target_length.shape[0]):
#         mask[i, :target_length[i]] = 1
#     return mask

def train(source_helper, target_input_helper, target_output_helper, valid_source_helper, valid_target_output_helper, char_dict):
    model = DRGD() 
    model.build()

    source_emd_weight = np.load(model.base_dir+model.get_config('data', 'source_embedding_filepath'))
    target_emd_weight = np.load(model.base_dir+model.get_config('data', 'target_embedding_filepath'))

    source_embedding = tf.get_variable(initializer=source_emd_weight, name='source_embedding')
    target_embedding = tf.get_variable(initializer=target_emd_weight, name='target_embedding')

    source_sequence = tf.nn.embedding_lookup(source_embedding, model.source_placeholder)
    target_sequence = tf.nn.embedding_lookup(target_embedding, model.target_input_placeholder)
    # epsilon = np.random.randn(model.get_config('train', 'batch_size'),
    #         model.get_config('data', 'target_max_seq_length'),
    #         model.get_config('decoder', 'variable_size'))

    encoder_output = model.encoder.encode(source_sequence, model.source_mask)
    # logit, mu, sigma = model.decoder.decode(target_sequence, encoder_output, epsilon)
    logit = model.decoder.decode(target_sequence, encoder_output, model.source_mask, model.target_mask)
    y_s = model.decoder.beam_search(encoder_output, model.blank_input_placeholder, target_embedding, model.source_mask, char_dict['<EOS>'])
    # y_s = tf.argmax(logit, axis=-1)
    # logit = model.decoder.decode(target_sequence, encoder_hidden_states, epsilon)
    # max_length = tf.constant(logit.shape[1], dtype=tf.int32)
    # mask = tf.concat([tf.ones(model.target_length), tf.zeros(max_length-model.target_length)], axis=-1)
    # logit = tf.multiply(logit, mask)
    # mu = tf.multiply(mu, mask)
    # sigma = tf.multiply(sigma, mask)
    # mask = build_mask(model.target_length, model.get_config('data', 'target_max_seq_length'))

    # y_s = tf.argmax(logit, axis=-1)
    crossent = cross_entropy_sequence_loss(logit, model.target_output_placeholder)
    # loss_1 = tf.reduce_mean(KL_loss(mu, sigma, model.target_mask), axis=0)
    # loss = tf.reduce_mean(tf.multiply(crossent, model.mask), axis=0)
    loss = tf.reduce_mean(crossent, axis=0)
    # loss = tf.reduce_mean(loss_2, axis=0)
    # tf.summary.scalar('loss', loss)
    train_op = build_train_op(loss)
    # optimizer = tf.train.GradientDescentOptimizer(0.0001)
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    # train_op = optimizer.minimize(loss)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth=True
    # saver = tf.train.Saver()

    with tf.Session(config=conf) as sess:
        writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(tf.global_variables_initializer())
        # merged_summary = tf.summary.merge_all()
        # all_var=tf.trainable_variables()
        for epoch in range(model.get_config('train', 'epoch_num')):
            for step in range(source_helper.maximum_step):
                source_input, source_mask = source_helper.next_batch()
                target_input, target_mask = target_input_helper.next_batch()
                target_output, _ = target_output_helper.next_batch()

                _, l = sess.run([train_op, loss], feed_dict={
                    model.source_placeholder: source_input,
                    model.target_input_placeholder: target_input,
                    model.target_output_placeholder: target_output,
                    model.source_mask: source_mask,
                    model.target_mask: target_mask})

                tf.logging.log_every_n(level=tf.logging.INFO, msg='epoch {}/{}, step {}/{}, loss: {}'.format(epoch, model.get_config('train', 'epoch_num'), step, source_helper.maximum_step, l), n=10)
                if step % 500 == 0:
                    final_output_all = list()
                    target_output_all = list()
                    for v_step in range(valid_source_helper.maximum_step):
                        source_input, source_mask = valid_source_helper.next_batch()
                        target_output, target_mask = valid_target_output_helper.next_batch()
                        target_length = np.count_nonzero(target_mask, axis=1)

                        blank_input = target_emd_weight[char_dict['<START>']]
                        blank_input = np.reshape(np.tile(blank_input, [model.get_config('train', 'batch_size'), 1]), (-1, model.get_config('data', 'emd_dim')))

                        final_output = sess.run(y_s, feed_dict={
                            model.source_placeholder: source_input,
                            model.source_mask: source_mask,
                            model.blank_input_placeholder: blank_input})
                        
                        final_output_all.append(final_output)
                        target_output_all.append(target_output)

                    final_output = np.concatenate(final_output_all, axis=0)
                    target_output = np.concatenate(target_output_all, axis=0)
                    # for index in range(final_output.shape[0]):
                    #     final_output[index] = final_output[index][: target_length[index]]
                    #     target_output[index] = target_output[index][: target_length[index]]

                    score = compute_metric_score(model.get_config('train', 'metric_path'), final_output, target_output, char_dict)
                    tf.logging.info('Rough metric score :\n{}'.format(score))
                    valid_source_helper.reset_cursor()
                    valid_target_output_helper.reset_cursor()
