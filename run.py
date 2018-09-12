import tensorflow as tf
import numpy as np
from model import DRGD
from train_op import build_train_op
from loss import KL_loss, cross_entropy_sequence_loss
import os
from metrics import compute_metric_score
import json

tf.logging.set_verbosity(tf.logging.INFO)

def train(source_helper, target_input_helper, target_output_helper, valid_source_helper, valid_target_output_helper, char_dict):
    model = DRGD() 
    model.build()

    emd_weight = np.load(model.get_config('data', 'embedding_filepath'))

    embedding = tf.get_variable(initializer=emd_weight, name='embedding')

    source_sequence = tf.nn.embedding_lookup(embedding, model.source_placeholder)
    target_sequence = tf.nn.embedding_lookup(embedding, model.target_input_placeholder)
    epsilon = np.random.randn(model.get_config('train', 'batch_size'),
            model.get_config('data', 'target_max_seq_length'),
            model.get_config('decoder', 'variable_size'))

    encoder_hidden_states = model.encoder.encode(source_sequence)
    logit, mu, sigma = model.decoder.decode(target_sequence, encoder_hidden_states, epsilon)
    y_s = model.decoder.beam_search(encoder_hidden_states, model.blank_input_placeholder, embedding)
    # logit = model.decoder.decode(target_sequence, encoder_hidden_states, epsilon)
    loss_1 = KL_loss(mu, sigma)
    loss_2 = cross_entropy_sequence_loss(logit, model.target_output_placeholder)
    loss = tf.reduce_mean(tf.add(loss_1, loss_2), axis=0)
    # loss = tf.reduce_mean(loss_2, axis=0)
    tf.summary.scalar('loss', loss)
    train_op = build_train_op(loss)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth=True
    saver = tf.train.Saver()

    with tf.Session(config=conf) as sess:
        writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(tf.global_variables_initializer())
        merged_summary = tf.summary.merge_all()
        # all_var=tf.global_variables()
        # for var in all_var:
        #     print(var.name)
        for epoch in range(model.get_config('train', 'epoch_num')):
            for step in range(source_helper.maximum_step):
                source_input = source_helper.next_batch()
                target_input = target_input_helper.next_batch()
                target_output = target_output_helper.next_batch()
                
                _, l, ms = sess.run([train_op, loss, merged_summary], feed_dict={
                    model.source_placeholder: source_input,
                    model.target_input_placeholder: target_input,
                    model.target_output_placeholder: target_output})
                # print('Epoch {}/{}, Step {}/{}, Loss: {}'.format(epoch, model.get_config('train', 'epoch_num'), step, source_helper.maximum_step, l))

                tf.logging.log_every_n(level=tf.logging.INFO, msg='epoch {}/{}, step {}/{}, loss: {}'.format(epoch, model.get_config('train', 'epoch_num'), step, source_helper.maximum_step, l), n=10)
                writer.add_summary(ms, step)
                if step % 100 == 0:
                    # v_loss = 0
                    final_output = list()
                    target_output = list()
                    for v_step in range(valid_source_helper.maximum_step):
                        source_input = valid_source_helper.next_batch()
                        # target_input = valid_target_input_helper.next_batch()
                        target_output_onestep = valid_target_output_helper.next_batch()

                        # ehs = sess.run([encoder_hidden_states], feed_dict={
                            # model.source_placeholder: source_input})
                            # model.target_input_placeholder: target_input,
                            # model.target_output_placeholder: target_output})
                        blank_input = emd_weight[char_dict['<START>']]
                        blank_input = np.reshape(np.tile(blank_input, [model.get_config('train', 'batch_size'), 1]), (-1, model.get_config('data', 'emd_dim')))
                        # print(blank_input.shape)
                        final_output_onestep = sess.run(y_s, feed_dict={
                            model.blank_input_placeholder: blank_input,
                            model.source_placeholder: source_input})
                        final_output.append(final_output_onestep)
                        target_output.append(target_output_onestep)

                    final_output = np.concatenate(final_output, axis=0)
                    target_output = np.concatenate(target_output, axis=0)

                    score = compute_metric_score(model.get_config('train', 'metric_path'), final_output, target_output, char_dict)
                    tf.logging.info('Rough metric score :\n{}'.format(score))

                    save_path = saver.save(sess, 'dumps/'+model.name+'_'+str(step/100)+'.ckpt')

