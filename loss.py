import tensorflow as tf
import numpy as np

def KL_loss(mu_1, log_sigma_1, mu_2=np.float32(0.0), log_sigma_2=np.float32(0.0)):
    sigma_1 = tf.exp(log_sigma_1)
    sigma_2 = tf.exp(log_sigma_2)
    dist_1 = tf.distributions.Normal(loc=mu_1, scale=sigma_1)
    dist_2 = tf.distributions.Normal(loc=mu_2, scale=sigma_2)
    loss = tf.reduce_mean(tf.reduce_mean(tf.distributions.kl_divergence(dist_1, dist_2), axis=-1), axis=-1)

    return loss


def cross_entropy_sequence_loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy, axis=-1)

    return loss
