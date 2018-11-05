import tensorflow as tf
import numpy as np

def cross_entropy_sequence_loss(logits, labels):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels, reduction=tf.losses.Reduction.NONE)
    loss = tf.reduce_mean(cross_entropy, axis=1)

    return loss
