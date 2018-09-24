import tensorflow as tf
from pydoc import locate
from config.configurable import Configurable

config = Configurable('train_op')

def create_learning_rate_decay_fn(decay_type, decay_steps, decay_rate, start_decay_at=0, stop_decay_at=1e9, min_learning_rate=None, staircase=False):
    if decay_type is None or decay_type == '':
        return None

    start_decay_at = tf.to_int32(start_decay_at)
    stop_decay_at = tf.to_int32(stop_decay_at)

    def decay_fn(learning_rate, global_step):
        global_step = tf.to_int32(global_step)

        decay_type_fn = getattr(tf.train, decay_type)
        decayed_learning_rate = decay_type_fn(
            learning_rate=learning_rate,
            global_step=tf.minimum(global_step, stop_decay_at) - start_decay_at,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name='decayed_learning_rate')

        final_lr = tf.train.piecewise_constant(
            x=global_step,
            boundaries=[start_decay_at],
            values=[learning_rate, decayed_learning_rate])

        if min_learning_rate:
            final_lr = tf.maximum(final_lr, min_learning_rate)
        return final_lr

    return decay_fn

def build_train_op(loss):
    learning_rate_decay_fn = create_learning_rate_decay_fn(
            **config.config['lr_decay'])
    
    optimizer = _create_optimizer()
    train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=config.config['learning_rate'],
            learning_rate_decay_fn=learning_rate_decay_fn,
            # learning_rate_decay_fn=None,
            clip_gradients=_clip_gradients,
            optimizer=optimizer,
            summaries=['learning_rate', 'loss', 'gradients', 'gradient_norm'])

    return train_op

def _clip_gradients(grads_and_vars):
    gradients, variables = zip(*grads_and_vars)
    clip_gradients, _ = tf.clip_by_global_norm(
            gradients, config.config['clip_gradients'])
    return list(zip(clip_gradients, variables))

def _create_optimizer():
    name = config.config['name']
    optimizer = tf.contrib.layers.OPTIMIZER_CLS_NAMES[name](
            learning_rate=config.config['learning_rate'],
            **config.config['params'])

    if config.config['sync_replicas'] > 0:
        optimizer = tf.train.SyncReplicasOptimizer(
                opt=optimizer,
                replicas_to_aggregate=config.config['sync_replicas_to_aggregate'],
                total_num_replicas=config.config['sync_replicas'])
        global_vars.SYNC_REPLICAS_OPTIMIZER = optimizer

    return optimizer


