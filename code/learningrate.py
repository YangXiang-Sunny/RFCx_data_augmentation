import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def warmup_cosine_decay(global_step,
                        learning_rate_base,
                        total_steps,
                        warmup_learning_rate=0.0,
                        warmup_steps= 0,
                        hold_base_rate_steps=0):
    """Defines a learning curve with warmup and cosine decay
        
    Args:
        global_step: (int) the current training step
        learning_rate_base: base learning rate, i.e. the max learning rate
        total_steps: total training steps to be applied
        warmup_learning_rate: learning rate to start warming up from, i.e. initial rate
        warmup_steps: number of steps to increase to the base learning rate
        hold_base_rate_steps: number of steps to hold the base learning rate
        
    Returns:
        The learning rate at the specified global step
        
    """
    
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
        np.pi *
        (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
        ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = tf.where(
          global_step > warmup_steps + hold_base_rate_steps,
          learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                         'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * tf.cast(global_step,
                                    tf.float32) + warmup_learning_rate
        learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                               learning_rate)
    return tf.where(global_step > total_steps, 0.0, learning_rate,
                    name='learning_rate')

class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    """Keras callback class for applying a learning rate schedule
    
    """
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = warmup_cosine_decay(global_step=self.global_step,
                                 learning_rate_base=self.learning_rate_base,
                                 total_steps=self.total_steps,
                                 warmup_learning_rate=self.warmup_learning_rate,
                                 warmup_steps=self.warmup_steps,
                                 hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr.numpy()))