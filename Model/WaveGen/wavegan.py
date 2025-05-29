import tensorflow as tf
from tensorflow.keras import layers, models


def conv1d_transpose(inputs, filters, kernel_width, stride=4, padding='same', upsample='zeros'):
    if upsample == 'zeros':
        x = tf.expand_dims(inputs, axis=1)
        x = layers.Conv2DTranspose(filters, (1, kernel_width), strides=(1, stride), padding=padding)(x)
        return x[:, 0]
    elif upsample == 'nn':
        _, w, nch = inputs.shape.as_list()
        x = tf.expand_dims(inputs, axis=1)
        x = tf.image.resize(x, [1, w * stride], method='nearest')
        x = x[:, 0]
        return layers.Conv1D(filters, kernel_width, strides=1, padding=padding)(x)
    else:
        raise NotImplementedError


"""
  Input: [None, 100]
  Output: [None, slice_len, 1]
"""
def WaveGANGenerator(z, slice_len=16384, nch=1, kernel_len=25, dim=64, use_batchnorm=False, upsample='zeros', train=False):
    assert slice_len in [16384, 32768, 65536]
    dim_mul = 16 if slice_len == 16384 else 32

    batchnorm = lambda x: layers.BatchNormalization()(x) if use_batchnorm else x

    x = layers.Dense(16 * dim * dim_mul)(z)
    x = tf.reshape(x, [-1, 16, dim * dim_mul])
    x = batchnorm(x)
    x = tf.nn.relu(x)
    dim_mul //= 2

    for i in range(4):
        x = conv1d_transpose(x, dim * dim_mul, kernel_len, 4, upsample=upsample)
        x = batchnorm(x)
        x = tf.nn.relu(x)
        dim_mul //= 2

    if slice_len == 16384:
        x = conv1d_transpose(x, nch, kernel_len, 4, upsample=upsample)
        x = tf.nn.tanh(x)
    elif slice_len == 32768:
        x = conv1d_transpose(x, dim, kernel_len, 4, upsample=upsample)
        x = batchnorm(x)
        x = tf.nn.relu(x)
        x = conv1d_transpose(x, nch, kernel_len, 2, upsample=upsample)
        x = tf.nn.tanh(x)
    elif slice_len == 65536:
        x = conv1d_transpose(x, dim, kernel_len, 4, upsample=upsample)
        x = batchnorm(x)
        x = tf.nn.relu(x)
        x = conv1d_transpose(x, nch, kernel_len, 4, upsample=upsample)
        x = tf.nn.tanh(x)

    return x


def lrelu(inputs, alpha=0.2):
  return tf.nn.leaky_relu(inputs, alpha=alpha)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.shape.as_list()

  phase = tf.random_uniform([], -rad, rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)
  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])
  return x


"""
  Input: [None, slice_len, nch]
  Output: [None] (linear output)
"""
def WaveGANDiscriminator(x, kernel_len=25, dim=64, use_batchnorm=False, phaseshuffle_rad=0):
    batchnorm = lambda x: layers.BatchNormalization()(x) if use_batchnorm else x
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad) if phaseshuffle_rad > 0 else x

    x = layers.Conv1D(dim, kernel_len, strides=4, padding='same')(x)
    x = lrelu(x)
    x = phaseshuffle(x)

    x = layers.Conv1D(dim * 2, kernel_len, strides=4, padding='same')(x)
    x = batchnorm(x)
    x = lrelu(x)
    x = phaseshuffle(x)

    x = layers.Conv1D(dim * 4, kernel_len, strides=4, padding='same')(x)
    x = batchnorm(x)
    x = lrelu(x)
    x = phaseshuffle(x)

    x = layers.Conv1D(dim * 8, kernel_len, strides=4, padding='same')(x)
    x = batchnorm(x)
    x = lrelu(x)
    x = phaseshuffle(x)

    x = layers.Conv1D(dim * 16, kernel_len, strides=4, padding='same')(x)
    x = batchnorm(x)
    x = lrelu(x)

    if x.shape[1] * 4 == 32768:
        x = layers.Conv1D(dim * 32, kernel_len, strides=2, padding='same')(x)
        x = batchnorm(x)
        x = lrelu(x)
    elif x.shape[1] * 4 == 65536:
        x = layers.Conv1D(dim * 32, kernel_len, strides=4, padding='same')(x)
        x = batchnorm(x)
        x = lrelu(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    return tf.squeeze(x, axis=1)

