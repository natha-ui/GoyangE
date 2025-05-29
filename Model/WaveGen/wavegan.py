import tensorflow as tf
from tensorflow.keras import layers, Model


def conv1d_transpose(inputs, filters, kernel_width, stride=4, padding='same', upsample='zeros'):
    if upsample == 'zeros':
        x = tf.expand_dims(inputs, axis=1)  # [B, 1, W, C]
        x = layers.Conv2DTranspose(filters, (1, kernel_width), strides=(1, stride), padding=padding)(x)
        return tf.squeeze(x, axis=1)
    elif upsample == 'nn':
        x = tf.expand_dims(inputs, axis=1)  # [B, 1, W, C]
        x = tf.image.resize(x, [1, inputs.shape[1] * stride], method='nearest')
        x = tf.squeeze(x, axis=1)
        return layers.Conv1D(filters, kernel_width, strides=1, padding=padding)(x)
    else:
        raise NotImplementedError(f"Unsupported upsample mode: {upsample}")


def WaveGANGenerator(z, slice_len=16384, nch=1, kernel_len=25, dim=64, use_batchnorm=False, upsample='zeros'):
    assert slice_len in [16384, 32768, 65536], "Unsupported slice_len"

    dim_mul = 16 if slice_len == 16384 else 32
    x = layers.Dense(16 * dim * dim_mul)(z)
    x = tf.reshape(x, [-1, 16, dim * dim_mul])
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    dim_mul //= 2

    for _ in range(4):
        x = conv1d_transpose(x, dim * dim_mul, kernel_len, 4, upsample=upsample)
        if use_batchnorm:
            x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        dim_mul //= 2

    if slice_len == 16384:
        x = conv1d_transpose(x, nch, kernel_len, 4, upsample=upsample)
    elif slice_len == 32768:
        x = conv1d_transpose(x, dim, kernel_len, 4, upsample=upsample)
        if use_batchnorm:
            x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = conv1d_transpose(x, nch, kernel_len, 2, upsample=upsample)
    elif slice_len == 65536:
        x = conv1d_transpose(x, dim, kernel_len, 4, upsample=upsample)
        if use_batchnorm:
            x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = conv1d_transpose(x, nch, kernel_len, 4, upsample=upsample)

    return tf.nn.tanh(x)


def lrelu(inputs, alpha=0.2):
    return tf.nn.leaky_relu(inputs, alpha=alpha)


def apply_phaseshuffle(x, rad, pad_type='REFLECT'):
    input_shape = tf.shape(x)
    b, x_len, nch = input_shape[0], input_shape[1], input_shape[2]

    phase = tf.random.uniform([], -rad, rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)
    x = x[:, phase_start:phase_start + x_len]
    x.set_shape([None, x_len, nch])
    return x


def WaveGANDiscriminator(x, kernel_len=25, dim=64, use_batchnorm=False, phaseshuffle_rad=0):
    def maybe_bn(y): return layers.BatchNormalization()(y) if use_batchnorm else y
    def maybe_ps(y): return apply_phaseshuffle(y, phaseshuffle_rad) if phaseshuffle_rad > 0 else y

    x = layers.Conv1D(dim, kernel_len, strides=4, padding='same')(x)
    x = lrelu(x)
    x = maybe_ps(x)

    x = layers.Conv1D(dim * 2, kernel_len, strides=4, padding='same')(x)
    x = maybe_bn(x)
    x = lrelu(x)
    x = maybe_ps(x)

    x = layers.Conv1D(dim * 4, kernel_len, strides=4, padding='same')(x)
    x = maybe_bn(x)
    x = lrelu(x)
    x = maybe_ps(x)

    x = layers.Conv1D(dim * 8, kernel_len, strides=4, padding='same')(x)
    x = maybe_bn(x)
    x = lrelu(x)
    x = maybe_ps(x)

    x = layers.Conv1D(dim * 16, kernel_len, strides=4, padding='same')(x)
    x = maybe_bn(x)
    x = lrelu(x)

    final_length = x.shape[1] * 4
    if final_length == 32768:
        x = layers.Conv1D(dim * 32, kernel_len, strides=2, padding='same')(x)
        x = maybe_bn(x)
        x = lrelu(x)
    elif final_length == 65536:
        x = layers.Conv1D(dim * 32, kernel_len, strides=4, padding='same')(x)
        x = maybe_bn(x)
        x = lrelu(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    return tf.squeeze(x, axis=1)
