import numpy as np
import tensorflow as tf
import librosa
from scipy.io.wavfile import read as wavread

# Decode audio file into a standard format (float32, [time, 1, channels])
def decode_audio(fp, fs=None, num_channels=1, normalize=False, fast_wav=False):
    fp = fp.numpy().decode('utf-8')  # Convert Tensor to string path

    if fast_wav:
        _fs, _wav = wavread(fp)
        if fs is not None and fs != _fs:
            raise NotImplementedError('Scipy cannot resample audio.')
        if _wav.dtype == np.int16:
            _wav = _wav.astype(np.float32) / 32768.
        elif _wav.dtype == np.float32:
            _wav = np.copy(_wav)
        else:
            raise NotImplementedError('Unsupported WAV dtype.')
    else:
        _wav, _fs = librosa.load(fp, sr=fs, mono=False)
        if _wav.ndim == 2:
            _wav = np.swapaxes(_wav, 0, 1)

    if _wav.ndim == 1:
        nsamps = _wav.shape[0]
        nch = 1
    else:
        nsamps, nch = _wav.shape
    _wav = np.reshape(_wav, [nsamps, 1, nch])

    if nch != num_channels:
        if num_channels == 1:
            _wav = np.mean(_wav, 2, keepdims=True)
        elif nch == 1 and num_channels == 2:
            _wav = np.concatenate([_wav, _wav], axis=2)
        else:
            raise ValueError('Channel mismatch.')

    if normalize:
        factor = np.max(np.abs(_wav))
        if factor > 0:
            _wav /= factor

    return _wav.astype(np.float32)

# Wrapper for tf.py_function
def tf_decode_audio(fp, fs, num_channels, normalize, fast_wav):
    return tf.py_function(
        lambda path: decode_audio(path, fs, num_channels, normalize, fast_wav),
        inp=[fp],
        Tout=tf.float32
    )

def decode_extract_and_batch(
    fps,
    batch_size,
    slice_len,
    decode_fs,
    decode_num_channels,
    decode_normalize=True,
    decode_fast_wav=False,
    decode_parallel_calls=tf.data.AUTOTUNE,
    slice_randomize_offset=False,
    slice_first_only=False,
    slice_overlap_ratio=0,
    slice_pad_end=False,
    repeat=False,
    shuffle=False,
    shuffle_buffer_size=None,
    prefetch_size=tf.data.AUTOTUNE,
    prefetch_gpu_num=None):

    dataset = tf.data.Dataset.from_tensor_slices(fps)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size or len(fps))

    if repeat:
        dataset = dataset.repeat()

    def _decode_map(fp):
        audio = tf_decode_audio(fp, decode_fs, decode_num_channels, decode_normalize, decode_fast_wav)
        audio.set_shape([None, 1, decode_num_channels])
        return audio

    dataset = dataset.map(_decode_map, num_parallel_calls=decode_parallel_calls)

    def _slice(audio):
        slice_hop = int(round(slice_len * (1. - slice_overlap_ratio)) + 1e-4)
        if slice_hop < 1:
            raise ValueError('Overlap ratio too high')

        if slice_randomize_offset:
            start = tf.random.uniform([], maxval=slice_len, dtype=tf.int32)
            audio = audio[start:]

        slices = tf.signal.frame(
            audio,
            frame_length=slice_len,
            frame_step=slice_hop,
            pad_end=slice_pad_end,
            pad_value=0.0,
            axis=0
        )

        if slice_first_only:
            slices = slices[:1]

        return tf.data.Dataset.from_tensor_slices(slices)

    dataset = dataset.flat_map(_slice)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)

    return dataset
