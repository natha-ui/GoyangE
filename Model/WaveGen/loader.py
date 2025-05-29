from scipy.io.wavfile import read as wavread
import numpy as np
import tensorflow as tf
import sys

def decode_audio(fp, fs=None, num_channels=1, normalize=False, fast_wav=False):
  if fast_wav:
    _fs, _wav = wavread(fp)
    if fs is not None and fs != _fs:
      raise NotImplementedError('Scipy cannot resample audio.')
    if _wav.dtype == np.int16:
      _wav = _wav.astype(np.float32) / 32768.
    elif _wav.dtype == np.float32:
      _wav = np.copy(_wav)
    else:
      raise NotImplementedError('Scipy cannot process atypical WAV files.')
  else:
    import librosa
    _wav, _fs = librosa.core.load(fp, sr=fs, mono=False)
    if _wav.ndim == 2:
      _wav = np.swapaxes(_wav, 0, 1)

  assert _wav.dtype == np.float32

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
      raise ValueError('Number of audio channels not equal to num specified')

  if normalize:
    factor = np.max(np.abs(_wav))
    if factor > 0:
      _wav /= factor

  return _wav

def decode_extract_and_batch(
    fps,
    batch_size,
    slice_len,
    decode_fs,
    decode_num_channels,
    decode_normalize=True,
    decode_fast_wav=False,
    decode_parallel_calls=1,
    slice_randomize_offset=False,
    slice_first_only=False,
    slice_overlap_ratio=0,
    slice_pad_end=False,
    repeat=False,
    shuffle=False,
    shuffle_buffer_size=None,
    prefetch_size=None,
    prefetch_gpu_num=None):

  dataset = tf.data.Dataset.from_tensor_slices(fps)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=len(fps))

  if repeat:
    dataset = dataset.repeat()

  def _decode_audio_shaped(fp):
    def _decode_audio_closure(_fp):
      return decode_audio(
        _fp,
        fs=decode_fs,
        num_channels=decode_num_channels,
        normalize=decode_normalize,
        fast_wav=decode_fast_wav)

    audio = tf.py_function(
        _decode_audio_closure,
        [fp],
        tf.float32)
    audio.set_shape([None, 1, decode_num_channels])
    return audio

  dataset = dataset.map(
      _decode_audio_shaped,
      num_parallel_calls=decode_parallel_calls)

  def _slice(audio):
    if slice_overlap_ratio < 0:
      raise ValueError('Overlap ratio must be greater than 0')
    slice_hop = int(round(slice_len * (1. - slice_overlap_ratio)) + 1e-4)
    if slice_hop < 1:
      raise ValueError('Overlap ratio too high')

    if slice_randomize_offset:
      start = tf.random.uniform([], maxval=slice_len, dtype=tf.int32)
      audio = audio[start:]

    audio_slices = tf.signal.frame(
        audio,
        slice_len,
        slice_hop,
        pad_end=slice_pad_end,
        pad_value=0,
        axis=0)

    if slice_first_only:
      audio_slices = audio_slices[:1]

    return audio_slices

  def _slice_dataset_wrapper(audio):
    audio_slices = _slice(audio)
    return tf.data.Dataset.from_tensor_slices(audio_slices)

  dataset = dataset.flat_map(_slice_dataset_wrapper)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

  dataset = dataset.batch(batch_size, drop_remainder=True)

  if prefetch_size is not None:
    dataset = dataset.prefetch(prefetch_size)
    if prefetch_gpu_num is not None and prefetch_gpu_num >= 0:
      dataset = dataset.apply(
          tf.data.experimental.prefetch_to_device(
            f'/device:GPU:{prefetch_gpu_num}'))

  iterator = iter(dataset)
  return next(iterator)
