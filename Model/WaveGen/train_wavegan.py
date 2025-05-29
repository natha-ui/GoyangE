import tensorflow as tf
import numpy as np
import os
import time
from functools import reduce

from loader import decode_extract_and_batch
from wavegan import WaveGANGenerator, WaveGANDiscriminator

def train(fps, args):
    # Create dataset
    dataset = decode_extract_and_batch(
        fps,
        batch_size=args.train_batch_size,
        slice_len=args.data_slice_len,
        decode_fs=args.data_sample_rate,
        decode_num_channels=args.data_num_channels,
        decode_fast_wav=args.data_fast_wav,
        decode_parallel_calls=4,
        slice_randomize_offset=not args.data_first_slice,
        slice_first_only=args.data_first_slice,
        slice_overlap_ratio=0. if args.data_first_slice else args.data_overlap_ratio,
        slice_pad_end=args.data_first_slice or args.data_pad_end,
        repeat=True,
        shuffle=True,
        shuffle_buffer_size=4096,
        prefetch_size=args.train_batch_size * 4,
        prefetch_gpu_num=args.data_prefetch_gpu_num)
    
    dataset = dataset.map(lambda x: x[:, :, 0])  # Drop channel dim if 1D
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Initialize models
    generator = WaveGANGenerator(latent_dim=args.wavegan_latent_dim, **args.wavegan_g_kwargs)
    discriminator = WaveGANDiscriminator(**args.wavegan_d_kwargs)

    # Optimizers
    if args.wavegan_loss == 'dcgan':
        g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        d_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    elif args.wavegan_loss == 'lsgan':
        g_optimizer = tf.keras.optimizers.RMSprop(1e-4)
        d_optimizer = tf.keras.optimizers.RMSprop(1e-4)
    elif args.wavegan_loss == 'wgan':
        g_optimizer = tf.keras.optimizers.RMSprop(5e-5)
        d_optimizer = tf.keras.optimizers.RMSprop(5e-5)
    elif args.wavegan_loss == 'wgan-gp':
        g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
        d_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    else:
        raise NotImplementedError()

    summary_writer = tf.summary.create_file_writer(args.train_dir)
    global_step = 0

    @tf.function
    def train_step(x_batch):
        nonlocal global_step

        # Sample z
        z = tf.random.uniform([args.train_batch_size, args.wavegan_latent_dim], -1., 1.)

        # Train discriminator
        for _ in range(args.wavegan_disc_nupdates):
            with tf.GradientTape() as d_tape:
                G_z = generator(z, training=True)
                D_x = discriminator(x_batch, training=True)
                D_G_z = discriminator(G_z, training=True)

                if args.wavegan_loss == 'dcgan':
                    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_x, labels=tf.ones_like(D_x))
                    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_z, labels=tf.zeros_like(D_G_z))
                    d_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)
                elif args.wavegan_loss == 'lsgan':
                    d_loss = 0.5 * (tf.reduce_mean((D_x - 1)**2) + tf.reduce_mean(D_G_z**2))
                elif args.wavegan_loss == 'wgan':
                    d_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)
                elif args.wavegan_loss == 'wgan-gp':
                    alpha = tf.random.uniform([args.train_batch_size, 1, 1], 0.0, 1.0)
                    interpolates = x_batch + alpha * (G_z - x_batch)
                    with tf.GradientTape() as gp_tape:
                        gp_tape.watch(interpolates)
                        D_interp = discriminator(interpolates, training=True)
                    grads = gp_tape.gradient(D_interp, [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
                    gp = tf.reduce_mean((slopes - 1.0)**2)
                    d_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x) + 10 * gp

            d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

            if args.wavegan_loss == 'wgan':
                for var in discriminator.trainable_variables:
                    var.assign(tf.clip_by_value(var, -0.01, 0.01))

        # Train generator
        with tf.GradientTape() as g_tape:
            G_z = generator(z, training=True)
            D_G_z = discriminator(G_z, training=True)

            if args.wavegan_loss in ['dcgan', 'lsgan']:
                if args.wavegan_loss == 'dcgan':
                    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D_G_z, labels=tf.ones_like(D_G_z)))
                elif args.wavegan_loss == 'lsgan':
                    g_loss = 0.5 * tf.reduce_mean((D_G_z - 1)**2)
            elif args.wavegan_loss in ['wgan', 'wgan-gp']:
                g_loss = -tf.reduce_mean(D_G_z)

        g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('G_loss', g_loss, step=global_step)
            tf.summary.scalar('D_loss', d_loss, step=global_step)
        global_step += 1

    # Training loop
    print("Starting training...")
    for batch in dataset:
        train_step(batch)



"""
  Creates and saves a MetaGraphDef for simple inference
  Tensors:
    'samp_z_n' int32 []: Sample this many latent vectors
    'samp_z' float32 [samp_z_n, latent_dim]: Resultant latent vectors
    'z:0' float32 [None, latent_dim]: Input latent vectors
    'flat_pad:0' int32 []: Number of padding samples to use when flattening batch to a single audio file
    'G_z:0' float32 [None, slice_len, 1]: Generated outputs
    'G_z_int16:0' int16 [None, slice_len, 1]: Same as above but quantizied to 16-bit PCM samples
    'G_z_flat:0' float32 [None, 1]: Outputs flattened into single audio file
    'G_z_flat_int16:0' int16 [None, 1]: Same as above but quantized to 16-bit PCM samples
  Example usage:
    import tensorflow as tf
    tf.reset_default_graph()

    saver = tf.train.import_meta_graph('infer.meta')
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'model.ckpt-10000')

    z_n = graph.get_tensor_by_name('samp_z_n:0')
    _z = sess.run(graph.get_tensor_by_name('samp_z:0'), {z_n: 10})

    z = graph.get_tensor_by_name('G_z:0')
    _G_z = sess.run(graph.get_tensor_by_name('G_z:0'), {z: _z})
"""
import os
import numpy as np
import tensorflow as tf
from wavegan import WaveGANGenerator

def float_to_int16(x):
    x = x * 32767.0
    x = tf.clip_by_value(x, -32767.0, 32767.0)
    return tf.cast(x, tf.int16)

def infer(args):
    infer_dir = os.path.join(args.train_dir, 'infer')
    os.makedirs(infer_dir, exist_ok=True)

    # Create the generator
    generator = WaveGANGenerator(latent_dim=args.wavegan_latent_dim, **args.wavegan_g_kwargs)
    dummy_z = tf.random.uniform([1, args.wavegan_latent_dim], -1.0, 1.0)
    _ = generator(dummy_z, training=False)  # Build model

    # Optionally add post-processing filter
    if args.wavegan_genr_pp:
        pp_filter = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=args.wavegan_genr_pp_len,
            padding='same',
            use_bias=False,
            name='pp_filt'
        )

        # Run dummy data through to build
        _ = pp_filter(generator(dummy_z, training=False))
    else:
        pp_filter = None

    # Sample latent vectors
    def sample_latents(n):
        return tf.random.uniform([n, args.wavegan_latent_dim], -1.0, 1.0)

    def generate_audio(z, flat_pad=0):
        G_z = generator(z, training=False)
        if pp_filter is not None:
            G_z = pp_filter(G_z)

        nch = G_z.shape[-1]
        G_z_padded = tf.pad(G_z, [[0, 0], [0, flat_pad], [0, 0]])
        G_z_flat = tf.reshape(G_z_padded, [-1, nch])

        G_z_int16 = float_to_int16(G_z)
        G_z_flat_int16 = float_to_int16(G_z_flat)

        return G_z, G_z_int16, G_z_flat_int16

    # Save weights
    checkpoint = tf.train.Checkpoint(generator=generator)
    if pp_filter is not None:
        checkpoint.pp_filter = pp_filter
    manager = tf.train.CheckpointManager(checkpoint, directory=infer_dir, max_to_keep=1)
    manager.save()

    print(f"Generator weights saved to {infer_dir}")
    print("You can restore them with tf.train.Checkpoint and generate audio with `generate_audio(z)`.")


"""
  Generates a preview audio file every time a checkpoint is saved
"""
import os
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wavwrite
from scipy.signal import freqz

def preview(args):
    preview_dir = os.path.join(args.train_dir, 'preview')
    os.makedirs(preview_dir, exist_ok=True)

    # Load or generate latent vectors _zs
    z_fp = os.path.join(preview_dir, 'z.pkl')
    if os.path.exists(z_fp):
        with open(z_fp, 'rb') as f:
            _zs = pickle.load(f)
    else:
        _zs = tf.random.uniform([args.preview_n, args.wavegan_latent_dim], -1.0, 1.0)
        _zs = _zs.numpy()
        with open(z_fp, 'wb') as f:
            pickle.dump(_zs, f)

    # Load generator model checkpoint
    infer_dir = os.path.join(args.train_dir, 'infer')
    checkpoint_dir = infer_dir
    generator = args.generator  # Assume the generator model is passed as part of args
    pp_filter = None

    # Optionally create pp_filter layer if enabled
    if args.wavegan_genr_pp:
        pp_filter = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=args.wavegan_genr_pp_len,
            padding='same',
            use_bias=False,
            name='pp_filt'
        )
        # Build pp_filter with dummy input to initialize weights
        dummy_input = tf.zeros((1, 16384, generator.output_shape[-1]))
        _ = pp_filter(dummy_input)

    # Setup checkpoint
    checkpoint = tf.train.Checkpoint(generator=generator)
    if pp_filter is not None:
        checkpoint.pp_filter = pp_filter
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

    # Setup TensorBoard summary writer
    summary_writer = tf.summary.create_file_writer(preview_dir)

    # Helper function to generate audio and optionally apply pp_filter
    def generate_audio(z_batch, flat_pad):
        G_z = generator(z_batch, training=False)  # [batch, time, channels]
        if pp_filter is not None:
            G_z = pp_filter(G_z)
        # Pad in time dimension
        G_z_padded = tf.pad(G_z, [[0, 0], [0, flat_pad], [0, 0]])
        nch = G_z.shape[-1]
        G_z_flat = tf.reshape(G_z_padded, [-1, nch])

        def float_to_int16(x):
            x = x * 32767.0
            x = tf.clip_by_value(x, -32767.0, 32767.0)
            return tf.cast(x, tf.int16)

        G_z_int16 = float_to_int16(G_z)
        G_z_flat_int16 = float_to_int16(G_z_flat)
        return G_z, G_z_int16, G_z_flat_int16

    print("Starting preview loop...")
    last_ckpt_fp = None

    while True:
        latest_ckpt_fp = manager.latest_checkpoint
        if latest_ckpt_fp is not None and latest_ckpt_fp != last_ckpt_fp:
            print(f'Preview: {latest_ckpt_fp}')
            # Restore checkpoint
            checkpoint.restore(latest_ckpt_fp).expect_partial()

            flat_pad = int(args.data_sample_rate / 2)
            z_tensor = tf.convert_to_tensor(_zs, dtype=tf.float32)

            G_z, G_z_int16, G_z_flat_int16 = generate_audio(z_tensor, flat_pad)

            # Save WAV files for each sample
            step_num = int(latest_ckpt_fp.split('-')[-1])  # extract step from checkpoint name

            for i in range(args.preview_n):
                preview_fp = os.path.join(preview_dir, f'{str(step_num).zfill(8)}_{i}.wav')
                wavwrite(preview_fp, args.data_sample_rate, G_z_flat_int16.numpy()[i])

            # Write TensorBoard audio summary (only first sample)
            with summary_writer.as_default():
                audio = tf.expand_dims(G_z_flat_int16[0], axis=0)
                audio = tf.cast(audio, tf.float32) / 32767.0  # normalize for summary.audio
                tf.summary.audio('preview_audio', audio, sample_rate=args.data_sample_rate, step=step_num, max_outputs=1)

            # Plot and save post-processing filter frequency response if enabled
            if args.wavegan_genr_pp:
                pp_kernel = pp_filter.weights[0].numpy()[:, 0, 0]
                w, h = freqz(pp_kernel)

                plt.figure()
                plt.title('Digital filter frequency response')
                plt.plot(w, 20 * np.log10(np.abs(h)), 'b')
                plt.ylabel('Amplitude [dB]', color='b')
                plt.xlabel('Frequency [rad/sample]')

                ax2 = plt.gca().twinx()
                angles = np.unwrap(np.angle(h))
                ax2.plot(w, angles, 'g')
                ax2.set_ylabel('Angle (radians)', color='g')
                plt.grid()
                plt.axis('tight')

                pp_img_fp = os.path.join(preview_dir, f'{str(step_num).zfill(8)}_ppfilt.png')
                plt.savefig(pp_img_fp)
                plt.close()

                # Log image summary
                img = plt.imread(pp_img_fp)
                with summary_writer.as_default():
                    tf.summary.image('pp_filt', tf.expand_dims(img, axis=0), step=step_num)

            print('Done')

            last_ckpt_fp = latest_ckpt_fp

        time.sleep(1)


"""
  Computes inception score every time a checkpoint is saved
"""
import os
import time
import pickle
import numpy as np
import tensorflow as tf

def incept(args):
    incept_dir = os.path.join(args.train_dir, 'incept')
    os.makedirs(incept_dir, exist_ok=True)

    # Load or generate latent vectors _zs
    z_fp = os.path.join(incept_dir, 'z.pkl')
    if os.path.exists(z_fp):
        with open(z_fp, 'rb') as f:
            _zs = pickle.load(f)
    else:
        _zs = tf.random.uniform([args.incept_n, args.wavegan_latent_dim], -1.0, 1.0)
        _zs = _zs.numpy()
        with open(z_fp, 'wb') as f:
            pickle.dump(_zs, f)

    # Load Generator (GAN) model
    generator = args.generator  # Assume generator model (tf.keras.Model) passed in args
    checkpoint_dir = os.path.join(args.train_dir, 'infer')
    checkpoint = tf.train.Checkpoint(generator=generator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

    # Load classifier model (Inception)
    classifier = args.classifier  # Assume classifier model (tf.keras.Model) passed in args
    classifier_ckpt = args.incept_ckpt_fp
    classifier_checkpoint = tf.train.Checkpoint(classifier=classifier)
    classifier_checkpoint.restore(classifier_ckpt).expect_partial()

    summary_writer = tf.summary.create_file_writer(incept_dir)

    best_score = 0.0
    last_ckpt_fp = None

    batch_size = 100

    while True:
        latest_ckpt_fp = manager.latest_checkpoint
        if latest_ckpt_fp is not None and latest_ckpt_fp != last_ckpt_fp:
            print(f'Incept: {latest_ckpt_fp}')
            checkpoint.restore(latest_ckpt_fp).expect_partial()

            step_num = int(latest_ckpt_fp.split('-')[-1])

            # Generate fake samples in batches
            G_zs = []
            for i in range(0, args.incept_n, batch_size):
                z_batch = tf.convert_to_tensor(_zs[i:i + batch_size], dtype=tf.float32)
                generated = generator(z_batch, training=False)  # [batch, time, channels]
                # Extract first channel as in original: [:, :, 0]
                generated = generated[:, :, 0].numpy()
                G_zs.append(generated)
            G_zs = np.concatenate(G_zs, axis=0)

            # Classify generated samples in batches
            preds = []
            for i in range(0, args.incept_n, batch_size):
                batch = tf.convert_to_tensor(G_zs[i:i + batch_size], dtype=tf.float32)
                batch = tf.expand_dims(batch, -1)  # Add channel dim if classifier expects it
                pred = classifier(batch, training=False).numpy()
                preds.append(pred)
            preds = np.concatenate(preds, axis=0)

            # Compute Inception Score
            split_size = args.incept_n // args.incept_k
            incept_scores = []
            for i in range(args.incept_k):
                split = preds[i * split_size:(i + 1) * split_size]
                py = np.mean(split, axis=0, keepdims=True)
                kl = split * (np.log(split + 1e-16) - np.log(py + 1e-16))
                kl = np.mean(np.sum(kl, axis=1))
                incept_scores.append(np.exp(kl))

            incept_mean = np.mean(incept_scores)
            incept_std = np.std(incept_scores)

            # Write summaries
            with summary_writer.as_default():
                tf.summary.scalar('incept_mean', incept_mean, step=step_num)
                tf.summary.scalar('incept_std', incept_std, step=step_num)

            # Save best score checkpoint
            if incept_mean > best_score:
                checkpoint_path = os.path.join(incept_dir, 'best_score')
                checkpoint.write(checkpoint_path)
                best_score = incept_mean

            print(f'Step {step_num} done - Incept mean: {incept_mean:.4f}, std: {incept_std:.4f}')

            last_ckpt_fp = latest_ckpt_fp

        time.sleep(1)


import os
import glob
import tensorflow as tf
from tensorflow.keras import layers

# Your conv1d_transpose, WaveGANGenerator, lrelu, apply_phaseshuffle, WaveGANDiscriminator
# are assumed imported here or defined above this script

def train(filepaths, args):
    # Your train loop here using tf.GradientTape etc.
    print("Starting training with {} files...".format(len(filepaths)))
    # Example dummy training step (replace with your actual training)
    # ...
    pass

def preview(args):
    print("Preview mode")
    # Implement preview logic here
    pass

def incept(args):
    print("Inception score evaluation mode")
    # Implement inception evaluation here
    pass

def infer(args):
    print("Inference mode")
    # Implement inference here
    pass

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['train', 'preview', 'incept', 'infer'])
    parser.add_argument('train_dir', type=str, help='Training directory')

    data_args = parser.add_argument_group('Data')
    data_args.add_argument('--data_dir', type=str, help='Data directory containing *only* audio files to load')
    data_args.add_argument('--data_sample_rate', type=int, help='Number of audio samples per second')
    data_args.add_argument('--data_slice_len', type=int, choices=[16384, 32768, 65536], help='Number of audio samples per slice (maximum generation length)')
    data_args.add_argument('--data_num_channels', type=int, help='Number of audio channels to generate (for >2, must match that of data)')
    data_args.add_argument('--data_overlap_ratio', type=float, help='Overlap ratio [0, 1) between slices')
    data_args.add_argument('--data_first_slice', action='store_true', help='If set, only use the first slice each audio example')
    data_args.add_argument('--data_pad_end', action='store_true', help='If set, use zero-padded partial slices from the end of each audio file')
    data_args.add_argument('--data_normalize', action='store_true', help='If set, normalize the training examples')
    data_args.add_argument('--data_fast_wav', action='store_true', help='Use scipy decoding for WAV files (faster)')
    data_args.add_argument('--data_prefetch_gpu_num', type=int, help='If nonnegative, prefetch examples to this GPU')

    wavegan_args = parser.add_argument_group('WaveGAN')
    wavegan_args.add_argument('--wavegan_latent_dim', type=int, help='Number of dimensions of the latent space')
    wavegan_args.add_argument('--wavegan_kernel_len', type=int, help='Length of 1D filter kernels')
    wavegan_args.add_argument('--wavegan_dim', type=int, help='Dimensionality multiplier for model of G and D')
    wavegan_args.add_argument('--wavegan_batchnorm', action='store_true', help='Enable batchnorm')
    wavegan_args.add_argument('--wavegan_disc_nupdates', type=int, help='Number of discriminator updates per generator update')
    wavegan_args.add_argument('--wavegan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'], help='Which GAN loss to use')
    wavegan_args.add_argument('--wavegan_genr_upsample', type=str, choices=['zeros', 'nn'], help='Generator upsample strategy')
    wavegan_args.add_argument('--wavegan_genr_pp', action='store_true', help='If set, use post-processing filter')
    wavegan_args.add_argument('--wavegan_genr_pp_len', type=int, help='Length of post-processing filter for DCGAN')
    wavegan_args.add_argument('--wavegan_disc_phaseshuffle', type=int, help='Radius of phase shuffle operation')

    train_args = parser.add_argument_group('Train')
    train_args.add_argument('--train_batch_size', type=int, help='Batch size')
    train_args.add_argument('--train_save_secs', type=int, help='How often to save model')
    train_args.add_argument('--train_summary_secs', type=int, help='How often to report summaries')

    preview_args = parser.add_argument_group('Preview')
    preview_args.add_argument('--preview_n', type=int, help='Number of samples to preview')

    incept_args = parser.add_argument_group('Incept')
    incept_args.add_argument('--incept_metagraph_fp', type=str, help='Inference model for inception score')
    incept_args.add_argument('--incept_ckpt_fp', type=str, help='Checkpoint for inference model')
    incept_args.add_argument('--incept_n', type=int, help='Number of generated examples to test')
    incept_args.add_argument('--incept_k', type=int, help='Number of groups to test')

    parser.set_defaults(
        data_dir=None,
        data_sample_rate=16000,
        data_slice_len=16384,
        data_num_channels=1,
        data_overlap_ratio=0.,
        data_first_slice=False,
        data_pad_end=False,
        data_normalize=False,
        data_fast_wav=False,
        data_prefetch_gpu_num=0,
        wavegan_latent_dim=100,
        wavegan_kernel_len=25,
        wavegan_dim=64,
        wavegan_batchnorm=False,
        wavegan_disc_nupdates=5,
        wavegan_loss='wgan-gp',
        wavegan_genr_upsample='zeros',
        wavegan_genr_pp=False,
        wavegan_genr_pp_len=512,
        wavegan_disc_phaseshuffle=2,
        train_batch_size=128,
        train_save_secs=60*15,
        train_summary_secs=360,
        preview_n=32,
        incept_metagraph_fp='./eval/inception/infer.meta',
        incept_ckpt_fp='./eval/inception/best_acc-103005',
        incept_n=5000,
        incept_k=10)

    args = parser.parse_args()

    # Make train dir if it doesn't exist
    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)

    # Save args to a file
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k},{v}\n")

    # Set model kwargs on args
    args.wavegan_g_kwargs = {
        'slice_len': args.data_slice_len,
        'nch': args.data_num_channels,
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        'use_batchnorm': args.wavegan_batchnorm,
        'upsample': args.wavegan_genr_upsample
    }
    args.wavegan_d_kwargs = {
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        'use_batchnorm': args.wavegan_batchnorm,
        'phaseshuffle_rad': args.wavegan_disc_phaseshuffle
    }

    # Eager execution is enabled by default in TF2; no need to disable it.

    if args.mode == 'train':
        fps = glob.glob(os.path.join(args.data_dir, '*'))
        if len(fps) == 0:
            raise Exception('Did not find any audio files in specified directory')
        print(f'Found {len(fps)} audio files in specified directory')
        infer(args)
        train(fps, args)
    elif args.mode == 'preview':
        preview(args)
    elif args.mode == 'incept':
        incept(args)
    elif args.mode == 'infer':
        infer(args)
    else:
        raise NotImplementedError(f"Mode {args.mode} not implemented")
