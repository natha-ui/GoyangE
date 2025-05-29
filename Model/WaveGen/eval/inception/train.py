from __future__ import print_function

if __name__ == '__main__':
  import glob
  import os
  import shutil
  import sys
  import time

  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()  # Required to run TF1.x code in TF2.x environments

  train_dir, nmin = sys.argv[1:3]
  nsec = int(float(nmin) * 60.)

  backup_dir = os.path.join(train_dir, 'backup')

  if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

  while tf.train.latest_checkpoint(train_dir) is None:
    print('Waiting for first checkpoint')
    time.sleep(1)

  while True:
    latest_ckpt = tf.train.latest_checkpoint(train_dir)

    # Sleep for two seconds in case file flushing
    time.sleep(2)

    for fp in glob.glob(latest_ckpt + '*'):
      _, name = os.path.split(fp)
      backup_fp = os.path.join(backup_dir, name)
      print(f'{fp}->{backup_fp}')
      shutil.copyfile(fp, backup_fp)
    print('-' * 80)

    # Sleep for the specified interval
    time.sleep(nsec)
