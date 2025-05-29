import glob
import os
import shutil
import sys
import time

import tensorflow as tf

def wait_for_first_checkpoint(train_dir):
    while tf.train.latest_checkpoint(train_dir) is None:
        print('Waiting for first checkpoint...')
        time.sleep(1)

def backup_checkpoints(train_dir, backup_dir, interval_sec):
    while True:
        latest_ckpt = tf.train.latest_checkpoint(train_dir)
        time.sleep(2)  # Ensure all checkpoint files are flushed

        for fp in glob.glob(latest_ckpt + '*'):
            name = os.path.basename(fp)
            backup_fp = os.path.join(backup_dir, name)
            print(f'{fp} -> {backup_fp}')
            shutil.copyfile(fp, backup_fp)

        print('-' * 80)
        time.sleep(interval_sec)

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <train_dir> <n_minutes>")
        sys.exit(1)

    train_dir = sys.argv[1]
    nmin = float(sys.argv[2])
    nsec = int(nmin * 60)

    backup_dir = os.path.join(train_dir, 'backup')
    os.makedirs(backup_dir, exist_ok=True)

    wait_for_first_checkpoint(train_dir)
    backup_checkpoints(train_dir, backup_dir, nsec)

if __name__ == '__main__':
    main()
