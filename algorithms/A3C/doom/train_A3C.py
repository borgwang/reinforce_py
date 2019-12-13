import argparse
import multiprocessing
import os
import threading
import time

import tensorflow as tf

from env_doom import Doom
from net import Net
from utils import print_net_params_number
from worker import Worker


def main(args):
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    tf.reset_default_graph()

    global_ep = tf.Variable(
        0, dtype=tf.int32, name='global_ep', trainable=False)
    
    env = Doom(visiable=False)
    Net(env.state_dim, env.action_dim, 'global', None)
    num_workers = args.parallel
    workers = []

    # create workers
    for i in range(num_workers):
        w = Worker(i, Doom(), global_ep, args)
        workers.append(w)

    print('%d workers in total.\n' % num_workers)
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if args.model_path is not None:
            print('Loading model...')
            ckpt = tf.train.get_checkpoint_state(args.model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Initializing a new model...')
            sess.run(tf.global_variables_initializer())
        print_net_params_number()

        # Start work process for each worker in a separated thread
        worker_threads = []
        for w in workers:
            run_fn = lambda: w.run(sess, coord, saver)
            t = threading.Thread(target=run_fn)
            t.start()
            time.sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)


if __name__ == '__main__':
    # ignore warnings by tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', default=None,
        help='Whether to use a saved model. (*None|model path)')
    parser.add_argument(
        '--save_path', default='/tmp/a3c_doom/model/',
        help='Path to save a model during training.')
    parser.add_argument(
        '--save_every', default=50, help='Interval of saving model')
    parser.add_argument(
        '--max_ep_len', default=300, help='Max episode steps')
    parser.add_argument(
        '--max_ep', default=3000, help='Max training episode')
    parser.add_argument(
        '--parallel', default=multiprocessing.cpu_count(),
        help='Number of parallel threads')
    main(parser.parse_args())
