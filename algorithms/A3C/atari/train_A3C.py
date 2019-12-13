import argparse
import itertools
import os
import threading
import time

import tensorflow as tf

from atari_env import A_DIM
from atari_env import S_DIM
from atari_env import make_env
from evaluate import Evaluate
from net import Net
from utils import print_params_nums
from worker import Worker


def main(args):
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    summary_writer = tf.summary.FileWriter(os.path.join(args.save_path, 'log'))
    global_steps_counter = itertools.count()  # thread-safe

    global_net = Net(S_DIM, A_DIM, 'global', args)
    num_workers = args.threads
    workers = []

    # create workers
    for i in range(1, num_workers + 1):
        worker_summary_writer = summary_writer if i == 0 else None
        worker = Worker(i, make_env(args), global_steps_counter,
                        worker_summary_writer, args)
        workers.append(worker)

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if args.model_path is not None:
            print('Loading model...\n')
            ckpt = tf.train.get_checkpoint_state(args.model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Initializing a new model...\n')
            sess.run(tf.global_variables_initializer())
        print_params_nums()
        # Start work process for each worker in a separated thread
        worker_threads = []
        for worker in workers:
            t = threading.Thread(target=lambda: worker.run(sess, coord, saver))
            t.start()
            time.sleep(0.5)
            worker_threads.append(t)

        if args.eval_every > 0:
            evaluator = Evaluate(
                global_net, summary_writer, global_steps_counter, args)
            evaluate_thread = threading.Thread(
                target=lambda: evaluator.run(sess, coord))
            evaluate_thread.start()

        coord.join(worker_threads)


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path', default=None, type=str,
        help='Whether to use a saved model. (*None|model path)')
    parser.add_argument(
        '--save_path', default='/tmp/a3c', type=str,
        help='Path to save a model during training.')
    parser.add_argument(
        '--max_steps', default=int(1e8), type=int, help='Max training steps')
    parser.add_argument(
        '--start_time', default=None, type=str, help='Time to start training')
    parser.add_argument(
        '--threads', default=16, type=int,
        help='Numbers of parallel threads. [num_cpu_cores] by default')
    # evaluate
    parser.add_argument(
        '--eval_every', default=500, type=int,
        help='Evaluate the global policy every N seconds')
    parser.add_argument(
        '--record_video', default=True, type=bool,
        help='Whether to save videos when evaluating')
    parser.add_argument(
        '--eval_episodes', default=5, type=int,
        help='Numbers of episodes per evaluation')
    # hyperparameters
    parser.add_argument(
        '--init_learning_rate', default=7e-4, type=float,
        help='Learning rate of the optimizer')
    parser.add_argument(
        '--decay', default=0.99, type=float,
        help='decay factor of the RMSProp optimizer')
    parser.add_argument(
        '--smooth', default=1e-7, type=float,
        help='epsilon of the RMSProp optimizer')
    parser.add_argument(
        '--gamma', default=0.99, type=float,
        help='Discout factor of reward and advantages')
    parser.add_argument('--tmax', default=5, type=int, help='Rollout size')
    parser.add_argument(
        '--entropy_ratio', default=0.01, type=float,
        help='Initial weight of entropy loss')
    parser.add_argument(
        '--clip_grads', default=40, type=float,
        help='global norm gradients clipping')
    parser.add_argument(
        '--epsilon', default=1e-5, type=float,
        help='epsilon of rmsprop optimizer')

    return parser.parse_args()


if __name__ == '__main__':
    # ignore warnings by tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # make GPU invisible
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    args = args_parse()
    main(args)
