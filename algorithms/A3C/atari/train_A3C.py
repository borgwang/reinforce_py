import threading
import os
import argparse
import time
import tensorflow as tf
import itertools

from atari_env import Atari
from net import Net
from worker import Worker
from utils import print_params_nums
from evaluate import Evaluate


def main(args):
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    summary_writer = tf.summary.FileWriter(os.path.join(args.save_path, 'log'))
    global_steps_counter = itertools.count()  # thread-safe

    global_net = Net(Atari.s_dim, Atari.a_dim, 'global', args)
    num_workers = args.threads
    workers = []

    # create workers
    for i in range(num_workers):
        worker_summary_writer = summary_writer if i == 0 else None
        w = Worker(i, Atari(args), global_steps_counter, worker_summary_writer,
                   args)
        workers.append(w)

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if args.model_path is not None:
            print 'Loading model...\n'
            ckpt = tf.train.get_checkpoint_state(args.model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print 'Initializing a new model...\n'
            sess.run(tf.global_variables_initializer())
        print_params_nums()
        # Start work process for each worker in a seperated thread
        worker_threads = []
        for w in workers:
            run = lambda: w.run(sess, coord, saver)
            t = threading.Thread(target=run)
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
        '--eval_every', default=300, type=int,
        help='Evaluate the global policy every N seconds')
    parser.add_argument(
        '--save_videos', default=True, type=bool,
        help='Whether to save videos when evaluating')
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
        '--clip_grads', default=40.0, type=float,
        help='global norm gradients clipping')
    parser.add_argument(
        '--epsilon', default=0.1, type=float,
        help='epsilon of rmsprop optimizer')

    return parser.parse_args()


if __name__ == '__main__':
    # ignore warnings by tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # make GPU invisible
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    args = args_parse()
    main(args)
