import threading
import multiprocessing
import os
import argparse
from time import sleep
import tensorflow as tf

from env_doom import Doom
from net import Net
from worker import Worker
from utils import print_net_params_number

def main(args):
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.gif_path is not None and not os.path.exists(args.gif_path):
        os.makedirs(args.gif_path)

    with tf.device('/cpu:0'):
        tf.reset_default_graph()
        # define global_ep as a tf-variable, so that we can
        # maintain its value when loading a saved model
        global_ep = tf.Variable(0, dtype=tf.int32, name='global_ep', trainable=False)
        env = Doom(visiable=False)
        master = Net(env.state_dim, env.action_dim, 'global', None)
        num_workers = multiprocessing.cpu_count()
        workers = []

        # create workers
        for i in range(num_workers):
            w = Worker(i, Doom(), global_ep, args)
            workers.append(w)
        print '%d workers in total.\n' % num_workers
        saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if args.model_path is not None:
            print 'Loading model...'
            ckpt = tf.train.get_checkpoint_state(args.model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            print 'Initializing a new model...'
            sess.run(tf.global_variables_initializer())
        print_net_params_number()

        # Start work process for each worker in a seperate thread
        worker_threads = []
        for w in workers:
            run = lambda: w.run(sess, coord, saver)
            t = threading.Thread(target=(run))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, help=
            'Whether to use a saved model. (*None|model path)')
    parser.add_argument('--save_path', default='./model/', help=
            'Path to save a model during training.')
    parser.add_argument('--gif_path', default=None, help=
            'Path of the generated gif during training.')
    parser.add_argument('--save_every', default=0, help=
            'Interval of saving model')
    parser.add_argument('--max_ep_len', default=300, help=
            'Max episode steps')
    parser.add_argument('--max_ep', default=3000, help=
            'Max training episode')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # ignore warnings by tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    main(args_parse())
