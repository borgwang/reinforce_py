import argparse
import tensorflow as tf


parser = argparse.ArgumentParser()
# Global arguments
parser.add_argument('--env', help='environment ID', default='Walker2d-v1')
parser.add_argument('--seed', help='RNG seed', type=int, default=931022)
parser.add_argument('--save_interval', type=int, default=0)
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--n_envs',  type=int, default=1)
parser.add_argument('--n_steps', type=int, default=int(1e6))

# Hyperparameters
parser.add_argument('--batch_steps', type=int, default=2048)
parser.add_argument('--minibatch', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--entropy_coef', type=float, default=0.0)
parser.add_argument('--v_coef', type=float, default=0.5)
parser.add_argument('--max_grad_norm', type=float, default=0.5)
parser.add_argument('--lam', type=float, default=0.95)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--clip_range', type=float, default=0.2)

args = parser.parse_args()

# Tensroflow Session Configuration
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
