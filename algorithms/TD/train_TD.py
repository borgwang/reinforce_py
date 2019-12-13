import argparse

from agents import TDAgent
from envs import GridWorld


def main(args):
    env = GridWorld()

    agent = TDAgent(env, epsilon=args.epsilon, gamma=args.discount, alpha=args.lr)
    agent.control(method=args.algorithm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algorithm', default='qlearn', help='(*qlearn | sarsa)')
    parser.add_argument(
        '--discount', type=float, default=0.9, help='discount factor')
    parser.add_argument(
        '--epsilon', type=float, default=0.3,
        help='parameter of epsilon greedy policy')
    parser.add_argument('--lr', type=float, default=0.05)
    main(parser.parse_args())
