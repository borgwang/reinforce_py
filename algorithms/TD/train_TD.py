import argparse
from agents import TDAgent
from envs import GridWorld


def main(args):
    # environment
    env = GridWorld()
    # agent
    agent = TDAgent(
        env, epsilon=args.epsilon, gamma=args.discout, alpha=0.05, lamda=0.7)
    agent.control(method=args.algorithm)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algorithm', default='qlearn', help='(*qlearn | sarsa)')
    parser.add_argument(
        '--discout', type=float, default=0.9, help='discout factor')
    parser.add_argument(
        '--epsilon', type=float, default=0.5,
        help='parameter of epsilon greedy policy')

    return parser.parse_args()


if __name__ == '__main__':
    main(args_parse())
