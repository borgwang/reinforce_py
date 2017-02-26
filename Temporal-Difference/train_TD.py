import argparse
import time
from agents import TDAgent
from Envs import GridWorld


def main():
    # environment
    env = GridWorld()
    # agent
    agent = TDAgent(env, epsilon=args.epsilon, gamma=args.discout, alpha=0.05, lamda=0.7)
    agent.control(method=args.algorithm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='qlearn', help='(*qlearn | sarsa)')
    parser.add_argument('--discout', type=float, default=0.9, help='discout factor')
    parser.add_argument('--epsilon', type=float, default=0.5, help='parameter of epsilon greedy policy')

    args = parser.parse_args()
    main()
