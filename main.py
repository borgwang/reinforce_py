import argparse
import time
from Agents import DPAgent, MCAgent, TDAgent
from Envs import GridWorld, RandomWalk

def run(env, agent):
    # environment
    if args.env == 'gridworld':
        env = GridWorld()
    elif args.env == 'randomwalk':
        env = RandomWalk()
    else:
        print 'Error environment'
        return

    # agent
    if args.agent == 'DP':
        agent = DPAgent(env, gamma=args.discout)
        while not agent.converge:
            agent.evaluate_policy()
            agent.improve_policy()
    elif args.agent == 'MC':
        agent = MCAgent(env, epsilon=args.epsilon, gamma=args.discout, alpha=0.1)
        agent.control()
    elif args.agent == 'TD':
        agent = TDAgent(env, epsilon=args.epsilon, gamma=args.discout, alpha=0.05, lamda=0.7)
        agent.control(method='sarsa')
    else:
        print 'Error agent'
        return

def main():
    run(env=args.env, agent=args.agent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--env',default='gridworld',help='choose the environment (*gridworld|randomwalk)')
    parser.add_argument('-a', '--agent', default='TD', help='choose the agent type (DP|MC|*TD)')
    parser.add_argument('--discout', type=float, default=0.9, help='discout factor')
    parser.add_argument('--epsilon', type=float, default=0.5, help='parameter of epsilon greedy policy')

    args = parser.parse_args()
    main()
