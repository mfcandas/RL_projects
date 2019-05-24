from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')

# agent = Agent(epsilon=0.2,
#               epsilon_decay=0.95,
#               epsilon_min=0.1,
#               alpha=0.8,
#               gamma=1)  # 5.28

# agent = Agent(epsilon=0.5,
#               epsilon_decay=0.95,
#               epsilon_min=0.1,
#               alpha=0.8,
#               gamma=1)  # 6.28

# agent = Agent(epsilon=0.8,
#               epsilon_decay=0.95,
#               epsilon_min=0.1,
#               alpha=0.8,
#               gamma=1)  # 5.28

# agent = Agent(epsilon=0.5,
#               epsilon_decay=0.95,
#               epsilon_min=0.1,
#               alpha=0.5,
#               gamma=1)  # 6.11

# agent = Agent(epsilon=0.5,
#               epsilon_decay=0.95,
#               epsilon_min=0.1,
#               alpha=0.9,
#               gamma=1)  # 5.53

# l_epsilon = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
# l_epsilon_decay = [0.99, 0.95]
# l_epsilon_min = [0.05, 0.1, 0.2]
# l_alpha = [0.3, 0.5, 0.7, 0.9]
# l_gamma = [1, 0.9]

# l_epsilon = [0.1, 0.15, 0.2]
# l_epsilon_decay = [0.99]
# l_epsilon_min = [0.01]
# l_alpha = [0.3, 0.5, 0.7, 0.9]
# l_gamma = [1, 0.9]

# l_epsilon = [0.1]
# l_epsilon_decay = [0.99]
# l_epsilon_min = [0.0]
# l_alpha = [0.3, 0.5, 0.7, 0.9]
# l_gamma = [1]

# l_epsilon = [0.1]
# l_epsilon_decay = [0.99]
# l_epsilon_min = [0.0]
# l_alpha = [0.3, 0.2, 0.1, 0.05]
# l_gamma = [1]


l_epsilon = [0.075]
l_epsilon_decay = [0.99]
l_epsilon_min = [0.0]
l_alpha = [0.3, 0.31, 0.33, 0.35]
l_gamma = [1]


run = 0
with open('restuls_6.txt', 'w+') as inf:
    for epsilon in l_epsilon:
        for epsilon_decay in l_epsilon_decay:
            for epsilon_min in l_epsilon_min:
                for alpha in l_alpha:
                    for gamma in l_gamma:
                        run += 1
                        inf.write(f'\n\nrun : {run} ================================================')
                        agent = Agent(epsilon=epsilon,
                                      epsilon_decay=epsilon_decay,
                                      epsilon_min=epsilon_min,
                                      alpha=alpha,
                                      gamma=gamma)

                        inf.write(f'\nepsilon: {agent.epsilon}'
                                  f'\nepsilon_decay: {agent.epsilon_decay}'
                                  f'\nepsilon_min: {agent.epsilon_min}'
                                  f'\nalpha: {agent.alpha}'
                                  f'\ngamma: {agent.gamma}')

                        avg_rewards, best_avg_reward = interact(env, agent)

                        inf.write(f'\nBest avg reward: {best_avg_reward}')

        #                 break
        #             break
        #         break
        #     break
        # break
