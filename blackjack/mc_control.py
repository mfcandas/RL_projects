import gym
import sys
import numpy as np
from collections import defaultdict
import random
import seaborn as sns
from matplotlib import pyplot as plt


# %%Main Control Run
def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    """
    :param env: gym blackjack environment
    :param num_episodes:
    :param alpha: used for constant-alpha MC control, weight of new Return (and 1-alpha is for the historical)
    :param gamma: discount rate used to calculate return G
    :param eps_start: epsilon-Greedy, probability of selecting non-optimal policy (exploration)
    :param eps_decay: decay rate of epsilon
    :param eps_min: minimum epsilon value (no more decay once reached to this value)
    :return dc_policy: dictionary of {s: action}, action that agent picks at state s
    :return Q: Q-table, dictionary of {s: numpy_array(action 0 expected-reward, action 1 expected-reward)}
    """

    nA = env.action_space.n  # number of actions
    Q = defaultdict(lambda: np.zeros(nA))  # initialize empty dictionary of arrays

    dc_policy = defaultdict(int)  # dictionary of {state: action}
    epsilon = eps_start

    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 10000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = generate_episode(env, dc_policy, epsilon)  # List[(state, action, reward)]
        epsilon = max(eps_min, epsilon*eps_decay)
        np_reward = np.array([r for _, _, r in episode])
        gamma_function = np.array([gamma ** t for t in range(len(np_reward) + 1)])

        for i, (s, a, r) in enumerate(episode):
            # print('episode:', episode)
            # print('s, a, r', s, a, r)

            Q[s][a] = alpha*sum(gamma_function[:-(i + 1)] * np_reward[i:]) + (1-alpha)*Q[s][a]

        dc_policy = update_policy(Q)

    return dc_policy, Q


def generate_episode(env, policy, epsilon):
    episode = []
    state = env.reset()
    while True:
        if random.uniform(0, 1) > epsilon:
            # generate episode using dc_policy with p = (1-epsilon)
            action = pick_optimal_action(env, state, policy)
        else:
            # generate episode using equiprobable with p = epsilon
            action = pick_random_action(env)

        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def pick_optimal_action(env, state, policy):
    if state in policy:
        return policy[state]
    else:
        pick_random_action(env)


def pick_random_action(env):
    action = env.action_space.sample()
    return action


def update_policy_my(policy, Q):
    for state, np_action in Q.items():
        policy[state] = np.argmax[np_action]
    return policy


def update_policy(Q):
    return {k: np.argmax(v) for k, v in Q.items()}


# %% Main flow ---------------------------------------------------------------------------------------------------------
bj_env = gym.make('Blackjack-v0')
print(bj_env.observation_space)
print(bj_env.action_space)

dc_final_policy, Q = mc_control(bj_env, 500000, 0.02)

# plot the policy with usable-ace
player_hand, dealer_hand, _, action = zip(*[[*k, v] for k, v in dc_final_policy.items() if k[2]])  # with usable ace
sns.scatterplot(x=dealer_hand, y=player_hand, size=action)
plt.show()

# plot the policy without usable-ace
player_hand, dealer_hand, _, action = zip(*[[*k, v] for k, v in dc_final_policy.items() if not k[2]])  # with usable ace
sns.scatterplot(x=dealer_hand, y=player_hand, size=action)
plt.show()