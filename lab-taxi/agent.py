import numpy as np
from collections import defaultdict
import random


class Agent:

    def __init__(self, nA=6, epsilon=0.2, epsilon_decay=0.95, epsilon_min=0.1, alpha=0.8, gamma=1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent (North, South, East, West, Pickup, Putdown)
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        # Epsilon-greedy policy
        return np.argmax(self.Q[state]) if self.epsilon < random.random() else np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        # SarsaMax (Q-learning) update for Q-function
        # self.Q[state][action] = self.sarsamax(state, action, reward, next_state)

        # Expected Sarsa (Q-learning) update for Q-function
        self.Q[state][action] = self.expected_sarsa(state, action, reward, next_state)

        # Reduce the epsilon till 0.1
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

    def sarsamax(self, state, action, reward, next_state):
        max_reward = np.max(self.Q[next_state])
        return self.Q[state][action] + self.alpha * (reward + self.gamma * max_reward - self.Q[state][action])

    def expected_sarsa(self, state, action, reward, next_state):
        exp_reward = np.max(self.Q[next_state]) * (1 - self.epsilon) + sum(self.Q[next_state]) * self.epsilon / self.nA
        return self.Q[state][action] + self.alpha * (reward + self.gamma * exp_reward - self.Q[state][action])