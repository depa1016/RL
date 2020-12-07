import random
import gym
import numpy as np

total_actions = 0
total_exploration = 0
total_policy_action = 0

env = gym.make("FrozenLake-v0")
random.seed(0)

action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}


def init_q():
    q_values = {}
    q_counters = {}
    for s in range(0, env.observation_space.n):
        for a in range(0, env.action_space.n):
            q_values[(s, a)] = 0
            q_counters[(s, a)] = 0
    return q_values, q_counters


def play_episode(q_values=None):
    global total_actions
    global total_exploration
    global total_policy_action
    alpha = 0.3
    discount = 1.0
    epsilon = 0.1
    reward_in_episode = 0
    while reward_in_episode <= 0:
        state = env.reset()
        done = False
        action = None
        while not done:
            oldstate = state
            oldaction = action
            # Exploration Policy
            if random.random() < epsilon:
                action = random.randint(0, 3)
                total_exploration += 1
                total_actions += 1
            # Greedy Policy
            else:
                relevant_qs = [q_values[(state, action)] for action in range(0, env.action_space.n)]
                # Wenn alle Values 0 -> random action statt argmax -> erster wert
                if q_values[(state,np.argmax(relevant_qs))] == 0:
                    total_actions += 1
                    total_policy_action += 1
                    action = random.randint(0, 3)
                # Greedy Policy
                else:
                    total_policy_action += 1
                    total_actions += 1
                    action = np.argmax(relevant_qs)


            state, reward, done, _ = env.step(action)
            if oldaction is not None: q_values[(oldstate, oldaction)] += alpha * (
                        reward + discount * q_values[(state, action)] - q_values[(oldstate, oldaction)])
            reward_in_episode += reward


def print_q_values(q_values):
    print("#########")
    for (s, a), v in q_values.items():
        print(s, action2string[a], v)


def main():
    q_values, q_counters = init_q()
    successful_episodes = 1000
    while successful_episodes > 0:
        play_episode(q_values)
        successful_episodes -= 1
        print("Episode succesfully finished, ",successful_episodes," Episodes remaining.")
    print_q_values(q_values)
    print("percentage of exploration: ", total_exploration / total_actions)
    print("percentage of random actions: ", total_policy_action / total_actions)


main()
env.render()