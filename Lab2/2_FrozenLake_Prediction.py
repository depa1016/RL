import random
import gym
import numpy as np

env = gym.make("FrozenLake-v0")
random.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()

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

    state = env.reset()
    done = False
    r_s = []
    s_a = []
    while not done:
        if q_values is None:
            action = random.randint(0, 3)
        else:
            relevant_qs = [q_values[(state, action)] for action in range(0, env.action_space.n)]
            action = np.argmax(relevant_qs)

        s_a.append((state, action))
        state, reward, done, _ = env.step(action)
        r_s.append(reward)
    return s_a, r_s


def print_q_values(q_values):
    print("#########")
    for (s, a), v in q_values.items():
        print(s, action2string[a], v)


def main():
    q_values, q_counters = init_q()
    successful_episodes = 1000
    while successful_episodes > 0:
        s_a, r_s = play_episode()

        # update q-values with MC-prediction
        for i, q in enumerate(s_a):
            return_i = sum(r_s[i:])
            q_counters[q] += 1
            q_values[q] += 1/q_counters[q] * (return_i - q_values[q])

        if sum(r_s) > 0:
            print_q_values(q_values)
            all_rewards = 0
            for i in range(0, 100):
                s_a, rewards = play_episode(q_values)
                all_rewards += sum(rewards)
            print(all_rewards / 100)
            successful_episodes -= 1


main()