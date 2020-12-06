import random
import gym
import numpy as np

env = gym.make("FrozenLake-v0")
random.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()

action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
zaehler = 0
nopolicycounter = 0

def init_q_values():
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
            global zaehler
            zaehler += 1
            action = random.randint(0, 3)
        else:
            global nopolicycounter
            nopolicycounter += 1
            relevant_qs = [q_values[(state, action)] for action in range(0, env.action_space.n)]
            action = np.argmax(relevant_qs)

        s_a.append((state, action))
        state, reward, done, _ = env.step(action)
        r_s.append(reward)
    return r_s, s_a


def print_q_values(q_values):
    print("#########")
    for (s, a), v in q_values.items():
        print(s, action2string[a], v)


def use_greedy_policy(q_values):
    episodeCounter = 0
    while episodeCounter < 100:
        episodeCounter += 1


def main():
    q_v, q_counter = init_q_values()
    successful_episodes = 1000
    while successful_episodes > 0:
        rewards, state_actions = play_episode()

        for i, s_a in enumerate(state_actions):
            q_counter[s_a] += 1
            return_i = sum(rewards[i:])
            q_v[s_a] += 1 / q_counter[s_a] * (return_i - q_v[s_a])

        if sum(rewards) > 0:
            # print_q_values(q_v)
            all_rewards = 0
            for i in range(0, 100):
                reward, state_actions = play_episode(q_v)
                all_rewards += sum(reward)
            print(all_rewards / 100)
            successful_episodes -= 1
    print("Zähler:", zaehler," Nopolicycounter:",nopolicycounter)


main()
