import random
import gym
import numpy as np

env = gym.make("FrozenLake-v0")
random.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()

action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
rewardsum=0
total_actions=0
total_exploration=0
total_random_action=0

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
    global total_random_action
    global rewardsum
    alpha=0.3
    discount=1.0
    epsilon=0.15
    r_s = []
    s_a = []
    rewardInEpisode = 0
    while rewardInEpisode <= 0:
        state = env.reset()
        done = False
        while not done:
            if(random.random() < epsilon):
                action = random.randint(0, env.action_space.n-1)
                total_exploration += 1
                total_actions += 1
            else:
                total_random_action += 1
                relevant_qs = [q_values[(state, action)] for action in range(0, env.action_space.n)]
                action = np.argmax(relevant_qs)
                total_actions += 1
            s_a.append((state, action))
            oldstate = state
            state, reward, done, _ = env.step(action)
            rewardInEpisode += reward
            rewardsum += reward
            #print("oldstate: ",oldstate, "newstate: ",state,"reward:",reward)
            q_values[(oldstate, action)] += alpha*(reward + discount*q_values[(state, action)]-q_values[(oldstate, action)])
            r_s.append(reward)


def print_q_values(q_values):
    print("#########")
    for (s, a), v in q_values.items():
        print(s, action2string[a], v)


def main():
    q_values, q_counters = init_q()
    successful_episodes = 10000
    while successful_episodes > 0:
            play_episode(q_values)
            successful_episodes -= 1
    print_q_values(q_values)
    print(rewardsum)
    print("percentage of exploration: ", total_exploration/total_actions )
    print("percentage of random actions: ", total_random_action/total_actions )

main()