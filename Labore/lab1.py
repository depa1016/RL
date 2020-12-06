import gym
import random

env = gym.make("FrozenLake-v0", is_slippery=False, map_name="4x4")

random.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()

no_of_actions = env.env.nA
action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

state = env.reset()
done = False
reward = 0
policyCounter = 0
stateActionDict = {}
bestPolicyStepcount = 100
foundBetterPolicy = False
while not policyCounter > 5:
    episodeCounter = 0
    policyCounter = policyCounter + 1
    if(reward>0): reward = 0
    while not reward > 0:
        stateActionPairsTemp = {}
        state = env.reset()
        episodeCounter = episodeCounter + 1
        done = False
        stepCounter=0
        while not done:
            if(state in stateActionDict): action = stateActionDict[state]
            else: action = random.randint(0, no_of_actions-1)  # choose a random action
            stateActionPairsTemp[state]=action
            state, reward, done, _ = env.step(action)
            stepCounter += 1
            print(f"\nAction:{action2string[action]}, new state:{state}, reward:{reward}")
            env.render()
    if bestPolicyStepcount < stepCounter:
        print("bessere policy gefunden!")
        bestPolicyStepcount = stepCounter
        foundBetterPolicy = True
    stateActionDict.update(stateActionPairsTemp)
    print("Es hat ",episodeCounter," Episoden benötigt!")

print(stateActionDict)
if foundBetterPolicy: print("es wurde eine bessere policy gefunden")
print("Policy 5 mal iteriert, zum Erreichen des Ziels wurden ", stepCounter, " Schritte benötigt.")