# Local imports
from agents import RandomAgent
from agents import TabularAgent
from agents import MonteCarloAgent
from agents import SARSAAgent
from agents import TDnAgent
from agents import DynaQAgent

# Python imports
import numpy as np
from time import time  # just to have timestamps in the files
import matplotlib.pyplot as plt

# OpenAI Gym imports
import gym
from gym import wrappers

def runAgent(agent, n, close=True):
    # We are only doing a single simulation. Increase 1 -> N to get more runs.
    for episode in range(n):
        #print(iteration)
        # Always start the simulation by resetting it
        state = env.reset()
        done = False

        # Either limit the number of simulation steps via a "for" loop, or continue
        # the simulation until a failure state is reached with a "while" loop
        while not done:

            # Render the environment. You will want to remove this or limit it to the
            # last simulation iteration with a "if iteration == last_one: env.render()"
            if episode % 500 == 0:
                env.render()

            # Have the agent
            #   1: determine how to act on the current state
            #   2: go to the next state with that action
            #   3: learn from feedback received if that was a good action
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, next_state, action, reward, done, episode)

            # Progress to the next state
            state = next_state

        #env.render()  # Render the last simluation frame.
    if close:
        env.close()  # Do not forget this! Tutorials like to leave it out.

    """
    MountainCar Movements
        0 - Move left
        1 - Don't move
        2 - Move right
    """
    #print(agent.value_table)

def testTDn(env):
    actualStateValues = np.arange(-20, 20, 2) / 20.0
    nRange = [2, 20, 50, 100, 200]
    learningRateRange = np.linspace(0, 1, 6)
    sqErrors = {}

    for n in nRange:
        ers = []
        for lr in learningRateRange:
            print("running TD(n) agent with \"n\" as: " + str(n) + " and learningRate as: " + str(lr))
            agent = TDnAgent(env, n)
            runAgent(agent, 5000, False)
            estimateStateValues = [np.mean(list(v)) for v in agent.Q]

            ers.append(np.mean([er**2 for er in actualStateValues - np.array(estimateStateValues)]))
        sqErrors[n] = ers

    env.close()

    plt.figure(figsize=[10, 6])
    for n in nRange:
        plt.plot(learningRateRange, sqErrors[n], label="n={}".format(n))

    plt.xlabel("learning rate")
    plt.ylabel("RMS error")
    plt.legend()

    plt.show()


# Remove the monitoring if you do not want a video
env = gym.make("MountainCar-v0")
# env = wrappers.Monitor(env, "./videos/" + str(time()) + "/")

# Change the agent to a different one by simply swapping out the class
# ex) RandomAgent(env) --> TabularAgent(env)
# agent = RandomAgent(env)
# agent = TabularAgent(env)
# agent = MonteCarloAgent(env)
# agent = SARSAAgent(env)
# agent = TDnAgent(env, 20)
agent = SARSAAgent(env, 60)

runAgent(agent, 20000)
# testTDn(env)