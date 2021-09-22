
  
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = 'C:/Users/raysp/Desktop/School/College/16_Senior/Fall Semester/CS 5640/Assn 2'


class RandomAgent(object):
    def __init__(self, environment):
        self.environment = environment

    def act(self, _):
        return self.environment.action_space.sample()

    def learn(self, state, next_state, action, reward):
        return


class TabularAgent(object):
    def __init__(self, environment):
            # Set up figure
        self.fig = plt.figure(figsize=(40,8), facecolor='white')
        self.qFunction0 = self.fig.add_subplot(1,5,1, frameon=True)
        self.qFunction0.set_title("Q for action 0")
        self.qFunction0.set_xticks([])
        self.qFunction0.set_yticks([])
        self.qFunction0.invert_yaxis()
        self.qFunction0.set_ylabel('velocity')
        self.qFunction0.set_xlabel('position')
        self.qFunction1 = self.fig.add_subplot(1,5,2, frameon=True)
        self.qFunction1.set_title("Q for action 1")
        self.qFunction1.set_xticks([])
        self.qFunction1.set_yticks([])
        self.qFunction1.invert_yaxis()
        self.qFunction1.set_xlabel('position')
        self.qFunction2 = self.fig.add_subplot(1,5,3, frameon=True)        
        self.qFunction2.set_title("Q for action 2")
        self.qFunction2.set_xticks([])
        self.qFunction2.set_yticks([])
        self.qFunction2.invert_yaxis()
        self.qFunction2.set_xlabel('position')
        self.policy = self.fig.add_subplot(1,5,4, frameon=True)   
        #self.policy.autoscale(False)
        self.policy.set_title("Policy b=left, r=right")
        self.policy.set_xticks([])
        self.policy.set_yticks([])
        self.policy.invert_yaxis()
        self.reward = self.fig.add_subplot(1,5,5, frameon=True)  
        self.reward.set_title("Reward vs. learning its")
        self.reward.set_xlabel('iterations')
        self.reward.set_ylabel('reward')
        plt.show(block=False)
        # setup the hyper parameters
        self.environment = environment
        self.exploreRate = 0.9
        self.learningRate = 0.2
        self.maxBoxes = 20
        # keep track of rewards for visualization
        self.rewardList = []
        self.oneReward = 0
        self.Q = np.zeros(shape=(self.maxBoxes, self.maxBoxes, self.environment.action_space.n))
        self.trajectory = []
        self.averageRewardList = []
        # setup an array indexed by state and action, each location stores a dictionary of next states and rewards
        self.nextStateReward = [[[dict(), dict(), dict()] for _ in range(self.maxBoxes)] for _ in range(self.maxBoxes)]
        #self.Q = np.random.uniform(low = -1, high = 1, size = (self.maxBoxes+1, self.maxBoxes+1, self.environment.action_space.n))


    def convertState(self, state):
        #     Observation:
        # Type: Box(2)
        # Num    Observation               Min            Max
        # 0      Car Position              -1.2           0.6
        # 1      Car Velocity              -0.07          0.07
        # converts the continuous values to integers for the table
        return (self.convert(state[0], self.environment.observation_space.low[0], self.environment.observation_space.high[0], self.maxBoxes), 
                self.convert(state[1], self.environment.observation_space.low[1], self.environment.observation_space.high[1], self.maxBoxes))
        
    def convert(self, value, minV, maxV, howMany):
        return int(round(howMany*(value - minV)/(maxV - minV)))
        
    def act(self, stateC):
        state = self.convertState(stateC)
        if np.random.random() < 1 - self.exploreRate:
            return np.argmax(self.Q[state[0], state[1]]) 
        else:
            return np.random.randint(0, self.environment.action_space.n)

    def learn(self, stateC, nextStateC, action, reward, done, iteration):
        state = self.convertState(stateC)
        self.trajectory.append(state)
        nextState = self.convertState(nextStateC)
        # remember this state transition
        if not nextState in self.nextStateReward[state[0]][state[1]][action]:
            self.nextStateReward[state[0]][state[1]][action][nextState] = reward
        # use the Bellman relationship to get a new estimate for Q
        self.Qupdate(state, action, reward, nextState)
        self.oneReward += reward
        if done:
            # self.learnFromMemory()
            self.trajectory.append(nextState)
            # reduce th explore rate
            self.exploreRate = max(self.exploreRate - 0.8/10000, 0)
            #print("reward = %d" % self.oneReward)
            self.rewardList.append(self.oneReward)
            self.oneReward = 0
            if iteration % 500 == 0:
                # for i in range(0, self.maxBoxes):
                #     self.Q[i,0,0] = -1000
                print("Updating graphs at %d iterations" % iteration)
                if iteration > 0:
                    self.averageRewardList.append(np.mean(np.array(self.rewardList[-500:])))
                self.qFunction0.imshow(self.Q[:,:,0].T, cmap='jet', vmin=-210, vmax=0)
                self.qFunction1.imshow(self.Q[:,:,1].T, cmap='jet', vmin=-210, vmax=0)
                self.qFunction2.imshow(self.Q[:,:,2].T, cmap='jet', vmin=-210, vmax=0)
                self.policy.cla()
                self.policy.set_title("Policy b=left, r=right")
                self.policy.set_xticks([])
                self.policy.set_yticks([])
                self.policy.invert_yaxis()
                self.policy.imshow(np.argmax(self.Q, axis = 2).T, cmap='bwr', vmin=0, vmax=2)
                self.policy.plot([self.trajectory[i][0] for i in range(0, len(self.trajectory))], 
                                 [self.trajectory[i][1] for i in range(0, len(self.trajectory))], "k",  linewidth=4)
                self.reward.plot([i for i in range(0,len(self.rewardList))], self.rewardList, c = 'k')
                self.reward.plot([i*500 for i in range(0,len(self.averageRewardList))], self.averageRewardList, c = 'r',  linewidth=4)
                self.reward.set_ylim(-210, -100)
                plt.draw()
                self.saveImageOne(iteration)
                plt.pause(0.001)
            self.trajectory = []
                
    def Qupdate(self, state, action, reward, nextState):
        qEstimate = reward + max(self.Q[nextState[0],nextState[1]])
        # incrementally update the Q value in the table
        self.Q[state[0], state[1], action] += self.learningRate * (qEstimate - self.Q[state[0], state[1], action])

    def learnFromMemory(self):
        for x in range(0, self.maxBoxes):
            for xDot in range(0, self.maxBoxes):
                for a in range(0, self.environment.action_space.n):
                    for nextState in self.nextStateReward[x][xDot][a]:
                        reward = self.nextStateReward[x][xDot][a][nextState]
                        #print(nextState, reward)
                        self.Qupdate((x, xDot), a, reward, nextState)
        
    def saveImageOne(self, iteration):
    #print pathOut + fileName
        fileName = "MC_" + str(iteration).rjust(6,'0')
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        #print(onePath + fileName + '.png')
        self.fig.savefig(PATH + fileName + '.png',  dpi=100)


class MonteCarloAgent(TabularAgent):
    def __init__(self, environment):
        super().__init__(environment)
        # keep track of rewards, actions, and states for monte carlo learning. [(s1, action, r1), (s2, action, r2), ... (sT, action, rT)]
        self.monteCarloTrajectory = []
        self.discountFactor = 0.9

    def learn(self, stateC, nextStateC, action, reward, done, episode):
        state = super().convertState(stateC)
        self.trajectory.append(state)
        nextState = super().convertState(nextStateC)
        # remember this state transition
        if not nextState in self.nextStateReward[state[0]][state[1]][action]:
            self.nextStateReward[state[0]][state[1]][action][nextState] = reward

        # # use the Bellman relationship to get a new estimate for Q
        # self.Qupdate(state, action, reward, nextState)

        self.oneReward += reward

        # keep track of rewards, actions, and states for Monte Carlo
        self.monteCarloTrajectory.append((state, action, reward))

        if done:
            self.monteCarloLearn()
            self.monteCarloTrajectory = []
            # self.learnFromMemory()
            self.trajectory.append(nextState)
            # reduce th explore rate
            self.exploreRate = max(self.exploreRate - 0.8 / 10000, 0)
            # print("reward = %d" % self.oneReward)
            self.rewardList.append(self.oneReward)
            self.oneReward = 0
            if episode % 500 == 0:
                # for i in range(0, self.maxBoxes):
                #     self.Q[i,0,0] = -1000
                print("Updating graphs at %d iterations" % episode)
                if episode > 0:
                    self.averageRewardList.append(np.mean(np.array(self.rewardList[-500:])))
                self.qFunction0.imshow(self.Q[:, :, 0].T, cmap='jet', vmin=-210, vmax=0)
                self.qFunction1.imshow(self.Q[:, :, 1].T, cmap='jet', vmin=-210, vmax=0)
                self.qFunction2.imshow(self.Q[:, :, 2].T, cmap='jet', vmin=-210, vmax=0)
                self.policy.cla()
                self.policy.set_title("Policy b=left, r=right")
                self.policy.set_xticks([])
                self.policy.set_yticks([])
                self.policy.invert_yaxis()
                self.policy.imshow(np.argmax(self.Q, axis=2).T, cmap='bwr', vmin=0, vmax=2)
                self.policy.plot([self.trajectory[i][0] for i in range(0, len(self.trajectory))],
                                 [self.trajectory[i][1] for i in range(0, len(self.trajectory))], "k", linewidth=4)
                self.reward.plot([i for i in range(0, len(self.rewardList))], self.rewardList, c='k')
                self.reward.plot([i * 500 for i in range(0, len(self.averageRewardList))], self.averageRewardList,
                                 c='r', linewidth=4)
                self.reward.set_ylim(-210, -100)
                plt.draw()
                super().saveImageOne(episode)
                plt.pause(0.001)
            self.trajectory = []

    def monteCarloLearn(self):
        T = len(self.monteCarloTrajectory) - 1
        t = T

        while True:
            if t < 0:
                break

            state = self.monteCarloTrajectory[t][0]
            action = self.monteCarloTrajectory[t][1]

            if t == T:  # if end of episode
                G = self.monteCarloTrajectory[t][2]
                self.Q[state[0], state[1], action] += self.learningRate * (G - self.Q[state[0], state[1], action])
                t -= 1

            elif t >= 0:
                G = 0
                for i in range(t + 1, T + 1):
                    # ignore discount factor  for now
                    G += self.monteCarloTrajectory[i][2]  # i-th reward
                self.Q[state[0], state[1], action] += self.learningRate * (G - self.Q[state[0], state[1], action])
                t -= 1


    # def monteCarloLearn(self):
    #     REWARD_INDEX = 2
    #     episodeLength = len(self.monteCarloTrajectory)
    #     trajectory = [i for i in reversed(self.monteCarloTrajectory)]
    #     visited = []
    #     discounts = np.logspace(0, episodeLength, num=episodeLength, base=self.discountFactor, endpoint=False)
    #
    #     for i in range(episodeLength):
    #         state = trajectory[i][0]
    #         action = trajectory[i][1]
    #         if state in visited:
    #             continue
    #         G = (trajectory[i][REWARD_INDEX] + sum([trajectory[j][REWARD_INDEX] for j in range(len(trajectory[:i]))]))
    #
    #         visited.append(state)
    #         self.Q[state[0], state[1], action] += self.learningRate * (G - self.Q[state[0], state[1], action])


class SARSAAgent(TabularAgent):
    def __init__(self, environment):
        super().__init__(environment)

    def act(self, stateC):
        state = self.convertState(stateC)
        if np.random.random() < 1 - self.exploreRate:
            return np.argmax(self.Q[state[0], state[1]])
        else:
            return np.random.randint(0, self.environment.action_space.n)

    def learn(self, stateC, nextStateC, action, reward, done, iteration):
        state = super().convertState(stateC)
        self.trajectory.append(state)
        nextState = super().convertState(nextStateC)
        # remember this state transition
        if not nextState in self.nextStateReward[state[0]][state[1]][action]:
            self.nextStateReward[state[0]][state[1]][action][nextState] = reward
        # use the Bellman relationship to get a new estimate for Q
        super().Qupdate(state, action, reward, nextState)
        self.oneReward += reward
        if done:
            super().learnFromMemory()
            self.trajectory.append(nextState)
            # reduce th explore rate
            self.exploreRate = max(self.exploreRate - 0.8 / 10000, 0)
            # print("reward = %d" % self.oneReward)
            self.rewardList.append(self.oneReward)
            self.oneReward = 0
            if iteration % 500 == 0:
                # for i in range(0, self.maxBoxes):
                #     self.Q[i,0,0] = -1000
                print("Updating graphs at %d iterations" % iteration)
                if iteration > 0:
                    self.averageRewardList.append(np.mean(np.array(self.rewardList[-500:])))
                self.qFunction0.imshow(self.Q[:, :, 0].T, cmap='jet', vmin=-210, vmax=0)
                self.qFunction1.imshow(self.Q[:, :, 1].T, cmap='jet', vmin=-210, vmax=0)
                self.qFunction2.imshow(self.Q[:, :, 2].T, cmap='jet', vmin=-210, vmax=0)
                self.policy.cla()
                self.policy.set_title("Policy b=left, r=right")
                self.policy.set_xticks([])
                self.policy.set_yticks([])
                self.policy.invert_yaxis()
                self.policy.imshow(np.argmax(self.Q, axis=2).T, cmap='bwr', vmin=0, vmax=2)
                self.policy.plot([self.trajectory[i][0] for i in range(0, len(self.trajectory))],
                                 [self.trajectory[i][1] for i in range(0, len(self.trajectory))], "k", linewidth=4)
                self.reward.plot([i for i in range(0, len(self.rewardList))], self.rewardList, c='k')
                self.reward.plot([i * 500 for i in range(0, len(self.averageRewardList))], self.averageRewardList,
                                 c='r', linewidth=4)
                self.reward.set_ylim(-210, -100)
                plt.draw()
                super().saveImageOne(iteration)
                plt.pause(0.001)
            self.trajectory = []


class TDnAgent(TabularAgent):
    def __init__(self, environment, n=50, learningRate=0.2):
        super().__init__(environment)
        # setup the hyper parameters
        self.learningRate = learningRate # 0.2
        self.n = n
        # keep track of rewards, actions, and states for TD(n) learning. [(s1, action, r1, nextState), ... (sT, action, rT, nextState)]
        self.tdTrajectory = []
        self.discountFactor = 0.9

    def learn(self, stateC, nextStateC, action, reward, done, episode):
        state = super().convertState(stateC)
        self.trajectory.append(state)
        nextState = super().convertState(nextStateC)
        # remember this state transition
        if not nextState in self.nextStateReward[state[0]][state[1]][action]:
            self.nextStateReward[state[0]][state[1]][action][nextState] = reward

        self.oneReward += reward

        # keep track of rewards, actions, and states for Monte Carlo
        self.tdTrajectory.append((state, action, reward, nextState))

        if done:
            self.temporalDifferenceLearn()
            self.tdTrajectory = []
            # self.learnFromMemory()
            self.trajectory.append(nextState)
            # reduce th explore rate
            self.exploreRate = max(self.exploreRate - 0.8 / 10000, 0)
            # print("reward = %d" % self.oneReward)
            self.rewardList.append(self.oneReward)
            self.oneReward = 0
            if episode % 500 == 0:
                # for i in range(0, self.maxBoxes):
                #     self.Q[i,0,0] = -1000
                print("Updating graphs at %d iterations" % episode)
                if episode > 0:
                    self.averageRewardList.append(np.mean(np.array(self.rewardList[-500:])))
                self.qFunction0.imshow(self.Q[:, :, 0].T, cmap='jet', vmin=-210, vmax=0)
                self.qFunction1.imshow(self.Q[:, :, 1].T, cmap='jet', vmin=-210, vmax=0)
                self.qFunction2.imshow(self.Q[:, :, 2].T, cmap='jet', vmin=-210, vmax=0)
                self.policy.cla()
                self.policy.set_title("Policy b=left, r=right")
                self.policy.set_xticks([])
                self.policy.set_yticks([])
                self.policy.invert_yaxis()
                self.policy.imshow(np.argmax(self.Q, axis=2).T, cmap='bwr', vmin=0, vmax=2)
                self.policy.plot([self.trajectory[i][0] for i in range(0, len(self.trajectory))],
                                 [self.trajectory[i][1] for i in range(0, len(self.trajectory))], "k", linewidth=4)
                self.reward.plot([i for i in range(0, len(self.rewardList))], self.rewardList, c='k')
                self.reward.plot([i * 500 for i in range(0, len(self.averageRewardList))], self.averageRewardList,
                                 c='r', linewidth=4)
                self.reward.set_ylim(-210, -100)
                plt.draw()
                super().saveImageOne(episode)
                plt.pause(0.001)
            self.trajectory = []

    def temporalDifferenceLearn(self):
        T = len(self.tdTrajectory)
        t = 0
        while True:
            if t >= 0:
                G = 0
                for i in range(t + 1, min(t + self.n, T)):
                    # ignore discount factor for now
                    G += self.tdTrajectory[i][2]  # get i-th reward
                if t + self.n < T:
                    state = self.tdTrajectory[t + self.n][0]
                    action = self.tdTrajectory[t + self.n][1]
                    G += self.Q[state[0], state[1], action]

                # update Q table
                state = self.tdTrajectory[t][0]
                action = self.tdTrajectory[t][1]
                self.Q[state[0], state[1], action] += self.learningRate * (G - self.Q[state[0], state[1], action])

            if t == T - 1:
                break

            t += 1