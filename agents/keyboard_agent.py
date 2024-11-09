import gym
from itertools import count
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from pvz import config
import matplotlib.pyplot as plt
import pygame
from pvz import config



class KeyboardAgent():
    DECIDE = False
    def __init__(self, n_plants=4, cell_size=75, offset_border=100):
        self.n_plants = n_plants
        self.cell_size = cell_size  # 그리드 셀의 크기
        self.offset_border = offset_border  # 그리드 시작 위치
        self.no_plant = None
        self.lane = None
        self.pos = None
        self.decide = False

    def set_parameters(self, lane, pos, no_plant):
        self.lane = lane
        self.pos = pos
        self.no_plant = no_plant
        self.decide = True
        KeyboardAgent.DECIDE = True

    def decide_action(self, observation, lane, pos, no_plant):
        if KeyboardAgent.DECIDE is True:
            #self.decide = False
            print(1 + self.no_plant + self.n_plants * (self.lane + config.N_LANES * self.pos))
            KeyboardAgent.DECIDE = False
            return 1 + self.no_plant + self.n_plants * (self.lane + config.N_LANES * self.pos)
        #print("keyboard agent에서 decide_action")
        return 0  # 선택하지 않은 경우

    def discount_rewards(self, r, gamma):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.shape[0])):
            running_add = running_add * gamma + r[t][0]
            discounted_r[t][0] = running_add
        return discounted_r

    def update(self, observation, actions, rewards):
        # discount rewards
        rewards = self.discount_rewards(rewards, gamma=0.9)
        print(rewards)




class PVZ():
    def __init__(self,render=True, max_frames = 1000, n_iter = 100000):
        self.env = gym.make('gym_pvz:pvz-env-v2')
        self.max_frames = max_frames
        self.render = render
        self._grid_size = config.N_LANES * config.LANE_LENGTH

        
    def get_actions(self):
        return list(range(self.env.action_space.n))

    def num_observations(self):
        return config.N_LANES * config.LANE_LENGTH + config.N_LANES + len(env.env.plant_deck) + 1

    def num_actions(self):
        return self.env.action_space.n

    def _transform_observation(self, observation):
        observation_zombie = self._grid_to_lane(observation[self._grid_size:2*self._grid_size])
        observation = np.concatenate([observation[:self._grid_size], observation_zombie, 
        [observation[2 * self._grid_size]/SUN_NORM], 
        observation[2 * self._grid_size+1:]])
        if (self.render):
            print(observation)
        return observation

    def _grid_to_lane(self, grid):
        grid = np.reshape(grid, (config.N_LANES, config.LANE_LENGTH))
        return np.sum(grid, axis=1)/HP_NORM

    def play(self,agent, epsilon=0):
        """ Play one episode and collect observations and rewards """

        summary = dict()
        summary['rewards'] = list()
        summary['observations'] = list()
        summary['actions'] = list()
        observation = self._transform_observation(self.env.reset())
        
        t = 0

        while(self.env._scene._chrono<self.max_frames):
            if(self.render):
                self.env.render()
            if np.random.random()<epsilon:
                # print("exploration")
                action = np.random.choice(self.get_actions(), 1)[0]
            else:
                action = agent.decide_action(observation)

            summary['observations'].append(observation)
            summary['actions'].append(action)
            observation, reward, done, info = self.env.step(action)
            observation = self._transform_observation(observation)
            summary['rewards'].append(reward)

            if done:
                break

        summary['observations'] = np.vstack(summary['observations'])
        summary['actions'] = np.vstack(summary['actions'])
        summary['rewards'] = np.vstack(summary['rewards'])
        return summary



if __name__ == "__main__":

    env = PVZ(render=True,max_frames = 1000)
    agent = KeyboardAgent()

    for episode_idx in range(1000):
        
        # play episodes
        summary = env.play(agent)
        summary['score'] = np.sum(summary["rewards"])
        print("Episode {}, mean score {}".format(episode_idx,summary['score']))
        print("n_iter {}".format(summary['rewards'].shape[0]))

        # Update agent
        agent.update(summary["observations"],summary["actions"],summary["rewards"])
        # print(agent.policy(torch.from_numpy(np.random.random(env.num_observations())).type(torch.FloatTensor)))
        