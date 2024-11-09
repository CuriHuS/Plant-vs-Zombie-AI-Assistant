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
# from .evaluate_agent import evaluate
# from .threshold import Threshold

import pygame

HP_NORM = 100
SUN_NORM = 200

frame_pos = None
frame_lane = None
frame_no_plant = None
plant_list = ['peashooter', 'sunflower', 'wallnut', 'potatomine']
is_selection_window_open = False  # 선택 창 상태를 위한 변수
click_count = 0

class PolicyNetV2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=50):
        super(PolicyNetV2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class ReinforceAgentV2():
    def __init__(self,input_size, possible_actions):
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self.possible_actions = possible_actions
        self.policy = PolicyNetV2(input_size, output_size=len(possible_actions))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3,)
        self.n_plants = 4

    def decide_action(self, observation):
        mask = self._get_mask(observation)
        # predict probabilities for actions
        var_s = Variable(torch.from_numpy(observation.astype(np.float32)))
        action_prob = torch.exp(self.policy.forward(var_s))
        action_prob[np.logical_not(mask)] = 0
        action_prob /= torch.sum(action_prob[mask])
        # select random action weighted by probabilities
        action =  np.random.choice(self.possible_actions, 1, p=action_prob.data.numpy())[0]
        return action

    def discount_rewards(self,r,gamma):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.shape[0])):
            running_add = running_add * gamma + r[t][0]
            discounted_r[t][0] = running_add
        return discounted_r

    def iterate_minibatches(self, observation, actions, rewards, batchsize, shuffle=False):
        assert len(observation) == len(actions)
        assert len(observation) == len(rewards)

        indices = np.arange(len(observation))
        if shuffle:
            np.random.shuffle(indices)
        #import pdb; pdb.set_trace()
        for start_idx in range(0, len(observation), batchsize):
            if shuffle:
                excerpt = indices[start_idx:min(start_idx + batchsize, len(indices))]
            yield observation[excerpt], actions[excerpt], rewards[excerpt]

    def update(self,observation, actions, rewards):
        # discounted reward
        rewards = self.discount_rewards(rewards, gamma = 0.99)
        self.optimizer.zero_grad()
        # L = log π(a | s ; θ)*A
        loss = 0
        for observation_batch, action_batch, reward_batch in self.iterate_minibatches(observation, actions, rewards, batchsize = 100, shuffle=True):
            #import pdb; pdb.set_trace()
            mask_batch = torch.Tensor([self._get_mask(s) for s in observation_batch]).type(torch.BoolTensor).detach()
            
            s_var =  Variable(torch.from_numpy(observation_batch.astype(np.float32)))
            a_var = Variable(torch.from_numpy(action_batch).view(-1).type(torch.LongTensor))
            A_var = Variable(torch.from_numpy(reward_batch.astype(np.float32)))
            
            pred = self.policy.forward(s_var)
            pred = pred / torch.Tensor([torch.sum(pred[i,:][mask_batch[i,:]]) for i in range(len(pred))]).view(-1,1)
            
            loss += F.nll_loss(pred * A_var,a_var)

        loss.backward(loss)
        self.optimizer.step()

    def save(self, nn_name):
        torch.save(self.policy, nn_name)

    def load(self, nn_name):
        self.policy = torch.load(nn_name)

    def _get_mask(self, observation):
        empty_cells = np.nonzero((observation[:self._grid_size]==0).reshape(config.N_LANES, config.LANE_LENGTH))
        mask = np.zeros(len(self.possible_actions), dtype=bool)
        mask[0] = True
        empty_cells = (empty_cells[0] + config.N_LANES * empty_cells[1]) * self.n_plants

        available_plants = observation[-self.n_plants:]
        for i in range(len(available_plants)):
            if available_plants[i]:
                idx = empty_cells + i + 1
                mask[idx] = True
        return mask




class PlayerV2():
    DECIDE = False
    def __init__(self,render=True, max_frames = 1000, n_iter = 100000):
        self.env = gym.make('gym_pvz:pvz-env-v2')
        self.max_frames = max_frames
        self.render = render
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self.lane = None
        self.pos = None
        self.no_plant = None
        self.signal = True

    def get_actions(self):
        return list(range(self.env.action_space.n))
    
    def take_action_for_env(self,action):
        self.env.take_action(action)
        return 200

    def num_observations(self):
        return config.N_LANES * config.LANE_LENGTH + config.N_LANES + len(self.env.plant_deck) + 1

    def num_actions(self):
        return self.env.action_space.n

    def _transform_observation(self, observation):
        observation = observation.astype(np.float64)
        observation_zombie = self._grid_to_lane(observation[self._grid_size:2*self._grid_size])
        observation = np.concatenate([observation[:self._grid_size], observation_zombie, 
        [observation[2 * self._grid_size]/SUN_NORM], 
        observation[2 * self._grid_size+1:]])
        if self.render:
            print(observation)
        return observation

    def _grid_to_lane(self, grid):
        grid = np.reshape(grid, (config.N_LANES, config.LANE_LENGTH))
        return np.sum(grid, axis=1)/HP_NORM

    def play(self, agent, epsilon=0):
        summary = {'rewards': [], 'observations': [], 'actions': []}
        observation = self._transform_observation(self.env.reset())
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((1450, 650))
        myfont = pygame.font.SysFont('calibri', 30)
        zombie_sprite = {"zombie": pygame.image.load("assets/zombie_scaled.png").convert_alpha(),
                    "zombie_cone": pygame.image.load("assets/zombie_cone_scaled.png").convert_alpha(),
                    "zombie_bucket": pygame.image.load("assets/zombie_bucket_scaled.png").convert_alpha(),
                    "zombie_flag": pygame.image.load("assets/zombie_flag_scaled.png").convert_alpha()}
        plant_sprite = {"peashooter": pygame.image.load("assets/peashooter_scaled.png").convert_alpha(),
                        "sunflower": pygame.image.load("assets/sunflower_scaled.png").convert_alpha(),
                        "wallnut": pygame.image.load("assets/wallnut_scaled.png").convert_alpha(),
                        "potatomine": pygame.image.load("assets/potatomine_scaled.png").convert_alpha()}
        projectile_sprite = {"pea": pygame.image.load("assets/pea.png").convert_alpha()}
        #render_frame(screen, self.get_render_info(), clock, myfont, zombie_sprite, plant_sprite, projectile_sprite)
        #pygame.event.get()
        while self.env._scene._chrono < self.max_frames * config.FPS:
            clock.tick(config.FPS)
            if True:
                #print("화면 렌더링")
                render_frame(screen, self.get_render_info(), clock, myfont, zombie_sprite, plant_sprite, projectile_sprite, agent)
            
            if np.random.random() < epsilon:
                action = np.random.choice(self.get_actions(), 1)[0]
            else:
                #print("reinforce_agent_v2에서 action 설정")
                action = agent.decide_action(observation,0,0,0)
                #action = agent.decide_action(observation, self.lane, self.pos, self.no_plant)
                    
            summary['observations'].append(observation)
            summary['actions'].append(action)
            observation, reward, done, info = self.env.step(action)
            observation = self._transform_observation(observation)
            summary['rewards'].append(reward)
            
            if done:
                break

        pygame.quit()
        summary['observations'] = np.vstack(summary['observations'])
        summary['actions'] = np.vstack(summary['actions'])
        summary['rewards'] = np.vstack(summary['rewards'])
        return summary

    def get_render_info(self):
        return self.env._scene._render_info

flag = 0

def render_frame(screen, render_info, clock, myfont, zombie_sprite, plant_sprite, projectile_sprite, agent):
    global flag, is_selection_window_open, click_count, frame_lane, frame_pos, frame_no_plant
    clock.tick(config.FPS)
    screen.fill((130, 200, 100))
    if render_info is not None:
        frame_info = render_info[flag]
        flag += 1
        flag = len(render_info) - 1
        frame_info = render_info[flag]
        print("렌더 프레임 수: ", len(render_info))
        print("진행 프레임 수: ", flag)
    else:
        None
    #frame_info = render_info[0] if render_info else None
   
    if frame_info:
        print(frame_info)
        #print("render_frame 그리기")
        cell_size = 75
        offset_border = 100
        offset_y = int(0.8 * cell_size)
        for i in range(config.LANE_LENGTH + 1):
            pygame.draw.line(screen, (0, 0, 0), (offset_border + i * cell_size, offset_border),
                             (offset_border + i * cell_size, offset_border + cell_size * config.N_LANES), 1)
        for j in range(config.N_LANES + 1):
            pygame.draw.line(screen, (0, 0, 0), (offset_border, offset_border + j * cell_size),
                             (offset_border + cell_size * config.LANE_LENGTH, offset_border + j * cell_size), 1)

        for lane in range(config.N_LANES):
            for zombie_name, pos, offset in frame_info["zombies"][lane]:
                zombie_name = zombie_name.lower()
                screen.blit(zombie_sprite[zombie_name],
                            (offset_border + cell_size * (pos + offset) - zombie_sprite[zombie_name].get_width(),
                            offset_border + lane * cell_size + offset_y - zombie_sprite[zombie_name].get_height()))
            for plant_name, pos in frame_info["plants"][lane]:
                plant_name = plant_name.lower()
                screen.blit(plant_sprite[plant_name],
                            (offset_border + cell_size * pos,
                            offset_border + lane * cell_size + offset_y - plant_sprite[plant_name].get_height()))
            for projectile_name, pos, offset in frame_info["projectiles"][lane]:
                projectile_name = projectile_name.lower()
                screen.blit(projectile_sprite[projectile_name],
                            (offset_border + cell_size * (pos + offset) - projectile_sprite[projectile_name].get_width(),
                            offset_border + lane * cell_size))

        sun_text = myfont.render(f'Sun: {frame_info["sun"]}', False, (0, 0, 0))
        score_text = myfont.render(f'Score: {frame_info["score"]}', False, (0, 0, 0))
        cooldowns_text = myfont.render(f'Cooldowns: {frame_info["cooldowns"]}', False, (0, 0, 0))
        time_text = myfont.render(f'Time: {frame_info["time"]}', False, (0, 0, 0))
        screen.blit(sun_text, (50, 600))
        screen.blit(score_text, (200, 600))
        screen.blit(cooldowns_text, (350, 600))
        screen.blit(time_text, (900, 100))

            # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                render_info = []
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN and not is_selection_window_open:  # 클릭 시
                click_count += 1
                mouse_x, mouse_y = pygame.mouse.get_pos()

                grid_x = (mouse_x - offset_border) // cell_size
                grid_y = (mouse_y - offset_border) // cell_size


                if 0 <= grid_x < config.LANE_LENGTH and 0 <= grid_y < config.N_LANES:
                    frame_lane = grid_y
                    frame_pos = grid_x
                    is_selection_window_open = True
                    print("grid를 lane, pos에 적용합니다.")
                print("grid 클릭 pos: ", frame_pos, "lane: ", frame_lane)
            
            elif event.type == pygame.MOUSEBUTTONDOWN and is_selection_window_open: #창이 오픈되어 있을 때.
                
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for idx, plant in enumerate(plant_list):
                    # 선택 창의 버튼 클릭 시
                    if 500 + idx * 100 <= mouse_x <= 500 + (idx + 1) * 100 and 300 <= mouse_y <= 400:
                        click_count = 1
                        if idx is 0:
                            frame_no_plant = 1
                        elif idx is 1: 
                            frame_no_plant = 0
                        else:
                            frame_no_plant = idx
                        
                        #click_count = agent.decide_action(0,0,0,0)
                        #click_count = agent.decide_action(0,0,0,0) + 100 # 리턴 값 101나옴 그러면, decide_action 리턴이 1이라는 뜻인데..
                agent.set_parameters(frame_lane,frame_pos, frame_no_plant)
                is_selection_window_open = False  # 선택 창 닫기
                print("식물 설치 pos: ", frame_pos, "lane: ", frame_lane, "no_plant: ", frame_no_plant)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # ESC 키로 창 닫기
                    is_selection_window_open = False


        # 선택 창 그리기 추가
        if is_selection_window_open:
            # 선택 창 배경 그리기
            pygame.draw.rect(screen, (200, 200, 200), (400, 200, 600, 300))
            pygame.draw.rect(screen, (0, 0, 0), (400, 200, 600, 300), 2)
            
            # 각 Plant 이미지와 버튼 그리기
            for idx, plant in enumerate(plant_list):
                screen.blit(plant_sprite[plant], (500 + idx * 100, 250))  # Plant 이미지
                pygame.draw.rect(screen, (0, 0, 0), (500 + idx * 100, 300, 100, 50), 2)  # 선택 버튼
                button_text = myfont.render('Select', False, (0, 0, 0))
                screen.blit(button_text, (520 + idx * 100, 310))  # 버튼 텍스트

    pygame.display.flip()

if __name__ == "__main__":

    env = PlayerV2(render=False,max_frames = 1000)
    agent = DiscreteAgentV2(
        input_size=env.num_observations(),
        possible_actions=env.get_actions()
    )
    # agent.policy = torch.load("saved/policy13_v2")
    sum_score = 0
    sum_iter = 0
    score_plt = []
    iter_plt = []
    eval_score_plt = []
    eval_iter_plt = []
    n_iter = 100000
    n_record = 500
    n_save = 1000
    n_evaluate = 10000
    n_iter_evaluation = 1000
    save = False
    best_score = None
    #threshold = Threshold(seq_length = n_iter, start_epsilon=0.1,
    #                      end_epsilon=0.0,interpolation='sinusoidal',
    #                      periods=np.floor(n_iter/(8*n_record)))
    # threshold = Threshold(seq_length = n_iter, start_epsilon=0.005, end_epsilon=0.005)

    for episode_idx in range(n_iter):
        
        # play episodes
        # epsilon = threshold.epsilon(episode_idx)
        summary = env.play(agent)
        summary['score'] = np.sum(summary["rewards"])
        # print("Episode {}, mean score {}".format(episode_idx,summary['score']))
        # print("n_iter {}".format(summary['rewards'].shape[0]))

        sum_score += summary['score']
        sum_iter += min(env.env._scene._chrono, env.max_frames)

        # Update agent
        agent.update(summary["observations"],summary["actions"],summary["rewards"])
        # print(agent.policy(torch.from_numpy(np.random.random(env.num_observations())).type(torch.FloatTensor)))

        if (episode_idx%n_record == n_record-1):
            if save:
                if sum_score >= best_score:
                    torch.save(agent.policy, nn_name)
                    best_score = sum_score
            print("---Episode {}, mean score {}".format(episode_idx,sum_score/n_record))
            print("---n_iter {}".format(sum_iter/n_record))
            score_plt.append(sum_score/n_record)
            iter_plt.append(sum_iter/n_record)
            sum_iter = 0
            sum_score = 0
            # input()
        if not save:
            if (episode_idx%n_save == n_save-1):
                s = input("Save? (y/n): ")
                if (s=='y'):
                    save = True
                    best_score = 0
                    nn_name = input("Save name: ")

        if (episode_idx%n_evaluate == n_evaluate-1):
            avg_score, avg_iter = evaluate(env, agent, n_iter_evaluation)
            print("\n----------->Episode {}, mean score {}".format(episode_idx,avg_score))
            print("----------->n_iter {}".format(avg_iter))
            eval_score_plt.append(avg_score)
            eval_iter_plt.append(avg_iter)
            # input()
        

    plt.plot(range(n_record, n_iter+1, n_record), score_plt)
    plt.plot(range(n_evaluate, n_iter+1, n_evaluate), eval_score_plt, color='red')
    plt.show()
    plt.plot(range(n_record, n_iter+1, n_record), iter_plt)
    plt.plot(range(n_evaluate, n_iter+1, n_evaluate), eval_iter_plt, color='red')
    plt.show()