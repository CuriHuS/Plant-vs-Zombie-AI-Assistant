from agents import ReinforceAgentV2, PolicyNetV2, PlayerV2, reinforce_agent_v2
from agents import KeyboardAgent
from agents import PlayerQ, QNetwork, QNetwork_DQN
from agents import ACAgent3, TrainerAC3
from pvz import config
import gym
import torch
import pygame

test_text_msg = "asd"
click_count = 0
mouse_x, mouse_y = 0, 0

selected_cells = [[False for _ in range(config.LANE_LENGTH)] for _ in range(config.N_LANES)]
is_selection_window_open = False  # 선택 창 상태를 위한 변수
selected_plant = None  # 선택된 Plant 저장
grid_x, grid_y = 0, 0  # 클릭한 그리드 좌표

no_plant = 0
lane = 0
pos = 0
decide = False

# Plant 리스트
plant_list = ['peashooter', 'sunflower', 'wallnut', 'potatomine']
class PVZ():
    def __init__(self, render=True, max_frames=1000):
        self.env = gym.make('gym_pvz:pvz-env-v2')
        self.max_frames = max_frames
        self.render = render
        self.action = None

    def get_actions(self):
        return list(range(self.env.action_space.n))

    def num_observations(self):
        return config.N_LANES * (config.LANE_LENGTH + 2)

    def num_actions(self):
        return self.env.action_space.n

    def play(self, agent):
        global click_count
        """ Play one episode and collect observations and rewards """
        observation = self.env.reset()
        t = 0

        for t in range(self.max_frames):
            if self.render:
                self.env.render()

            self.action = agent.decide_action(observation) # 이건 아닌 듯?
            observation, reward, done, info = self.env.step(self.action)
            if done:
                break

    def get_render_info(self):
        return self.env._scene._render_info

      
def render(render_info, env2):
    pvz_env = gym.make('gym_pvz:pvz-env-v2')
    global test_text_msg, click_count, mouse_x, mouse_y, is_selection_window_open, no_plant, lane, pos, decide
    pygame.init()
    pygame.font.init()
    myfont = pygame.font.SysFont('calibri', 30)

    screen = pygame.display.set_mode((1450, 650))
    zombie_sprite = {"zombie": pygame.image.load("assets/zombie_scaled.png").convert_alpha(),
                    "zombie_cone": pygame.image.load("assets/zombie_cone_scaled.png").convert_alpha(),
                    "zombie_bucket": pygame.image.load("assets/zombie_bucket_scaled.png").convert_alpha(),
                    "zombie_flag": pygame.image.load("assets/zombie_flag_scaled.png").convert_alpha()}
    plant_sprite = {"peashooter": pygame.image.load("assets/peashooter_scaled.png").convert_alpha(),
                    "sunflower": pygame.image.load("assets/sunflower_scaled.png").convert_alpha(),
                    "wallnut": pygame.image.load("assets/wallnut_scaled.png").convert_alpha(),
                    "potatomine": pygame.image.load("assets/potatomine_scaled.png").convert_alpha()}
    projectile_sprite = {"pea": pygame.image.load("assets/pea.png").convert_alpha()}
    clock = pygame.time.Clock()
    cell_size = 75
    offset_border = 100
    offset_y = int(0.8 * cell_size)
    cumulated_score = 0

    while render_info:
        clock.tick(config.FPS)
        screen.fill((130, 200, 100))
        frame_info = render_info.pop(0)

        # The grid
        for i in range(config.LANE_LENGTH + 1):
            pygame.draw.line(screen, (0, 0, 0), (offset_border + i * cell_size, offset_border),
                            (offset_border + i * cell_size, offset_border + cell_size * (config.N_LANES)), 1)
        for j in range(config.N_LANES + 1):
            pygame.draw.line(screen, (0, 0, 0), (offset_border, offset_border + j * cell_size),
                            (offset_border + cell_size * (config.LANE_LENGTH), offset_border + j * cell_size), 1)

        # Objects
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

        # Text
        test_text = myfont.render("Click Count: " + str(click_count), False, (0, 0, 0))
        screen.blit(test_text, (50, 500))
        test_text = myfont.render("Click Point: " + str(mouse_x) + " " + str(mouse_y), False, (0, 0, 0))
        screen.blit(test_text, (300, 500))

        sun_text = myfont.render('Sun: ' + str(frame_info["sun"]), False, (0, 0, 0))
        screen.blit(sun_text, (50, 600))
        cumulated_score += frame_info["score"]
        score_text = myfont.render('Score: ' + str(cumulated_score), False, (0, 0, 0))
        screen.blit(score_text, (200, 600))
        cooldowns_text = myfont.render('Cooldowns: ' + str(frame_info["cooldowns"]), False, (0, 0, 0))
        screen.blit(cooldowns_text, (350, 600))
        time = myfont.render('Time: ' + str(frame_info["time"]), False, (0, 0, 0))
        screen.blit(time, (900, 100))

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                render_info = []
            elif event.type == pygame.MOUSEBUTTONDOWN and not is_selection_window_open:  # 클릭 시
                click_count += 1
                mouse_x, mouse_y = pygame.mouse.get_pos()

                grid_x = (mouse_x - offset_border) // cell_size
                grid_y = (mouse_y - offset_border) // cell_size

                if 0 <= grid_x < config.LANE_LENGTH and 0 <= grid_y < config.N_LANES:
                    is_selection_window_open = True
            
            elif event.type == pygame.MOUSEBUTTONDOWN and is_selection_window_open: #창이 오픈되어 있을 때.
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for idx, plant in enumerate(plant_list):
                    # 선택 창의 버튼 클릭 시
                    if 500 + idx * 100 <= mouse_x <= 500 + (idx + 1) * 100 and 300 <= mouse_y <= 400:
                        click_count = 1
                        agent.set_parameters(0,0,0)
                        env2.set_parameters(agent, 0,0,0)
                        pvz_env.take_action(1)
                        print(env2.take_action_for_env(1))
                        env2.take_action_for_env(1)
                        #click_count = agent.decide_action(0,0,0,0)
                        #click_count = agent.decide_action(0,0,0,0) + 100 # 리턴 값 101나옴 그러면, decide_action 리턴이 1이라는 뜻인데..
                        is_selection_window_open = False  # 선택 창 닫기

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
    print("rendering 종료됨")
    pygame.quit()


agent_type = "Keyboard"  # DDQN or Reinforce or AC or Keyboard


if __name__ == "__main__":
    pygame.init()
    if agent_type == "Reinforce":
        env = PlayerV2(render=False, max_frames=500 * config.FPS)
        agent = ReinforceAgentV2(
            input_size=env.num_observations(),
            possible_actions=env.get_actions()
        )
        agent.load("agents/agent_zoo/dfq5_dqn")

    if agent_type == "DDQN":
        env = PlayerQ(render=False)
        agent = torch.load("agents/agent_zoo/dfq5_epsexp")

    if agent_type == "AC":
        env = TrainerAC3(render=False, max_frames=500 * config.FPS)
        agent = ACAgent3(
            input_size=env.num_observations(),
            possible_actions=env.get_actions()
        )
        agent.load("agents/agent_zoo/ac_policy_v1", "agents/agent_zoo/ac_value_v1")

    if agent_type == "Keyboard":
        env = PlayerV2(render=False, max_frames=1000 * config.FPS)
        agent = KeyboardAgent()

    env.play(agent)
    render_info = env.get_render_info()
    #render(render_info, env)