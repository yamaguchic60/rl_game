import pygame
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

pygame.init()

# パラメータの設定
clock_rate = 6000
WIDTH, HEIGHT = 800, 800
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Multi-Agent RL Game")

# 色の設定
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# エージェントの設定
agent_size = 20
agent_speed = 20
num_agents = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device')
agents = [{'pos': [random.randint(0, WIDTH - agent_size), random.randint(0, HEIGHT - agent_size)], 'color': RED, 'q_table': torch.zeros((WIDTH // agent_speed, HEIGHT // agent_speed, 4), device=device)} for _ in range(num_agents)]

# 目標の設定
goal_size = 20
goal_pos = [random.randint(0, WIDTH - goal_size), random.randint(0, HEIGHT - goal_size)]

# Q学習の設定
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# リアルタイムプロットの設定
time_intervals = []
fig, ax = plt.subplots()
line, = ax.plot(time_intervals)

def choose_action(state, q_table):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return actions[torch.argmax(q_table[state]).item()]

def update_q_table(state, action, reward, next_state, q_table):
    action_index = actions.index(action)
    best_next_action = torch.argmax(q_table[next_state]).item()
    td_target = reward + gamma * q_table[next_state][best_next_action]
    td_error = td_target - q_table[state][action_index]
    q_table[state][action_index] += alpha * td_error

def get_state(agent_pos, goal_pos):
    relative_x = (goal_pos[0] - agent_pos[0]) // agent_speed
    relative_y = (goal_pos[1] - agent_pos[1]) // agent_speed
    return (relative_x, relative_y)

def update_plot(frame):
    line.set_ydata(time_intervals)
    line.set_xdata(range(len(time_intervals)))
    ax.relim()
    ax.autoscale_view()
    return line,

running = True
time_interval = 0
time_before = pygame.time.get_ticks()

ani = FuncAnimation(fig, update_plot, blit=True)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for agent in agents:
        state = get_state(agent['pos'], goal_pos)
        action = choose_action(state, agent['q_table'])

        if action == 'UP' and agent['pos'][1] > 0:
            agent['pos'][1] -= agent_speed
        elif action == 'DOWN' and agent['pos'][1] < HEIGHT - agent_size:
            agent['pos'][1] += agent_speed
        elif action == 'LEFT' and agent['pos'][0] > 0:
            agent['pos'][0] -= agent_speed
        elif action == 'RIGHT' and agent['pos'][0] < WIDTH - agent_size:
            agent['pos'][0] += agent_speed

        reward = -1
        if agent['pos'][0] >= goal_pos[0] - int(goal_size / 2) and agent['pos'][0] <= goal_pos[0] + int(goal_size / 2) and agent['pos'][1] >= goal_pos[1] - int(goal_size / 2) and agent['pos'][1] <= goal_pos[1] + int(goal_size / 2):
            reward = 100
            time_interval = pygame.time.get_ticks() - time_before
            time_intervals.append(time_interval)
            print(f'Agent {agents.index(agent)} arrived!, the time interval is {time_interval} ms')
            goal_pos = [random.randint(0, WIDTH - goal_size), random.randint(0, HEIGHT - goal_size)]
            time_before = pygame.time.get_ticks()
            print(f'q_table: {agent["q_table"]}')

            # 到達したエージェントのQテーブルを他のエージェントにコピー
            for other_agent in agents:
                if other_agent != agent:
                    other_agent['q_table'] = agent['q_table'].clone()

        next_state = get_state(agent['pos'], goal_pos)
        update_q_table(state, action, reward, next_state, agent['q_table'])

    win.fill(WHITE)
    pygame.draw.rect(win, GREEN, (goal_pos[0], goal_pos[1], goal_size, goal_size))
    for agent in agents:
        pygame.draw.rect(win, agent['color'], (agent['pos'][0], agent['pos'][1], agent_size, agent_size))
    pygame.display.update()
    pygame.time.Clock().tick(clock_rate)

    plt.pause(0.001)

pygame.quit()
sys.exit()