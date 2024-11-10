import pygame
import sys
import numpy as np
import random

pygame.init()

# パラメータの設定
clock_rate = 60
WIDTH, HEIGHT = 400, 400
goal_size = 20
# ウィンドウの設定
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple RL Game")

# 色の設定
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# エージェントの設定
agent_size = 20
agent_pos = [WIDTH // 2, HEIGHT // 2]
agent_speed = 20

# 目標の設定

goal_pos = [random.randint(0, WIDTH - goal_size), random.randint(0, HEIGHT - goal_size)]

# Q学習の設定
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
q_table = np.zeros((WIDTH // agent_speed, HEIGHT // agent_speed, len(actions)))
alpha = 0.1
gamma = 0.9
epsilon = 0.1

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(q_table[state[0], state[1]])]

def update_q_table(state, action, reward, next_state):
    action_index = actions.index(action)
    best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
    td_target = reward + gamma * q_table[next_state[0], next_state[1], best_next_action]
    td_error = td_target - q_table[state[0], state[1], action_index]
    q_table[state[0], state[1], action_index] += alpha * td_error

def get_state(pos):
    return (pos[0] // agent_speed, pos[1] // agent_speed)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = get_state(agent_pos)
    action = choose_action(state)

    if action == 'UP' and agent_pos[1] > 0:
        agent_pos[1] -= agent_speed
    elif action == 'DOWN' and agent_pos[1] < HEIGHT - agent_size:
        agent_pos[1] += agent_speed
    elif action == 'LEFT' and agent_pos[0] > 0:
        agent_pos[0] -= agent_speed
    elif action == 'RIGHT' and agent_pos[0] < WIDTH - agent_size:
        agent_pos[0] += agent_speed

    reward = -1
    if agent_pos[0] >= goal_pos[0] -int((goal_size/2)) and agent_pos[0] <= goal_pos[0] + int((goal_size/2)) and agent_pos[1] >= goal_pos[1] - int((goal_size/2)) and agent_pos[1] <= goal_pos[1] + int((goal_size/2)):
        reward = 100
        print('Goal!')
        goal_pos = [random.randint(0, WIDTH - goal_size), random.randint(0, HEIGHT - goal_size)]

    next_state = get_state(agent_pos)
    update_q_table(state, action, reward, next_state)

    win.fill(WHITE)
    pygame.draw.rect(win, GREEN, (goal_pos[0], goal_pos[1], goal_size, goal_size))
    pygame.draw.rect(win, RED, (agent_pos[0], agent_pos[1], agent_size, agent_size))
    pygame.display.update()
    pygame.time.Clock().tick(clock_rate)

pygame.quit()
sys.exit()