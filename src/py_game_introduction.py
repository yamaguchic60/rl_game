import pygame
import sys

pygame.init()

# set parameter here
clock_rate = 60


# Set up the display
WIDTH, HEIGHT = 800, 800
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Game")

# Set up the colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Set up the ball
ball_radius = 20
ball_speed = [5, 1]
ball_pos = [WIDTH // 2, HEIGHT // 2]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
    ball_pos[0] += ball_speed[0]
    ball_pos[1] += ball_speed[1]


    if ball_pos[0]<ball_radius or ball_pos[0]>WIDTH-ball_radius:
        ball_speed[0] = -ball_speed[0]
    if ball_pos[1]<ball_radius or ball_pos[1]>HEIGHT-ball_radius:
        ball_speed[1] = -ball_speed[1]
    win.fill(WHITE)

    pygame.draw.circle(win, RED, ball_pos, ball_radius)
    pygame.display.update()
    pygame.time.Clock().tick(clock_rate)
pygame.quit()
sys.exit()