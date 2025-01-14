import numpy as np
import pygame
import time

pygame.init()

maze = [
    [0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 2, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1]
]

maze_rows = len(maze)
maze_cols = len(maze[0])
q_values = np.zeros((maze_rows, maze_cols, 4))

actions = ['up', 'down', 'left', 'right']

rewards = np.zeros((maze_rows, maze_cols))
for rows in range(maze_rows):
    for cols in range(maze_cols):
        if maze[rows][cols] == 0:
            rewards[rows][cols] = -1
        elif maze[rows][cols] == 1:
            rewards[rows][cols] = -100
        elif maze[rows][cols] == 2:
            rewards[rows][cols] = 100

CELL_SIZE = 80
PADDING = 2
WINDOW_SIZE = (len(maze[0]) * CELL_SIZE, len(maze) * CELL_SIZE)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Q-Learning Maze Solver")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

def draw_maze(agent_pos=None):
    screen.fill(WHITE)
    for i in range(maze_rows):
        for j in range(maze_cols):
            x = j * CELL_SIZE
            y = i * CELL_SIZE
            cell_rect = pygame.Rect(x + PADDING, y + PADDING, 
                                  CELL_SIZE - 2*PADDING, CELL_SIZE - 2*PADDING)

            if maze[i][j] == 1:
                pygame.draw.rect(screen, BLACK, cell_rect)
            elif maze[i][j] == 2:
                pygame.draw.rect(screen, GREEN, cell_rect)
            else:  
                pygame.draw.rect(screen, WHITE, cell_rect, 1)

    if agent_pos:
        agent_x = agent_pos[1] * CELL_SIZE + CELL_SIZE//4
        agent_y = agent_pos[0] * CELL_SIZE + CELL_SIZE//4
        pygame.draw.circle(screen, RED, (agent_x + CELL_SIZE//4, agent_y + CELL_SIZE//4), 
                         CELL_SIZE//4)

    pygame.display.flip()

def is_terminal_state(current_row_index, current_col_index):
    return maze[current_row_index][current_col_index] == 2

def get_next_action(current_row_index, current_col_index, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        return np.argmax(q_values[current_row_index, current_col_index])

def get_next_location(current_row_index, current_col_index, chosen_action):
    new_posx = current_row_index
    new_posy = current_col_index
    if actions[chosen_action] == 'up' and current_row_index > 0:
        if maze[current_row_index-1][current_col_index] != 1:
            new_posx -= 1
    elif actions[chosen_action] == 'down' and current_row_index < maze_rows - 1:
        if maze[current_row_index+1][current_col_index] != 1:
            new_posx += 1
    elif actions[chosen_action] == 'left' and current_col_index > 0:
        if maze[current_row_index][current_col_index-1] != 1:
            new_posy -= 1
    elif actions[chosen_action] == 'right' and current_col_index < maze_cols - 1:
        if maze[current_row_index][current_col_index+1] != 1:
            new_posy += 1
    return new_posx, new_posy

def get_shortest_path():
    current_x, current_y = 0, 0
    shortest_path = []
    shortest_path.append([current_x, current_y])
    while not is_terminal_state(current_x, current_y):
        action_take = get_next_action(current_x, current_y, 0.0)
        current_x, current_y = get_next_location(current_x, current_y, action_take)
        shortest_path.append([current_x, current_y])
    return shortest_path

learning_rate = 0.9
discount_factor = 0.95
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.01
epochs = 1000

for episode in range(epochs):
    agent_x, agent_y = 0, 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    while not is_terminal_state(agent_x, agent_y):

        draw_maze((agent_x, agent_y))

        action_chosen = get_next_action(agent_x, agent_y, epsilon)
        old_posx, old_posy = agent_x, agent_y
        agent_x, agent_y = get_next_location(old_posx, old_posy, action_chosen)
        reward = rewards[agent_x, agent_y]
        old_q_value = q_values[old_posx, old_posy, action_chosen]
        temporal_difference = reward + (discount_factor * np.max(q_values[agent_x, agent_y])) - old_q_value
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_posx, old_posy, action_chosen] = new_q_value

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 1000 == 0:
        print(f"Episode {episode}/{epochs}")

print("Training complete!")

def visualize_shortest_path():
    path = get_shortest_path()
    print("Showing shortest path:", path)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for pos in path:

            draw_maze(pos)
            pygame.time.wait(500)  

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

        pygame.time.wait(2000)
        running = False

    pygame.quit()

visualize_shortest_path()
