import pygame
import numpy as np

class SimpleGridEnv:
    def __init__(self, grid_size=(5, 5), start_pos=(0, 0), goal_pos=(4, 4), obstacles=[]):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles
        self.reset()
        self.done = False

        pygame.init()
        self.screen = pygame.display.set_mode((500, 500))
        pygame.display.set_caption("Simple Grid World")

    def reset(self):
        self.player_pos = list(self.start_pos)
        self.done = False
        return self.player_pos

    def step(self, action):
        if self.done:
            return self.player_pos, 0, True, {}

        if action == 0:  # Up
            self.player_pos[1] = max(0, self.player_pos[1] - 1)
        elif action == 1:  # Down
            self.player_pos[1] = min(self.grid_size[1] - 1, self.player_pos[1] + 1)
        elif action == 2:  # Left
            self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif action == 3:  # Right
            self.player_pos[0] = min(self.grid_size[0] - 1, self.player_pos[0] + 1)

        if tuple(self.player_pos) == self.goal_pos:
            reward = 1
            self.done = True
        elif tuple(self.player_pos) in self.obstacles:
            reward = -1
            self.done = True
        else:
            reward = 0

        return self.player_pos, reward, self.done, {}

    def render(self):
        self.screen.fill((0, 0, 0))
        grid_width = self.screen.get_width() / self.grid_size[0]
        grid_height = self.screen.get_height() / self.grid_size[1]

        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                rect = pygame.Rect(x * grid_width, y * grid_height, grid_width, grid_height)
                color = (255, 255, 255)  # Default to white
                if (x, y) == self.goal_pos:
                    color = (0, 255, 0)  # Goal is green
                elif (x, y) in self.obstacles:
                    color = (255, 0, 0)  # Obstacles are red
                elif (x, y) == tuple(self.player_pos):
                    color = (0, 0, 255)  # Player is blue

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        pygame.display.flip()

    def close(self):
        pygame.quit()

# Example usage
if __name__ == "__main__":
    env = SimpleGridEnv(obstacles=[ (2, 2), (3, 3)])
    env.reset()
    env.render()

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = np.random.randint(1,4)
        _, _, done, _ = env.step(action)
        env.render()  
        
    

    env.close()
