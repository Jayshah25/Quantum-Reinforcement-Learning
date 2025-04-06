# qrl/env/cleaning_robot_v0.py

import pygame
import numpy as np
import imageio
import random
from qrl.env.base import BaseEnv

class CleaningRobotV0(BaseEnv):
    def __init__(self, grid_size=8, dirt_count=10, max_steps=200):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = 60
        self.window_size = self.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size + 50))
        pygame.display.set_caption("Cleaning Robot")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.agent_pos = [0, 0]
        self.dirt_count = dirt_count
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.steps = 0
        self.done = False
        self.dirt_cells = set()
        self.frames = []  

        while len(self.dirt_cells) < self.dirt_count:
            cell = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if cell != tuple(self.agent_pos):
                self.dirt_cells.add(cell)

        return self._get_obs()


    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, {}

        self.steps += 1

        x, y = self.agent_pos
        if action == "UP" and y > 0:
            y -= 1
        elif action == "DOWN" and y < self.grid_size - 1:
            y += 1
        elif action == "LEFT" and x > 0:
            x -= 1
        elif action == "RIGHT" and x < self.grid_size - 1:
            x += 1

        self.agent_pos = [x, y]

        # ✅ Automatically clean if on dirt
        if tuple(self.agent_pos) in self.dirt_cells:
            self.dirt_cells.remove(tuple(self.agent_pos))
            reward = 1  # Optional: reward for cleaning
        else:
            reward = 0

        if len(self.dirt_cells) == 0 or self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, {}


    def render(self, capture=False):
        self.screen.fill((255, 255, 255))

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (230, 230, 230), rect, 0)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        for x, y in self.dirt_cells:
            dirt_rect = pygame.Rect(x * self.cell_size + 10, y * self.cell_size + 10,
                                    self.cell_size - 20, self.cell_size - 20)
            pygame.draw.rect(self.screen, (139, 69, 19), dirt_rect)

        ax, ay = self.agent_pos
        agent_rect = pygame.Rect(ax * self.cell_size + 15, ay * self.cell_size + 15,
                                self.cell_size - 30, self.cell_size - 30)
        pygame.draw.ellipse(self.screen, (0, 100, 255), agent_rect)

        text = self.font.render(f"Steps: {self.steps} | Dirt Remaining: {len(self.dirt_cells)}", True, (0, 0, 0))
        self.screen.blit(text, (10, self.window_size + 10))

        pygame.display.flip()

        if capture:
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))  # Pygame's format is (width, height)
            self.frames.append(frame)

        self.clock.tick(10)

    def save_video(self, filename="cleaning_robot.mp4", fps=10):
        if self.frames:
            imageio.mimsave(filename, self.frames, fps=fps)
            print(f"Video saved as {filename}")
        else:
            print("No frames to save.")



    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

    def close(self):
        pygame.quit()

    def _get_obs(self):
        # Return agent position and dirt state as a tuple
        return {
            "agent": tuple(self.agent_pos),
            "dirt": list(self.dirt_cells)
        }

    def action_space(self):
        return self.actions

    def observation_space(self):
        return {"agent": (0, 0), "dirt": []}