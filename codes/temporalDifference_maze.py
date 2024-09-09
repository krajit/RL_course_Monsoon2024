import pygame
import sys
import itertools, time
import numpy as np
 
# Initialize Pygame
pygame.init()

# Define constants
WIDTH, HEIGHT = 500, 500  # 5x5 grid with each square 50x50 pixels
SQUARE_SIZE = 100
GRID_SIZE = 5

# Create the display surface
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('5x5 Grid with Triangles')

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Define color gradient
START_COLOR = pygame.Color('red')
END_COLOR = pygame.Color('blue')

def interpolate_color(start_color, end_color, t):
    """ Interpolate between start_color and end_color based on t (0 to 1) """
    r = int(start_color.r + (end_color.r - start_color.r) * t)
    g = int(start_color.g + (end_color.g - start_color.g) * t)
    b = int(start_color.b + (end_color.b - start_color.b) * t)
    return pygame.Color(r, g, b)

def value_to_color(x):
    """ Convert the value of x to a color """
    t = (x + 10) / 10  # Normalize x from -10 to 0 to a range from 0 to 1
    return interpolate_color(START_COLOR, END_COLOR, t)

def draw_grid(screen, Q):
    font = pygame.font.Font(None, 24)  # Create a font object
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x_pos = col * SQUARE_SIZE
            y_pos = row * SQUARE_SIZE
            pygame.draw.rect(screen, WHITE, (x_pos, y_pos, SQUARE_SIZE, SQUARE_SIZE), 1)
            
            center_x = x_pos + SQUARE_SIZE // 2
            center_y = y_pos + SQUARE_SIZE // 2
            
            # Define triangles
            triangles = [
                [(x_pos, y_pos), (center_x, center_y), (x_pos + SQUARE_SIZE, y_pos)],
                [(x_pos + SQUARE_SIZE, y_pos), (center_x, center_y), (x_pos + SQUARE_SIZE, y_pos + SQUARE_SIZE)],
                [(x_pos + SQUARE_SIZE, y_pos + SQUARE_SIZE), (center_x, center_y), (x_pos, y_pos + SQUARE_SIZE)],
                [(x_pos, y_pos + SQUARE_SIZE), (center_x, center_y), (x_pos, y_pos)]
            ]
            
            # Get the color for the current x value
            
            
            # Draw triangles and add text
            for i, triangle in enumerate(triangles):
                color = value_to_color(Q[row,col,i])
                pygame.draw.polygon(screen, color, triangle)
                # Add text to the center of the triangle
                text = font.render(f'Tri {i+1}', True, BLACK)
                text_rect = text.get_rect(center=(center_x, center_y))
                screen.blit(text, text_rect)

def main():
    x = -10  # Initial value of x
    Q = {(s1,s2,a): -10*np.random.rand() for (s1,s2,a) in itertools.product(range(5),range(5),range(4))}
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and x < 0:
                    x += 1
                elif event.key == pygame.K_DOWN and x > -10:
                    x -= 1

        screen.fill(BLACK)  # Clear the screen with a black background
        draw_grid(screen, Q)  # Draw the grid and triangles
        pygame.display.flip()  # Update the display

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    
    main()
