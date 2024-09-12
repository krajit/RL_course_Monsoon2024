import pygame

pygame.init()

# Set up window
width, height = 800, 600
screen = pygame.display.set_mode((width, height))

# Colors
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(black)

    # Simulate two screens
    pygame.draw.rect(screen, red, (0, 0, width // 2, height))  # Left side (first screen)
    pygame.draw.rect(screen, blue, (width // 2, 0, width // 2, height))  # Right side (second screen)

    pygame.display.flip()

pygame.quit()
