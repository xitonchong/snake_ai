import pygame 

pygame.init() 
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock() 
running = True 

while running: 
    # poll for events 
    # pygame.QUIT evne tmeans the user clicked X to close your window 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running=False 

    # fill the screen with a color to wip away anything from last frame 
    screen.fill('purple')


    # RENDER YOUR GAME 
    pygame.display.flip() 

    clock.tick(60) 
pygame.quit() 