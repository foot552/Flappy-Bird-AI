import pygame
import neat
import time
import os
import random

pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800

win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

STAT_FONT = pygame.font.SysFont("comicsans", 50)

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y): # was init_
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0 #time in jump
        self.height = self.y

    def move(self):
        self.tick_count += 1

        d = self.vel*self.tick_count + 1.5*self.tick_count**2 #how much movement up/down

        if d >= 16:
            d = 16

        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win): #bird flapping up and flapping down
        self.img_count += 1 #keep track of for how long we have shown a certain image

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2 #when flapping up again it starts with showing IMGS2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft= (self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)
class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP
    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))
    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point =  bird_mask.overlap(top_mask, top_offset)
        if t_point or b_point:
            return True

        return False
class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):   
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))
        

def draw_window(win, bird, pipes, base, score):
    win.blit(BG_IMG, (0,0))
    bird.draw(win)
    for pipe in pipes:
        pipe.draw(win)
    text = STAT_FONT.render("Score: " + str(score), 1,(255,255,255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()    
    

def main(genomes, config): #We can also use this as the fitness function(for multiple birds)
    nets = [] #Keeping track of bird that particular neural network controls
    ge = []   #Keeping track of genomes

    for _,g in genomes:  #This code block sets neural network for genomes
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))#appending bird object which is acted upon by this neural network
        g.fitness = 0
        ge.append(g)
    birds = [] #'bird' Variable is now a list coz we will have multiple of them
    base = Base(730)
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0
    
    

    run = True
    while run:
        base.move()
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()


        pipe_ind = 0  #Just a small block of code for preventing an error
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x + pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False #If no birds left , then quit game
            break

        for x, bird in enumerate(birds): #Making every bird in our list to move
            bird.move()
            ge[x].fitness += 0.1  #For each second a bird stays alive, we add 0.1 fitness so that its encouraged to stay alive
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom))) # nets stands for neural network...So we r feeding the neural network our bird y poition , distance between bird and top pipe and distance between bird and bottom pipe...abs() gives absolute value of the number as distance is alsways positive...The output is given by NEAT and the CONFIG file
            if ouput[0] > 0.5: #As output neurons is a list , we put [0]
                bird.jump()
        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds): #For birds x,  in enumerate(birds)(new addition) because now bird is a list
                if pipe.collide(bird):
                    ge[x].fitness -= 1 #Every time a bird hits a pipe, fitness dcreases by 1
                    birds.remove(bird) #Removing bird that collides with pipe
                    birds.pop(x) #Last three code bloacks removes data of bird that collides...the pop() functions removes these data from the list
                    nets.pop(x)
                    ge.pop(x)
                    
            
                if not pipe.passed and pipe.x < bird.x:#Now this block was outside the loop(FOR BIRD IN BIRDS)..but now , as there r multiple birds , it is part of this loop
                    pipe.passed = True
                    add_pipe = True
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:#Now this block was inside(FOR BIRD IN BIRDS) block...but now it is a pipe, not a bird so its outside now
                rem.append(pipe)      
            pipe.move()
        if add_pipe:
            score += 1
            for g in ge:#Gets the names of birds whose score increased by 1 , and adds 5 to fitness level
                g.fitness += 5
            pipes.append(Pipe(700))
        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):  #New loop here for the below block of code    
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)#Removing each bird that touches ground and touches sky
                nets.pop(x)
                ge.pop(x)
                   
           
        base.move()
        draw_window(win, birds, pipes, base, score)
   



def run(config_path):  #Another Block of code required by NEAT
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_path)#In this small passage we define all sub-headings in CONFIG File we will be using...Example- Default Reproduction...etc            
    p = neat.Population(config) #Creating the population
    p.add_reporter(neat.StdOutReporter(True)) #Just a code block that gives us a few stats while running the code
    stats = neat.StatisticsReporter()
    p.add_addreporter(stats) #Last 3 lines just print stats while running code
    
    winner = p.run(main,50) #Defining our fitness function and how many Generations we will run .."Main" tells our function and 50 tells how many generations we will run 
if __name__ == "__main__":   #Block of code required by NEAT to load CONFIG file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.txt")
    run(config_path)
