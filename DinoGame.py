from kivy.graphics import Rectangle, Color
import gc
import random as rd
from NetworkClasses import *
from typing import List
#

class DinoGameManager:
    # class that manages the computational aspect of a single dino game loop (backend)
    # (handles user input, spawns obstacles ...)

    # class constants
    MOD = 5 # change rate
    ERR = 1 # error amount
    GAME_SPEED = 10 # the speed of the game
    GAME_ACC = 250 # the acceleration of the game
    BIRD_HEIGHTS = [100, 65, 0] # all possible bird obstacle heights
    OUT_LEN = 4 # all possible outputs: duck,no_action,jump,unduck
    IN_LEN = 5 # all possible inputs: dist_x,dist_y,obst_height,ducking,jumping

    @staticmethod
    def Find_Dino_Images(path:str):
        # a static function receives a path string
        # the function loads the asset pngs, creates and returns a dictionary with the asset path
        # and the image size (dimensions)
        from pathlib import Path
        import re
        from PIL import Image

        p = Path(path)
        images = {}
        for path in p.iterdir():
            if not path.is_dir():
                continue
            for x in path.iterdir():
                x = str(x)
                name = re.sub('[0-9]?\.png', '', re.sub('.*/.*/', '', x))
                img = Image.open(x)
                if name in images:
                    images[name] += [{'path': x, 'size': (img.width, img.height)}]
                else:
                    images[name] = [{'path': x, 'size': (img.width, img.height)}]
        return images

    def __init__(self,canv_size=(800, 600),path='assets',img=None):
        # an init function for the game manager class
        # the function creates or receives the dictionary with the paths to the game assets (pngs)
        # the function also receives the size of the canvas
        self.time = 0 # current game time (also the score)
        self.mtime = 0 # tells the obstacles and the dino which animation frame to be on
        # this variable is in the backend class because the change affects the size and bounds of the dino and obstacle
        self.actions = [self.dino_unduck,self.dino_duck,self.dino_no_action,self.dino_jump] # an of possible actions

        # this variable is a dictionary of data about each image in the project
        if img is None:
            self.imgs = DinoGameManager.Find_Dino_Images(path)
        else:
            self.imgs = img

        self.inputs = None

        self.dino:Dino = Dino(self.imgs['DinoRun'][0]['size'],self.imgs['DinoDuck'][0]['size']) # dino object
        self.obstacles:List[Obstacle] = [] # obstacles array
        self.canv_size = canv_size # the size of the canvas

    def spawn(self):
        # adds random obstacle object to spawn queue
        rand1 = rd.randint(0, 2)
        rand = rd.randint(0, 2)
        o = None
        if rand1 == 0: # bird
            o = Obstacle((self.canv_size[0],DinoGameManager.BIRD_HEIGHTS[rand]),self.imgs['Bird'][0]['size'],'Bird',1)
        if rand1 == 1: # SmallCactus
            o = Obstacle((self.canv_size[0],0),self.imgs['SmallCactus'][rand]['size'],'SmallCactus', 0, type=rand)
        if rand1 == 2: # LargeCactus
            o = Obstacle((self.canv_size[0],0), self.imgs['LargeCactus'][rand]['size'], 'LargeCactus', 0, type=rand)

        self.obstacles.append(o)

    def updateAll(self, dt):
        # updates all computational aspects of a single game
        if not self.dino.enabled:
            return

        self.updateMtime(DinoGameManager.MOD)
        self.time += dt*DinoGameManager.GAME_SPEED*2

        if len(self.obstacles) == 0:
            self.spawn()

        self.dino.update(self.obstacles[0],dt,DinoGameManager.ERR)

        for obst in self.obstacles:
            if obst.update(-(DinoGameManager.GAME_SPEED+self.time//DinoGameManager.GAME_ACC),0,DinoGameManager.ERR):
                self.obstacles.remove(obst)
                del obst
                self.spawn()

    def updateMtime(self,t):
        # updates the m time variable based on passed change rate and current game time (score)
        self.mtime = int((self.time*3 // t) % 2)

    def get_inputs(self):
        # returns the inputs for the neural network gathered from data inside the game
        # such as distance to the closest obstacle and the current state of the dino
        self.inputs = np.ones((DinoGameManager.IN_LEN,1))
        if self.dino.enabled:
            self.inputs[0] = self.obstacles[0].x - self.dino.x # distance x
            self.inputs[1] = self.obstacles[0].y - self.dino.y # distance y
            self.inputs[2] = self.obstacles[0].size[1] # height of obstacle
            self.inputs[3] = self.dino.ducking # ducking
            self.inputs[4] = self.dino.jumping # jumping
            # self.inputs[5] = self.time # time
        return self.inputs

    def dino_not_dead(self):
        # checks if the dino is not dead yet (returns boolean)
        return self.dino.enabled

    def dino_unduck(self):
        # calls the unduck function inside the dino class which causes the dino to stop ducking
        self.dino.unduck()

    def dino_duck(self):
        # calls the duck function inside the dino class which causes the dino to duck
        self.dino.duck()

    def dino_jump(self):
        # class the jump function inside the dino class which causes the dino to jump
        self.dino.jump()

    def dino_no_action(self):
        # doesn't do anything but is an available action for the dino to take
        # (mostly for aesthetic and consistency reasons)
        pass

    def reset(self):
        # resets all used variables in the class got a new game
        self.time = 0
        self.mtime = 0
        self.obstacles = []
        self.dino.reset()

class DinoGameVisManager:
    # class that manages the visual aspects of a single dino game (draws and updates the canvas), (frontend)
    # connects the visual to the computational parts of the game

    def __init__(self,xoff,yoff,canvas,gm:DinoGameManager = None):
        # an init function for the visual game manager class
        # receives the x and y offset of game canvas (position on screen), a canvas object
        # and a game manager object that points to the game that needs to be displayed
        # initializes object to be drawn to the canvas
        self.canvas = canvas # canvas object
        self.xoff = xoff # the x offset of drawing to canvas
        self.yoff = yoff # the y offset of drawing to canvas
        self.gm = gm # game manager object

        self.dinov:DinoVis = None # visual dino object
        self.obstv:ObstacleVis = None # visual obstacle (closest to the dino) object

        self.init_canvas()
        self.set_dino()

    def init_canvas(self):
        # initializes the objects that are drawn to the canvas
        # e.g. the dino image and the obstacle image (initialized as blank)
        with self.canvas:
            # visual
            dino = Rectangle()
            self.dinov = DinoVis(dino)

            obst = Rectangle(pos=(-100,0))
            self.obstv = ObstacleVis(obst)

    def set_gm(self, gm:DinoGameManager):
        # sets the game manager to the passed game manager
        self.gm = gm

    def set_dino(self):
        # sets the dino in the visual dino class if possible
        if self.gm is None or self.dinov is None or self.obstv is None:
            return
        self.dinov.dino = self.gm.dino

    def set_obstacle(self):
        # sets the current obstacle if possible and returns success value
        if self.gm is None or self.dinov is None or self.obstv is None:
            return False

        if len(self.gm.obstacles) == 0:
            return False

        self.obstv.obst = self.gm.obstacles[-1]
        return True

    def draw_all(self):
        # updates all visual aspects, calls the draw functions in the dino and obstacle classes
        if self.canvas is None or self.gm is None:
            return
        # if not self.gm.dino.enabled:
        #     return
        self.dinov.draw(self.gm,self.gm.mtime,self)
        if self.set_obstacle():
            self.obstv.draw(self.gm,self.gm.mtime,self)

    def key_up(self, keycode):
        # function receives user input if and what keyboard key was released
        # if key is the down arrow the function tells the game manager to unduck the dino
        if keycode[1] == "down" or keycode[1] == 's':
            self.gm.dino_unduck()

    def key_down(self, keycode):
        # function receives user input if and what keyboard key was pressed
        # if key is down arrow the function tells the game manager to duck the dino
        # if key is up arrow the function tells the game manager to make the dino jump
        if keycode[1] == 'spacebar' or keycode[1] == 'up' or keycode[1] == 'w':
            self.gm.dino_jump()
        if keycode[1] == "down" or keycode[1] == 's':
            self.gm.dino_duck()

    def reset(self):
        # resets all needed variables in class for a new game
        self.obstv.rect.pos = (-100,0)
        self.gm.reset()


class Dino:
    # class that manages the computational aspect of the dino player, (backend)

    JUMP_VEL = 20 # the velocity of the jump
    JUMP_ACC = -1 # the acceleration of the fall
    DUCK_ACC = -1.5 # the acceleration of the fall while ducking

    def __init__(self,size,duck): # size -> (width,height)
        # init function for the dino class
        # receives the size of the dino and the size of the dino while ducking (bounds)
        self.x = 0 # x postion
        self.y = 0 # current y position
        self.prevY = 0 # y position in previous frame

        self.psize = [size,duck] # possible sizes for the dino
        self.size = self.psize[0] # current size

        self.enabled = True # Dead or not
        self.jumping = False # Jumping or not
        self.ducking = False # Ducking or not

        self.a = Dino.JUMP_ACC # current acceleration
        self.v = 0 # speed

    def check_collisions(self,other,err):
        # checks for collision between this object and 'other' object with the passed error range
        # returns true if collisions found
        xCol = (self.x + self.size[0] >= other.x + err >= self.x) or (self.x + self.size[0] >= other.x + other.size[0] >= self.x) # check for collisions in x coordinates
        yCol = (self.y + self.size[1] >= other.y >= self.y) or (self.y + self.size[1] >= other.y + other.size[1]*0.75>= self.y) # check for collision in y coordinates
        return xCol and yCol

    def die(self):
        # updates the dino to be dead
        # print("dead")
        self.size = self.psize[0]
        self.enabled = False

    def update(self,other,dt,err):
        # update function for moving dino (jumping and falling)
        # the function receives the closest obstacle (other) passed from the game manager object
        # the time between each update and the error range for collision detection
        if not self.enabled:
            return
        if self.check_collisions(other,err):
            self.die()
        else:
            if self.jumping and self.y == 0 != self.prevY:
                self.jumping = False
            self.prevY = self.y # to check if after jump
            self.v += self.a
            self.y = max(self.y + self.v,0) # stopping ground

    def jump(self):
        # updates the dino jump
        if not self.enabled:
            return
        if not self.jumping: # start jumping
            self.jumping = True
            self.v = Dino.JUMP_VEL

    def duck(self):
        # updates the dino to duck
        if not self.enabled:
            return
        if self.size == self.psize[0]:
            self.ducking = True
            self.size = self.psize[1]
            self.a = Dino.DUCK_ACC

    def unduck(self):
        # updates the dinp to stop ducking
        if self.size == self.psize[1]:
            self.ducking = False
            self.size = self.psize[0]
            self.a = Dino.JUMP_ACC

    def reset(self):
        # resets all dino variables for a new game
        self.x = 0
        self.y = 0
        self.prevY = 0

        self.enabled = True  # Dead or not
        self.jumping = False  # Jumping or not
        self.ducking = False  # Ducking or not

        self.v = 0  # speed

class DinoVis:
    # class that manages the visual aspect of the dino player, (frontend)

    def __init__(self,rect,dino:Dino=None):
        # init function for the visual dino class
        # receives the rectangle object (the shape that is drawn to the canvas)
        # and a dino object
        self.rect = rect
        self.dino:Dino = dino

    def draw(self, gm: DinoGameManager, mt, vis:DinoGameVisManager):
        # draws dino or changes dino animation frame
        # the function receives a game manager object, the current animation frame and a visual game manager object
        if self.dino is None or self.rect is None:
            return
        if not self.dino.enabled:
            self.rect.source = gm.imgs['DinoDead'][0]['path']
            self.rect.size = gm.imgs['DinoDead'][0]['size']
        elif self.dino.ducking:  # ducking
            self.rect.source = gm.imgs['DinoDuck'][mt]['path']
            self.rect.size = gm.imgs['DinoDuck'][mt]['size']
        elif self.dino.jumping:  # jumping
            self.rect.source = gm.imgs['DinoJump'][0]['path']
            self.rect.size = gm.imgs['DinoJump'][0]['size']
        else:  # running
            self.rect.source = gm.imgs['DinoRun'][mt]['path']
            self.rect.size = gm.imgs['DinoRun'][mt]['size']
        self.rect.pos = (self.dino.x+vis.xoff, self.dino.y+vis.yoff)


class Obstacle:
    # class that manages the computational aspect of an obstacle, (backend)
    BIRD_OFFSET = 25 # the y offset of the bird while flapping after flapping its wings

    def __init__(self,pos,size,img,change,*,type=0):
        # init function for the obstacle class
        # receives starting position size image path of the obstacle, if the obstacle changes and the type
        self.x = pos[0] # x position of the obstacle
        self.y = pos[1] # y position of the obstacle
        self.size = size # the size of the obstacle (bounds)

        self.type = type # the type of obstacle (bird, cactus)
        self.img = img # the path to the obstacle image
        self.change = change # does the obstacle change shape or size during game

    def update(self,offset,bounds,err):
        # updates the x position of the obstacle and returns true if the obstacle is out of bounds
        self.x = self.x + offset
        return self.checkOutOfBounds(bounds,err)

    def checkOutOfBounds(self,bounds,err):
        # returns true if the obstacle x position is out of bounds
        return self.x + self.size[0] + err < bounds

class ObstacleVis:
    # class that manages the visual aspect of the obstacles, (frontend)
    def __init__(self,rect,obst:Obstacle=None):
        # init function for the visual obstacle class
        # receives the rectangle object (the shape that is drawn to the canvas)
        # and an obstacle object
        self.rect = rect
        self.obst:Obstacle = obst

    def draw(self, gm:DinoGameManager, mt, vis:DinoGameVisManager):
        # draws obstacle or changes obstacle animation frame
        # the function receives a game manager object, the current animation frame and a visual game manager object
        if self.obst is None or self.rect is None:
            return
        if self.obst.change:
            self.rect.pos = (self.obst.x + vis.xoff, self.obst.y+Obstacle.BIRD_OFFSET*mt + vis.yoff)
            self.rect.source = gm.imgs[self.obst.img][mt]['path']
            self.rect.size = gm.imgs[self.obst.img][mt]['size']
        else:
            self.rect.source = gm.imgs[self.obst.img][self.obst.type]['path']
            self.rect.size = gm.imgs[self.obst.img][self.obst.type]['size']
            self.rect.pos = (self.obst.x + vis.xoff, self.obst.y + vis.yoff)


class DinoGen:
    # class that manages the computational aspect of generations in the game (backend)
    def __init__(self,n,maxnotmin=True):  # passed to gen From Controller class
        # an init function for the DinoGen class
        # the function receives the number of players in each generation and a boolean value
        # when the boolean is true the maximum outputs of the neural network are taken
        # if false then the minimum values
        self.n = n
        self.maxnotmin = maxnotmin

        self.nets: Nets = None # a neural networks object
        self.games: List[DinoGameManager] = [] # all the game manager objects in single generation

        self.output = None # the last values the neural network calculated
        self.out = None # the last actions picked by the neural network (index of action)

        self.best = [0] # an array of the indexes of the worst to best players in current generation
        self.bestPlayers = [] # an array of the best players in each generation

        self.gen = 0 # generation number

    def init_games(self,canv_size=(800, 600), path='assets', img=None):
        # initializes the game managers
        self.games = [DinoGameManager(canv_size=canv_size,path=path,img=img) for _ in range(self.n)]

    def init_net(self,inputLength, netShape, aFuncH, aFuncO, mut_func, setWeights=True):
        # initializes the neural networks
        self.nets = Nets(self.n, inputLength, netShape, aFuncH, aFuncO, mut_func)
        if setWeights:
            self.nets.net_weights_init()

    def set_params_single(self,weights,biases):
        # sets the weights and biases of the networks
        if self.not_initialized():
            return
        self.nets.set_variables(weights,biases)

    def not_initialized(self):
        # returns true if both the all the game managers and the networks are already initializes and false else
        return (self.nets is None) or (self.games is None)

    def not_all_dead(self):
        # function returns true if not all the dinos are dead, false if all are dead
        return self.check_dead() > 0

    def check_dead(self):
        # the function counts the number of active (dinos that are alive) games and returns it
        count = 0
        for game in self.games:
            if game.dino_not_dead():
                count += 1
        return count

    def single_pass(self):
        # a single action pass for all the games
        # this function calculates the networks outputs and makes each game take the next action
        if self.not_initialized():
            return

        inputs = np.zeros((self.n,self.nets.input_length,1))

        for idx,game in enumerate(self.games):
            inputs[idx] = game.get_inputs()

        self.out,self.output = self.nets.forward_pass(inputs,self.maxnotmin)

        for idx,game in enumerate(self.games): # automatically ignores dead dinos because dead dinos don't do actions
            game.actions[self.out[idx,0]]()

    def single_frame(self,dt):
        # runs a single game frame, meaning it runs a frame of the game manager and then picks an action for each dino
        # receives the delta time from the last frame\update
        for game in self.games:
            game.updateAll(dt)
        self.single_pass()

    def run_games(self,dt):
        # runs all games in current generation until all dinos are dead
        # receives the delta time from the last frame\update
        if self.not_initialized():
            return

        while self.not_all_dead():
            self.single_frame(dt)

        print("all done")

    def mutate(self):
        # function takes the best player or the two best players in the generation and calls the mutate
        # and crossover function in the players nets
        # and remembers the best net and score of the generation
        self.bestPlayers += [(self.games[self.best[-1]].time,*self.nets.get_ith_net(self.best[-1]))]
        if len(self.best) > 1:
            self.nets.mutate([self.best[-1],self.best[-2]])
        else:
            self.nets.mutate([self.best[-1]])
        self.gen += 1

    def restart(self):
        # the function restarts all the games
        for game in self.games:
            game.reset()

    def return_best(self) -> int:
        # the function returns the best players index in the current generation
        # if it's a new best player it adds it to the end of the best players list in the current generation
        best_gm: int = 0
        for i in range(self.n):
            if self.games[best_gm].time < self.games[i].time:
                best_gm = i
        if self.best[-1] != best_gm:
            self.best += [best_gm]
        return best_gm

    def get_best_players(self):
        # returns the best players in the whole training period (all the generations)
        return self.bestPlayers

class DinoGenVis:
    # class that manages the visual aspect of generations in the game (frontend)
    def __init__(self,n,canvas):
        # an init function for the DinoGenVis class
        # receives a canvas object and the number of players in the generation
        self.n = n

        self.canvas = canvas

        self.vis_games: List[DinoGameVisManager] = [] # an array of visual game managers

    def init_games(self,xoff,yoff,gen_obj:DinoGen):
        # initializes all the visual game managers
        self.vis_games = [DinoGameVisManager(xoff,yoff,self.canvas, gen_obj.games[i]) for i in range(self.n)]

    def single_frame(self):
        # draws a single frame of all the games
        for vis_game in self.vis_games:
            vis_game.draw_all()


