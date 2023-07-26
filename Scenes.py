import numpy as np
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color
from kivy.uix.image import Image
from PIL import Image
from kivy.core.window import Window
from kivy.clock import Clock
from DinoGame import *
from NetworkClasses import *
import random as rd
from GraphingUtil import Graph as hp


# all the scene number codes:

class SceneCodes:
    # scene code constants class
    INSTRUCTIONS = -1
    MENU_SCENE = 0

    PLAY_SCENE = 1
    PLAY_DEATH_SCENE = 2

    LEARN_SELECT_SCENE = 3
    LEARN_SCENE = 4
    LEARN_DEATH_SCENE = 5

    VIEW_BEST_SCENE = 6

    PLAY_BEST_SCENE = 7
    PLAY_BEST_DEATH_SCENE = 8


class SceneManager:
    # a class that manages the game, the learning and the scenes (visual parts of the game)
    BEST_FILENAME = "best_players.npy"  # the file which contains the best players data

    def __init__(self, gw, n, net_shape, actv_f, out_f, mut_f):
        # an init function for the scene manager which initializes all the scene objects
        # the function receives the game widget object (in the main file), the amount of agents
        # the shape of the network, the activation function for the hidden layers
        # and the activation function for the output layer
        # and the mutation function
        self.gw = gw
        self.current_scene = SceneCodes.MENU_SCENE
        self.best_players = []

        self.instructions = InstructScene(self, gw, (800, 600))
        self.menu_scene = MenuScene(self, gw, (800, 600), net_shape, actv_f, out_f, mut_f)

        self.play_scene = PlayScene(self, gw, (800, 300))
        self.play_death_scene = DeathScene(self, gw)

        self.learn_select_scene = LearnSelectScene(self, gw, (800, 600))
        self.learn_scene = LearnScene(self, gw, (800, 600), n, net_shape, actv_f, out_f, mut_f)
        self.learn_death_scene = LearnDeathScene(self, gw)

        self.view_best_scene = ViewBestScene(self, gw, (800, 600), net_shape, actv_f, out_f, mut_f)

        self.play_best_scene = PlayBestScene(self, gw, (800, 600), net_shape, actv_f, out_f, mut_f)
        self.play_best_death_scene = PlayBestDeathScene(self, gw, (800, 600))

        self.set_scene(SceneCodes.MENU_SCENE)

    def reset_scene(self):
        # clears canvas and resets current scene
        self.gw.clear_widgets()
        self.gw.canvas.clear()

    def set_scene(self, scene):
        # this function resets the scene, sets the current scene and initializes it
        # the function receives the new current scene
        self.reset_scene()
        self.current_scene = scene

        if self.current_scene == SceneCodes.MENU_SCENE:
            self.menu_scene.initialize_scene()
        elif self.current_scene == SceneCodes.INSTRUCTIONS:
            self.instructions.initialize_scene()
        elif self.current_scene == SceneCodes.PLAY_SCENE:
            self.play_scene.initialize_scene()
        elif self.current_scene == SceneCodes.PLAY_DEATH_SCENE:
            self.play_death_scene.set_score(self.play_scene.get_score())
            self.play_death_scene.initialize_scene()
        elif self.current_scene == SceneCodes.LEARN_SELECT_SCENE:
            self.learn_select_scene.initialize_scene()
        elif self.current_scene == SceneCodes.LEARN_SCENE:
            self.learn_scene.set_gen_amount(self.learn_select_scene.get_gen_amount())
            self.learn_scene.initialize_scene()
        elif self.current_scene == SceneCodes.LEARN_DEATH_SCENE:
            self.best_players = self.learn_scene.get_best_players()
            self.learn_death_scene.set_score(self.best_players[self.find_best_player()][0])
            self.learn_death_scene.initialize_scene()
        elif self.current_scene == SceneCodes.VIEW_BEST_SCENE:
            self.view_best_scene.set_best_players(self.best_players)
            self.view_best_scene.initialize_scene()
        elif self.current_scene == SceneCodes.PLAY_BEST_SCENE:
            self.play_best_scene.set_best(self.best_players[self.find_best_player()])
            self.play_best_scene.initialize_scene()
        elif self.current_scene == SceneCodes.PLAY_BEST_DEATH_SCENE:
            self.play_best_death_scene.set_winner_score(*self.play_best_scene.get_winner())
            self.play_best_death_scene.initialize_scene()

    def find_best_player(self):
        # the function returns the best player index out of the best players of all generations
        best = 0
        for i in range(len(self.best_players)):
            if self.best_players[i][0] > self.best_players[best][0]:
                best = i
        return best

    def save_best_players(self):
        # this function takes the current best players networks and saves them to a file
        npplayers = np.array(self.best_players)
        np.save(SceneManager.BEST_FILENAME, npplayers)
        print("saved successfully!")

    def update_scene(self, dt):
        # this function updates the current scene (if it needs to be updated, such as the game scene)
        # the function receives the delta time between each update
        if self.current_scene == SceneCodes.MENU_SCENE:
            self.menu_scene.update_scene(dt)
        elif self.current_scene == SceneCodes.PLAY_SCENE:
            self.play_scene.update_scene(dt)
        elif self.current_scene == SceneCodes.LEARN_SCENE:
            self.learn_scene.update_scene(dt)
        elif self.current_scene == SceneCodes.VIEW_BEST_SCENE:
            self.view_best_scene.update_scene(dt)
        elif self.current_scene == SceneCodes.PLAY_BEST_SCENE:
            self.play_best_scene.update_scene(dt)

    def key_up(self, keycode):
        # this function calls the key up function in current scene (if the scene needs user input)
        # the function receives a keycode from the outside call of the function to know which key was pressed
        if self.current_scene == SceneCodes.PLAY_SCENE:
            self.play_scene.key_up(keycode)
        elif self.current_scene == SceneCodes.PLAY_BEST_SCENE:
            self.play_best_scene.key_up(keycode)

    def key_down(self, keycode):
        # this function calls the key down function in current scene (if the scene needs user input)
        # the function receives a keycode from the outside call of the function to know which key was released
        if self.current_scene == SceneCodes.PLAY_SCENE:
            self.play_scene.key_down(keycode)
        elif self.current_scene == SceneCodes.PLAY_BEST_SCENE:
            self.play_best_scene.key_down(keycode)

    def set_best_players(self, best):
        # sets the variable best_players to the inputted value (new best players)
        # the function receives the new best players
        self.best_players = best


class Scene:
    # a general template for a scene object class
    def __init__(self, scene_manager, gw, window_size):
        # an init function for the Scene class
        # for each scene class that extends this class it receives a scene manager object (the current scene manager),
        # the game widget object (in the main file), and the window size of the current scene
        self.scene_manager = scene_manager
        self.gw = gw  # game widget
        self.window_size = window_size

    def update_window_size(self):
        # this function updates the current window size
        if (Window.size is not self.window_size) and (self.window_size is not None):
            Window.size = self.window_size

    def initialize_scene(self):
        # general function for initializing scene (implemented in extended classes)
        # this function is a template
        pass

    def update_scene(self, dt):
        # general function for updating scene (implemented in extended classes)
        # this function is a template
        pass

    def next_scene(self, value):
        # calls the end scene function (the value received isn't used, in order to call the function on a button press
        # [binding the function to a button], it has to receive a value)
        self.end_scene()

    def end_scene(self):
        # general function for ending scene (implemented in extended classes)
        # this function is a template
        pass


# menu + instructions
class InstructScene(Scene):
    # the instructions scene class that extends the scene class
    # basic instructions for all other buttons in the game are written in this scene
    def __init__(self, scene_manager, gw, window_size):
        # initializes the object
        super().__init__(scene_manager, gw, window_size)

    def initialize_scene(self):
        # extended function that initializes the scene
        self.update_window_size()

        with self.gw.canvas:
            Color(0.9, 0.85, 0.9, 1)
            Rectangle(size=Window.size)
            Color(0.6, 0.6, 0.6, 1)
            Rectangle(size=(620, 300), pos=(95, 105))

        label = Label(font_size='55sp', halign='center', pos=(350, 450))
        label.text = 'Instructions'
        label.color = [0.3, 0.3, 0.3, 1]
        label.bold = True
        self.gw.add_widget(label)

        text = Label(font_size='20sp', halign='left', pos=(350, 235))
        text.text = "   Play - lets you play the basic game" \
                    "\n\n   Learn - lets you train N generations of dinos" \
                    "\n\n   View Best - lets you view all the best dinos from the N\n                       " \
                    "generations, meaning N different dinos" \
                    "\n\n   Play Best - lets you play against the best player currently found"
        self.gw.add_widget(text)

        text2 = Label(font_size='16sp', halign='center', pos=(352, 115))
        text2.text = "\n\nThe menu automatically tries to load the best players file (\"best_players.npy\") \nif the file isn't found you need to train the dinos (press the learn button)"
        self.gw.add_widget(text2)

        instruct = Button(text='Menu', font_size='20sp', halign='center', pos=(10, 10))
        instruct.width += 45
        instruct.height -= 45
        instruct.bind(on_press=self.to_menu)
        self.gw.add_widget(instruct)

    def to_menu(self, value):
        # calls the scene manager and changes current scene to menu scene
        # this function is bound to a button
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.MENU_SCENE)


class MenuScene(Scene):
    # menu scene class that extends the scene class
    # this scene lets the user pick what to do in the game
    # this scene also tries to load the best players file in the background if the file exists
    def __init__(self, scene_manager, gw, window_size, net_shape, actv_f, out_f, mut_f):
        # init function for the class
        super().__init__(scene_manager, gw, window_size)
        self.best_players_exist = False # does the save file exist

        self.compVis: DinoGenVis = None
        self.comp: DinoGen = None

        self.net_shape = net_shape

        self.activation_function = actv_f
        self.output_function = out_f
        self.mutation_function = mut_f

    def initialize_scene(self):
        # extended function that initializes the scene
        self.update_window_size()
        if not self.best_players_exist:
            self.load_best()

        with self.gw.canvas:
            Color(0.9, 0.85, 0.9, 1)
            Rectangle(size=Window.size)

        label = Label(font_size='80sp', halign='center', pos=(350, 400))
        label.text = 'Dino Game'
        label.color = [0.3, 0.3, 0.3, 1]
        label.bold = True
        self.gw.add_widget(label)

        play = Button(text='Play', font_size='40sp', halign='center', pos=(210, 255))
        play.width += 80
        play.height -= 20
        play.bind(on_press=self.to_play)
        self.gw.add_widget(play)

        learn = Button(text='Learn', font_size='40sp', halign='center', pos=(410, 255))
        learn.width += 80
        learn.height -= 20
        learn.bind(on_press=self.to_learn)
        self.gw.add_widget(learn)

        best = Button(text='View Best', font_size='33sp', halign='center', pos=(210, 155))
        best.width += 80
        best.height -= 20
        best.bind(on_press=self.to_best)
        self.gw.add_widget(best)

        playVbest = Button(text='Play Best', font_size='33sp', halign='center', pos=(410, 155))
        playVbest.width += 80
        playVbest.height -= 20
        playVbest.bind(on_press=self.to_vbest)
        self.gw.add_widget(playVbest)

        instruct = Button(text='Instructions', font_size='20sp', halign='center', pos=(10, 5))
        instruct.width += 45
        instruct.height -= 45
        instruct.bind(on_press=self.to_instructions)

        # background game
        self.comp = DinoGen(1)
        self.comp.init_games(canv_size=Window.size)
        self.comp.init_net(DinoGameManager.IN_LEN, self.net_shape, self.activation_function, self.output_function,
                           self.mutation_function)
        with self.gw.canvas:
            Color(0.9, 0.9, 1, 1)
            Rectangle(pos=(0, 45), size=self.comp.games[0].imgs['Track'][0]['size'],
                      source=self.comp.games[0].imgs['Track'][0]['path'])
        self.compVis = DinoGenVis(1, self.gw.canvas)
        self.compVis.init_games(30, 60, self.comp)

        self.gw.add_widget(instruct)

    def update_scene(self, dt):
        # extended function that updates the visual and the computational part of the scene
        if self.comp.not_all_dead():
            self.comp.single_frame(dt)
            self.compVis.single_frame()
        else:
            self.comp.mutate()
            self.comp.restart()

    def to_play(self, value):
        # calls the scene manager and changes current scene to play scene
        # this function is bound to a button
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.PLAY_SCENE)

    def to_learn(self, value):
        # calls the scene manager and changes current scene to learing scene
        # this function is bound to a button
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.LEARN_SELECT_SCENE)

    def to_best(self, value):
        # calls the scene manager and changes current scene to showing best players scene
        # this function is bound to a button
        if not self.best_players_exist:
            print("no players to test!")
            return
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.VIEW_BEST_SCENE)

    def to_vbest(self, value):
        # calls the scene manager and changes current scene to scene where user plays vs best player saved
        # this function is bound to a button
        if not self.best_players_exist:
            print("no players to test!")
            return
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.PLAY_BEST_SCENE)

    def to_instructions(self, value):
        # calls the scene manager and changes current scene to instructions scene
        # this function is bound to a button
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.INSTRUCTIONS)

    def load_best(self):
        # tries to load a best players file
        try:
            best = np.load(SceneManager.BEST_FILENAME, allow_pickle=True)
        except FileNotFoundError:
            print("file not found")
        else:
            if best is not None:
                self.scene_manager.set_best_players(best)
                self.best_players_exist = True


# regular play scenes
class PlayScene(Scene):
    # play scene class that extends the scene class
    # this scene is the basic game (user only, no computer)
    def __init__(self, scene_manager, gw, window_size):
        # init function for class
        super().__init__(scene_manager, gw, window_size)

        self.score = None
        self.gmv: DinoGameVisManager = None
        self.gm: DinoGameManager = None

    def initialize_scene(self):
        # extended function that initializes the scene
        self.update_window_size()

        with self.gw.canvas:
            Color(0.9, 0.85, 0.9, 1)
            Rectangle(size=Window.size)

        with self.gw.canvas:
            Color(0.9, 0.9, 1, 0.95)

        self.gm = DinoGameManager()
        self.gmv = DinoGameVisManager(30, 0, self.gw.canvas, self.gm)

        self.score = Label(text='Score: 0.0', font_size='20sp', halign='right', pos=(670, 220))
        self.score.color = [0.3, 0.3, 0.3, 1]
        self.score.bold = True
        self.gw.add_widget(self.score)

    def update_scene(self, dt):
        # extended function that updates the visual and the computational part of the scene
        if self.gm.dino_not_dead():
            self.score.text = 'Score: ' + str(int(self.gm.time))
            self.gm.updateAll(dt)
            self.gmv.draw_all()
        else:
            self.end_scene()

    def key_up(self, keycode):
        # calls key up function in the game manager, passes user input to game
        self.gmv.key_up(keycode)

    def key_down(self, keycode):
        # calls key down function in the game manager, passes user input to game
        self.gmv.key_down(keycode)

    def get_score(self):
        # returns the score value from the game
        return int(self.gm.time)

    def end_scene(self):
        # extended function that ends the scene
        # calls the scene manager and changes current scene to the death scene of the basic game
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.PLAY_DEATH_SCENE)


class DeathScene(Scene):
    # death scene class that extends the scene class
    # this scene is the death scene of the basic game (user only, no computer)
    def __init__(self, scene_manager, gw):
        # init function for class
        super().__init__(scene_manager, gw, None)
        self.score = 0

    def set_score(self, score):
        # sets the score variable to the inputted score
        self.score = score

    def initialize_scene(self):
        # extended function that initializes the scene
        self.update_window_size()  # unnecessary

        with self.gw.canvas:
            Color(0.9, 0.85, 0.9, 1)
            Rectangle(size=Window.size)

        score = Label(font_size='40sp', halign='center', pos=(350, 200))
        score.text = 'SCORE: ' + str(int(self.score))
        score.color = [0.3, 0.3, 0.3, 1]
        score.bold = True
        self.gw.add_widget(score)

        menuButton = Button(text='Menu', font_size='40sp', halign='center', pos=(310, 80))
        menuButton.width += 80
        menuButton.height -= 20
        menuButton.bind(on_press=self.next_scene)
        self.gw.add_widget(menuButton)

    def end_scene(self):
        # extended function that ends the scene
        # calls the scene manager and changes current scene to the menu scene
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.MENU_SCENE)


# learning scenes
class LearnSelectScene(Scene):
    # learning select scene class that extends the scene class
    # this scene is the select screen for learning scene
    # (lets the user pick the number of generations for the genetic algorythm)
    def __init__(self, scene_manager, gw, window_size):
        # init function for the class
        super().__init__(scene_manager, gw, window_size)
        self.labelText: Label = None
        self.genText: TextInput = None

    def initialize_scene(self):
        # extended function that initializes the scene
        self.update_window_size()

        with self.gw.canvas:
            Color(0.9, 0.85, 0.9, 1)
            Rectangle(size=Window.size)

        with self.gw.canvas:
            Color(0.8, 0.8, 0.8, 0.7)
            Rectangle(size=(600, 400), pos=(100, 100))

        self.labelText = Label(text='How Many Generations:', font_size='40sp', halign='center', valign='center',
                               pos=(350, 350))
        self.labelText.color = [0.3, 0.3, 0.3, 1]
        self.labelText.bold = True
        self.gw.add_widget(self.labelText)

        self.genText = TextInput(text='10', font_size='50sp', width=130, height=60, halign='center', pos=(335, 270))
        self.genText.background_color = [1, 1, 1, 0.4]
        self.genText.multiline = False
        self.genText.padding = [4, 5, 5, 5]
        self.gw.add_widget(self.genText)

        menuButton = Button(text='Start', font_size='40sp', halign='center', pos=(310, 150))
        menuButton.width += 80
        menuButton.height -= 20
        menuButton.bind(on_press=self.next_scene)
        self.gw.add_widget(menuButton)

    def get_gen_amount(self):
        # returns the value (number if generations) entered into the text area in the game
        # the function also returns 1 if no value is entered
        if self.genText is None:
            return 1
        if self.genText.text == '':
            return 1
        if self.genText.text.isdigit():
            return int(self.genText.text)
        return 1

    def end_scene(self):
        # extended function that ends the scene
        # calls the scene manager and changes the current scene to the learning scene
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.LEARN_SCENE)


class LearnScene(Scene):
    # learn scene class that extends the scene class
    # this scene is scene where the computer learns how to play the game while running for the amount of generations
    # entered before
    def __init__(self, scene_manager, gw, window_size, n, net_shape, actv_f, out_f, mut_f):
        # init function for the class
        super().__init__(scene_manager, gw, window_size)
        self.scores = []
        self.numGen = 1

        self.timeText: TextInput = None
        self.timeLabel: Label = None
        self.bestPlayer: Label = None
        self.alive: Label = None
        self.gen: Label = None
        self.score: Label = None

        self.prev_best_gm = None
        self.gm_vis: DinoGenVis = None
        self.net: NetVis = None
        self.gm: DinoGen = None

        self.net_shape = net_shape
        self.n = n

        self.activation_function = actv_f
        self.output_function = out_f
        self.mutation_function = mut_f

    def set_gen_amount(self, gen):
        # sets the number of generations to the inputted amount
        self.numGen = gen

    def initialize_scene(self):
        # extended function that initializes the scene
        self.update_window_size()

        self.scores = []
        with self.gw.canvas:
            Color(0.9, 0.85, 0.9, 1)
            Rectangle(size=Window.size)

        with self.gw.canvas:
            Color(1, 0.95, 0.6, 1)
            Rectangle(size=(Window.size[0], Window.size[1] / 2), pos=(0, Window.size[1] / 2))
            Color(0.3, 0.6, 0.7, 0.7)

        self.gm = DinoGen(self.n)
        self.gm.init_games(canv_size=Window.size)
        self.gm.init_net(DinoGameManager.IN_LEN, self.net_shape, self.activation_function, self.output_function,
                         self.mutation_function)

        self.net = NetVis(self.gw.canvas, (650, 425), (120, 150), self.gm)
        self.net.init_nodes(1)
        self.net.init_lines(2)
        self.net.draw_canvas()

        with self.gw.canvas:
            Color(0.9, 0.9, 1, 0.6)

        self.gm_vis = DinoGenVis(self.n, self.gw.canvas)
        self.gm_vis.init_games(30, 0, self.gm)

        self.prev_best_gm = 0

        self.score = Label(text='Score: 0.0', font_size='20sp', halign='right', pos=(670, 220))
        self.score.color = [0.3, 0.3, 0.3, 1]
        self.score.bold = True
        self.gw.add_widget(self.score)

        self.gen = Label(text='Gen: ', font_size='20sp', halign='right', pos=(300, 525))
        self.gen.color = [0.3, 0.3, 0.3, 1]
        self.gen.bold = True
        self.gw.add_widget(self.gen)

        self.alive = Label(text='Alive: ', font_size='20sp', halign='left', pos=(400, 525))
        self.alive.color = [0.3, 0.3, 0.3, 1]
        self.alive.bold = True
        self.gw.add_widget(self.alive)

        self.bestPlayer = Label(text='0s Brain', font_size='20sp', halign='right', pos=(660, 525))
        self.bestPlayer.color = [0.3, 0.3, 0.3, 1]
        self.bestPlayer.bold = True
        self.gw.add_widget(self.bestPlayer)

        self.timeLabel = Label(text='Time:', font_size='20sp', halign='left', pos=(0, 525))
        self.timeLabel.color = [0.3, 0.3, 0.3, 1]
        self.timeLabel.bold = True
        self.gw.add_widget(self.timeLabel)

        self.timeText = TextInput(text='1', font_size='20sp', width=70, height=32, halign='left', pos=(80, 557))
        self.timeText.background_color = [1, 1, 1, 0.1]
        self.timeText.multiline = False
        self.timeText.padding = [4, 5, 5, 5]
        self.gw.add_widget(self.timeText)

    def update_scene(self, dt):
        # extended function that updates the visual and the computational part of the scene
        self.score.text = 'Score: ' + str(int(self.gm.games[self.prev_best_gm].time))
        self.alive.text = 'Alive: ' + str(self.gm.check_dead())
        self.gen.text = 'Gen: ' + str(self.gm.gen) + '/' + str(self.numGen)
        self.bestPlayer.text = str(self.prev_best_gm) + 's Brain'

        if self.gm.gen >= self.numGen:
            self.end_scene()
            return
        else:
            val = 1
            if self.timeText.text.isdigit():
                val = int(self.timeText.text)

            for _ in range(val):
                if self.gm.not_all_dead():
                    self._display_best()
                    self.gm.single_frame(dt)
                    self.gm_vis.single_frame()
                    self.net.update()
                else:
                    self.scores += [int(self.gm.games[self.prev_best_gm].time)]
                    hp.plot(self.scores)
                    self.gm.mutate()
                    self.gm.restart()

    def _display_best(self):
        # function that gets the current best game in the generation
        # and displays the best game at the top of the game window
        if self.prev_best_gm != (best_gm := self.gm.return_best()):
            self.gm_vis.vis_games[self.prev_best_gm].yoff = 0
            self.prev_best_gm = best_gm
        else:
            self.gm_vis.vis_games[best_gm].yoff = Window.size[1] / 2

    def get_best_players(self):
        # returns the best players from all the generations as an array of networks
        return self.gm.get_best_players()

    def end_scene(self):
        # extended function that ends the scene
        # calls the scene manager and changes current scene to the learn death scene
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.LEARN_DEATH_SCENE)


class LearnDeathScene(Scene):
    # learn death scene that extends the scene class
    # this scene is the displayed after all the generations of dinos are dead
    # the scene lets the user decide if they want to save the new best players to file or not
    def __init__(self, scene_manager, gw):
        # init function for class
        super().__init__(scene_manager, gw, None)
        self.score = 0

    def set_score(self, score):
        # sets the score to the inputted score
        self.score = score

    def initialize_scene(self):
        # extended function that initializes the scene
        self.update_window_size()

        with self.gw.canvas:
            Color(0.9, 0.85, 0.9, 1)
            Rectangle(size=Window.size)

        score = Label(font_size='40sp', halign='center', pos=(350, 360))
        score.text = 'Best Score: \n' + str(int(self.score))
        score.color = [0.3, 0.3, 0.3, 1]
        score.bold = True
        self.gw.add_widget(score)

        saveButton = Button(text='Save', font_size='40sp', halign='center', pos=(410, 180))
        saveButton.width += 80
        saveButton.height -= 20
        saveButton.bind(on_press=self.save_data)
        self.gw.add_widget(saveButton)

        menuButton = Button(text='Menu', font_size='40sp', halign='center', pos=(210, 180))
        menuButton.width += 80
        menuButton.height -= 20
        menuButton.bind(on_press=self.next_scene)
        self.gw.add_widget(menuButton)

    def save_data(self, value):
        # the function class the scene manager to save the best players calculated in the previous scene
        self.scene_manager.save_best_players()

    def end_scene(self):
        # extended function that ends the scene
        # calls the scene manager and changes current scene to the menu scene
        hp.remove_out()
        self.gw.bind_keys()
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.MENU_SCENE)


# view best scene
class ViewBestScene(Scene):
    # view best scene that extends the scene class
    # this scene lets the user view the best players performance from the saved file in directory
    # (simulates a new game with the network of the best players, one after the other)
    def __init__(self, scene_manager, gw, window_size, net_shape, actv_f, out_f, mut_f):
        # init function for class
        super().__init__(scene_manager, gw, window_size)

        self.timeLabel: Label = None
        self.timeText: TextInput = None
        self.gen = None
        self.score = None

        self.best_players = None
        self.gm_vis: DinoGenVis = None
        self.net: NetVis = None
        self.gm: DinoGen = None
        self.current_game = 0

        self.net_shape = net_shape
        self.n = 1

        self.activation_function = actv_f
        self.output_function = out_f
        self.mutation_function = mut_f

    def set_best_players(self, best_players):
        # sets the best players to the passed value
        self.best_players = best_players

    def initialize_scene(self):
        # extended function that initializes the scene
        self.update_window_size()

        self.current_game = 0

        with self.gw.canvas:
            Color(0.9, 0.85, 0.9, 1)
            Rectangle(size=Window.size)

        with self.gw.canvas:
            Color(1, 0.95, 0.6, 1)
            Rectangle(size=(Window.size[0], Window.size[1] / 2), pos=(0, Window.size[1] / 2))
            Color(0.3, 0.6, 0.7, 0.7)

        self.gm = DinoGen(self.n)
        self.gm.init_games(canv_size=Window.size)
        self.gm.init_net(DinoGameManager.IN_LEN, self.net_shape, self.activation_function, self.output_function,
                         self.mutation_function, setWeights=False)
        self.change_game()

        self.net = NetVis(self.gw.canvas, (650, 410), (120, 150), self.gm)
        self.net.init_nodes(1)
        self.net.init_lines(2)
        self.net.draw_canvas()

        with self.gw.canvas:
            Color(0.9, 0.9, 1, 0.6)

        self.gm_vis = DinoGenVis(self.n, self.gw.canvas)
        self.gm_vis.init_games(30, 300, self.gm)

        self.score = Label(text='Score: 0.0', font_size='20sp', halign='right', pos=(655, 525))
        self.score.color = [0.3, 0.3, 0.3, 1]
        self.score.bold = True
        self.gw.add_widget(self.score)

        self.gen = Label(text='Gen: ', font_size='20sp', halign='right', pos=(300, 525))
        self.gen.color = [0.3, 0.3, 0.3, 1]
        self.gen.bold = True
        self.gw.add_widget(self.gen)

        self.timeLabel = Label(text='Time:', font_size='20sp', halign='left', pos=(0, 525))
        self.timeLabel.color = [0.3, 0.3, 0.3, 1]
        self.timeLabel.bold = True
        self.gw.add_widget(self.timeLabel)

        self.timeText = TextInput(text='1', font_size='20sp', width=70, height=32, halign='left', pos=(80, 557))
        self.timeText.background_color = [1, 1, 1, 0.1]
        self.timeText.multiline = False
        self.timeText.padding = [4, 5, 5, 5]
        self.gw.add_widget(self.timeText)

        menuButton = Button(text='Menu', font_size='40sp', halign='center', pos=(220, 80))
        menuButton.width += 80
        menuButton.height -= 20
        menuButton.bind(on_press=self.next_scene)
        self.gw.add_widget(menuButton)

        nextButton = Button(text='>', font_size='40sp', halign='center', pos=(420, 80))
        nextButton.width = 50
        nextButton.height = 80
        nextButton.bind(on_press=self.next_dino)
        self.gw.add_widget(nextButton)

        next10Button = Button(text='>>', font_size='40sp', halign='center', pos=(480, 80))
        next10Button.width = 70
        next10Button.height = 80
        next10Button.bind(on_press=self.next_10_dino)
        self.gw.add_widget(next10Button)

    def update_scene(self, dt):
        # extended function that updates the visual and the computational part of the scene
        if self.best_players is None:  # if best players aren't initialized
            return
        if self.current_game < len(self.best_players):
            self.score.text = 'Score: ' + str(int(self.gm.games[0].time)) + '/' + str(
                int(self.best_players[self.current_game][0]))
        self.gen.text = 'Gen: ' + str(self.current_game) + '/' + str(len(self.best_players))

        val = 1
        if self.timeText.text.isdigit():
            val = int(self.timeText.text)

        for _ in range(val):
            if self.gm.not_all_dead():
                self.gm.single_frame(dt)
                self.gm_vis.single_frame()
                self.net.update()
            else:  # player died go to next player or death scene
                self.current_game += 1
                if not self.change_game():
                    self.end_scene()
                return

    def next_dino(self, value):
        # change the currently viewed generation by 1 generations
        # this function is bound to a button
        self.current_game += 1
        if not self.change_game():
            self.end_scene()

    def next_10_dino(self, value):
        # change the currently viewed generation by 10 generations
        # this function is bound to a button
        self.current_game += 10
        if not self.change_game():
            self.end_scene()

    def change_game(self):
        # change current game to ith game in best games
        if self.best_players is None:  # if best players aren't initialized
            return False

        if 0 <= self.current_game < len(self.best_players):
            self.gm.set_params_single(*self.best_players[self.current_game][1:])
            self.gm.restart()
            return True
        else:  # for emphasis
            return False

    def end_scene(self):
        # extended function that ends the scene
        # calls the scene manager and changes current scene to the menu scene
        self.gw.bind_keys()
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.MENU_SCENE)


# play best
class PlayBestScene(Scene):
    # play best scene extends the scene class
    # this scene lets the user play agains the computer - the best calculated player available
    # (either calculated in the learning scene or loaded from file)
    def __init__(self, scene_manager, gw, window_size, net_shape, actv_f, out_f, mut_f):
        # init function for class
        super().__init__(scene_manager, gw, window_size)

        self.net: NetVis = None
        self.best = None

        self.score: Label = None

        self.humanVis: DinoGameVisManager = None
        self.human: DinoGameManager = None

        self.compVis: DinoGenVis = None
        self.comp: DinoGen = None

        self.net_shape = net_shape
        self.n = 1

        self.activation_function = actv_f
        self.output_function = out_f
        self.mutation_function = mut_f

    def initialize_scene(self):
        # extended function that initializes the scene
        self.update_window_size()

        with self.gw.canvas:
            Color(0.9, 0.85, 0.9, 1)
            Rectangle(size=Window.size)

        with self.gw.canvas:
            Color(1, 0.95, 0.6, 1)
            Rectangle(size=(Window.size[0], Window.size[1] / 2), pos=(0, Window.size[1] / 2))
            Color(0.3, 0.6, 0.7, 0.7)

        self.comp = DinoGen(self.n)
        self.comp.init_games(canv_size=Window.size)
        self.comp.init_net(DinoGameManager.IN_LEN, self.net_shape, self.activation_function, self.output_function,
                           self.mutation_function, setWeights=False)
        self.set_best_game()

        self.net = NetVis(self.gw.canvas, (650, 425), (120, 150), self.comp)
        self.net.init_nodes(1)
        self.net.init_lines(2)
        self.net.draw_canvas()

        with self.gw.canvas:
            Color(0.9, 0.9, 1, 0.6)

        self.compVis = DinoGenVis(self.n, self.gw.canvas)
        self.compVis.init_games(30, 300, self.comp)

        with self.gw.canvas:
            Color(0.9, 0.9, 1, 0.95)

        self.human = DinoGameManager()
        self.humanVis = DinoGameVisManager(30, 0, self.gw.canvas, self.human)

        self.score = Label(text='Score: 0.0', font_size='23sp', halign='center', pos=(350, 230))
        self.score.color = [0.3, 0.3, 0.3, 1]
        self.score.bold = True
        self.gw.add_widget(self.score)

    def update_scene(self, dt):
        # extended function that updates the visual and the computational part of the scene
        self.score.text = 'Score: ' + str(int(self.human.time))

        if self.comp.not_all_dead():
            self.comp.single_frame(dt)
            self.compVis.single_frame()
            self.net.update()
        else:
            self.end_scene()

        if self.human.dino_not_dead():
            self.human.updateAll(dt)
            self.humanVis.draw_all()
        else:
            self.end_scene()

    def key_up(self, keycode):
        # calls key up function in the game manager, passes user input to game
        self.humanVis.key_up(keycode)

    def key_down(self, keycode):
        # calls key down function in the game manager, passes user input to game
        self.humanVis.key_down(keycode)

    def set_best(self, best):
        # sets the best player out of the current best players for the computer to play with
        self.best = best

    def get_winner(self):
        # returns 0 if human won and 1 if computer won + score
        if self.comp.not_all_dead() and not self.human.dino_not_dead():
            return 1, self.human.time
        return 0, self.human.time

    def set_best_game(self):
        # sets the computer player to the best possible player
        if self.best is None:
            return
        self.comp.set_params_single(*self.best[1:])

    def end_scene(self):
        # extended function that ends the scene
        # calls the scene manager and changes current scene to the play best death scene
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.PLAY_BEST_DEATH_SCENE)


class PlayBestDeathScene(Scene):
    # play best death scene class that extends the scene class
    # after either the computer or the user dies in the play best scene this scene is displayed
    # the scene displays the winner (either the computer or user) and the score of the winner
    def __init__(self, scene_manager, gw, window_size):
        # init function for class
        super().__init__(scene_manager, gw, window_size)

        self.score = 0
        self.winner = 0

    def set_winner_score(self, winner, score):
        # sets the winner (computer, user) and the score based on the passed values
        self.winner = winner
        self.score = score

    def initialize_scene(self):
        # extended function that initializes the scene
        self.update_window_size()

        with self.gw.canvas:
            Color(0.9, 0.85, 0.9, 1)
            Rectangle(size=Window.size)

        winner = Label(font_size='40sp', halign='center', pos=(350, 400))
        winner.text = 'The winner is: \n' + ('computer' if self.winner == 1 else 'human')
        winner.color = [0.3, 0.3, 0.3, 1]
        winner.bold = True
        self.gw.add_widget(winner)

        score = Label(font_size='40sp', halign='center', pos=(350, 280))
        score.text = 'Score: ' + str(int(self.score))
        score.color = [0.3, 0.3, 0.3, 1]
        score.bold = True
        self.gw.add_widget(score)

        menuButton = Button(text='Menu', font_size='40sp', halign='center', pos=(310, 180))
        menuButton.width += 80
        menuButton.height -= 20
        menuButton.bind(on_press=self.next_scene)
        self.gw.add_widget(menuButton)

    def end_scene(self):
        # extended function that ends the scene
        # calls the scene manager and changes current scene to the menu scene
        if self.scene_manager is None:
            return
        self.scene_manager.set_scene(SceneCodes.MENU_SCENE)
