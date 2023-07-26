from kivy.app import App
# ------- maybe unnecessary --------
# from kivy.uix.label import Label
# from kivy.uix.textinput import TextInput
# from kivy.uix.button import Button
# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.widget import Widget
# from kivy.graphics import Rectangle, Color
# from kivy.uix.image import Image
# from PIL import Image
# from kivy.core.window import Window
# from kivy.clock import Clock
# from DinoGame import *
# from NetworkClasses import *
# import random as rd
# import helper as hp
from DinoGame import *
from Scenes import *

class GameWidget(Widget):
    # class that runs contains and manages the main parts of the project
    def __init__(self, **kwargs):
        # init function for the game widget that initializes the scene manager
        super().__init__(**kwargs)
        self._keyboard = None
        self.bind_keys()

        with self.canvas:
            Color(0.9,0.85,0.9,1)
            Rectangle(size=Window.size)

        self.n = 50
        net_shape = [(3,1),(DinoGameManager.OUT_LEN, 1)]
        self.scene_manager = SceneManager(self,self.n,net_shape,GameWidget.Activation,GameWidget.Activation,GameWidget.bell)
        Clock.schedule_interval(self.run_game,0.01)

    def run_game(self,dt):
        # a function called every 0.01 seconds that runs the scene manager update function
        self.scene_manager.update_scene(dt)

    def bind_keys(self):
        # binds the keyboard events to the functions listed below for use while running
        self._keyboard = Window.request_keyboard(self._on_keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_key_down)
        self._keyboard.bind(on_key_up=self._on_key_up)

    def _on_keyboard_closed(self):
        # unbinds the keyboard functions when the game screen is exited i.e.
        self._keyboard.unbind(on_key_down=self._on_key_down)
        self._keyboard.bind(on_key_up=self._on_key_up)
        self._keyboard = None

    def _on_key_down(self,keyboard,keycode,text,modifiers):
        # function that activates on the press of a keyboard key (an event)
        # receives the keycode of the key pressed (a string of the pressed key)
        self.scene_manager.key_down(keycode)

    def _on_key_up(self,keyboard,keycode):
        # function that activates on the release of a keyboard key (an event)
        # receives the keycode of the key pressed (a string of the pressed key)
        self.scene_manager.key_up(keycode)

    @staticmethod
    def Activation(x):
        # activation function for all layers in network - hidden layers and output layer ~
        return max(x, 0.1 * x)
    @staticmethod
    def bell(x):
        # a mathematical function for calculating mutation probabilities ~
        return 0.1 * x ** 3

class MyApp(App):
    # class responsible for managing the app
    # extends the app class from the kivy library
    def build(self):
        # function that builds the canvas and general game widget (game widget that manages the whole game)
        return GameWidget()


if __name__ == '__main__':
    # this runs the whole program
    # run this file to play
    app = MyApp()
    app.run()