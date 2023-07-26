
import matplotlib.pyplot as plt
from IPython import display

class Graph:
    # utilities class that has static functions for displaying and clearing graph

    plt.ion()
    @staticmethod
    def plot(scores):
        # function that displays a graph of the inputted scores
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.ylim(ymin=0)
        # plt.xticks(range(len(scores)))
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.show(block=False)
        plt.pause(.1)

    @staticmethod
    def remove_out():
        # clears the drawn graph
        display.clear_output()
        plt.close()
