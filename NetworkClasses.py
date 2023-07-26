import numpy as np
import time as t
from kivy.graphics import Rectangle, Color, Ellipse, Line


class Nets:
    # a neural network class (of type fully connected), that lets users access weights and biases
    # this class acts as the brain for the agents (mathematical function that makes the decisions)
    def __init__(self, n, inputLength, netShape, aFuncH, aFuncO, mut_func):
        # the function initializes the neural network class for n neural networks
        # the function receives the number of agents (neural networks), the number of inputs, the shape of the network,
        # the activation function for the hidden layers ,the activation function for the output layer
        # and a function that describes the probability distributions for mutation
        self.net_length = 0

        self.net_shape = netShape
        self.input_length = inputLength
        self.net_len()

        self.n = n  # agents amount
        self.weights = []
        self.bias = []

        self.aFuncH = np.vectorize(aFuncH) # most common ReLu\SoftPlus\Sigmoid
        self.aFuncO = np.vectorize(aFuncO) # most common ReLu\SoftPlus\Sigmoid
        self.mut = np.vectorize(mut_func)

    def net_weights_init(self):
        # the function initializes the weights and biases of the networks
        # format of net shape [(hidden L)..., (output L)] ::=> (neuron length, sequence length)
        # last tuple is output layer
        start = t.time()
        prev = self.input_length
        for lSeq in self.net_shape:  # each layer
            for l in range(lSeq[1]):
                self.weights += [2*np.random.random((self.n, lSeq[0], prev))-1]
                prev = lSeq[0]
                self.bias += [2*np.random.random((self.n, prev, 1))-1]
        print(f"timeRandom:{t.time() - start}")

    def net_len(self):
        # the function returns the number of layers in the network shape and saves it to the net_length variable
        if self.net_shape is not None:
            self.net_length = sum([i[1] for i in self.net_shape])

    def linear(self, x, w, b, aFunc):
        # linear sum (x input, w,b, activation function) in forward pass calculation
        return aFunc(w @ x + b)

    def forward_pass(self, inputs, maxnotmin):
        # the forwards pass of the network
        # the function receives and input vector and a boolean that tells the function it needs to return the maximum
        # of the outputs or the minimum of the outputs

        # initializing inputs
        x = inputs.copy()  # not necessary
        # calculation forward passes
        for i in range(self.net_length):
            if i < self.net_length - 1:
                x = self.linear(x, self.weights[i], self.bias[i], self.aFuncH)  # probably redundant
            else:
                x = self.linear(x, self.weights[i], self.bias[i], self.aFuncO)  # probably redundant
        # mapping outputs
        if maxnotmin: # case max function
            return x.argmax(axis=1),x
        else: # case min function
            return x.argmin(axis=1),x

    def mutate(self,parent_idxs):
        # the function mutates the current weights and biases based on the specified parents and mutation function
        # the function receives the indexes of the weights of the parents
        # the function takes the networks of the parents randomly overlaps them n times (crossover) to create n new networks
        # then to mutate the network a small random offset is added to all the weights and biases of every network
        zz = np.vectorize(Nets.zigzag)
        n_parents = len(parent_idxs)
        new_w = []
        new_b = []

        for i in range(self.net_length):
            shape_w = self.weights[i].shape
            shape_b = self.bias[i].shape
            # crossover calculation for bias and weights with equal distribution of all parents
            crossover_w = np.array([[self.weights[i][np.random.choice(parent_idxs,self.n),y,x] for x in range(shape_w[2])] for y in range(shape_w[1])]).swapaxes(0,1).T
            crossover_b = np.array([[self.bias[i][np.random.choice(parent_idxs,self.n),y,x] for x in range(shape_b[2])] for y in range(shape_b[1])]).swapaxes(0,1).T
            # mutation calculation for bias and weights
            new_w += [zz(crossover_w + self.mut(np.random.uniform(-1.0,1.0,shape_w)))]
            new_b += [zz(crossover_b + self.mut(np.random.uniform(-1.0,1.0,shape_b)))]

        self.weights = new_w
        self.bias = new_b

    def get_ith_net(self,i):
        # function returns the ith network
        ws = [w[i] for w in self.weights]
        bs = [b[i] for b in self.bias]
        return ws,bs

    def set_variables(self,ws,bs):
        # the function receives new weights and biases and updates the current ones
        self.weights = ws
        self.bias = bs

    def __len__(self):
        # returns the amount of layers in the network
        return self.net_length

    @staticmethod
    def zigzag(x):
        # mathematical function that limits the range of x in a specific way
        # when x is positive between 0 and 1 and when x is negative between -1 and 0
        return np.sign(x) * (x % 2 - 2 * (x % 1)) * (2 * (x % 2 - (x % 1)) - 1)


class NetVis:
    # a purely visualizing class
    # the class displays a specified network and shows how it is updated while the game is running

    def __init__(self,canvas,pos,size,gm):
        # an init function for the visualizer network class
        # the function receives the size of the display area [width, height], the position of the display in the canvas
        # the function receives the game manager and a canvas object
        self.nodes = []
        self.lines = []

        self.nodeSize = 0

        self.gm = gm

        self.pos = pos
        self.size = size

        self.canvas = canvas

    def init_nodes(self, size=1):
        # the function initializes the nodes objects
        # the function receives the relative size of the nodes
        mx = self.gm.nets.input_length
        for tmp in self.gm.nets.net_shape: # finding the largest layer
            mx = max(tmp[0], mx)

        self.nodeSize = min(self.size[0] / (self.gm.nets.net_length+2),self.size[1] / (mx+2))*size
        offY = (self.size[1] - self.nodeSize) / (mx)
        print(self.gm.nets.net_length)
        offX = (self.size[0] - self.nodeSize) / (self.gm.nets.net_length)
        tmpY = (self.size[1] - (self.gm.nets.input_length-1)*offY - self.nodeSize)/2
        tmpX = 0

        layer = []
        for _ in range(self.gm.nets.input_length):
            # with self.canvas:
            layer += [Ellipse(pos=(tmpX+self.pos[0],tmpY+self.pos[1]),size=(self.nodeSize,self.nodeSize))]
            tmpY += offY
        self.nodes += [layer]

        for lSeq in self.gm.nets.net_shape:  # each layer
            for l in range(lSeq[1]):
                layer = []
                tmpY = (self.size[1] - (lSeq[0]-1)*offY - self.nodeSize)/2
                tmpX += offX
                for _ in range(lSeq[0]):
                    # with self.canvas:
                    layer += [Ellipse(pos=(tmpX+self.pos[0], tmpY+self.pos[1]), size=(self.nodeSize, self.nodeSize))]
                    tmpY += offY
                self.nodes += [layer]

    def init_lines(self,width):
        # the function initializes the line objects
        # the function receives the width of the lines
        for i in range(1,len(self.nodes)):
            layer = []
            for node1 in self.nodes[i-1]:
                for node2 in self.nodes[i]:
                    layer += [Line(points=[node1.pos[0]+self.nodeSize/2,node1.pos[1]+self.nodeSize/2,node2.pos[0]+self.nodeSize/2,node2.pos[1]+self.nodeSize/2],width=width)]
            self.lines += [layer]

    def draw_canvas(self):
        # the function draws the line and node objects to the canvas
        for layer in self.lines:
            for line in layer:
                self.canvas.add(line)
        for layer in self.nodes:
            for node in layer:
                self.canvas.add(node)

    def update(self):
        # the function updates the network based on the new inputs and outputs
        best = self.gm.best[-1]
        inputs = self.normalize(self.gm.games[best].inputs)
        inputs[-1] = np.sign(inputs[-1])
        inputs[-2] = np.sign(inputs[-2])

        for node,inpu in zip(self.nodes[0],inputs[::-1]):
            pass
            node.angle_end = self.interpolate(np.abs(inpu), 360, 0)
            # with self.canvas:
            #     self.canvas.remove(node)
            #     Color(*self.interpolate(np.abs(inpu),[1,1,1,0.9],[0,0,0,0.9]))
            #     self.canvas.add(node)


        for layer in range(self.gm.nets.net_length):
            for w,line in zip(self.gm.nets.weights[layer][best].T.flatten()[:,np.newaxis],self.lines[layer]):
                pass
                line.width = self.interpolate(np.abs(w),2.5,0.5)
                # with self.canvas:
                #     self.canvas.remove(line)
                #     Color(*self.interpolate(w,[1, .5, .2, 0.8], [0.2, .5, 1, 0.8]))
                #     self.canvas.add(line)


        outputs = self.normalize(self.gm.output[best])

        for node,out in zip(self.nodes[-1],outputs[::-1]):
            pass
            node.angle_end = self.interpolate(np.abs(out),360,0)
            # with self.canvas:
            #     self.canvas.remove(node)
            #     Color(*self.interpolate(np.abs(out),[1,1,1,0.9],[0,0,0,0.9]))
            #     self.canvas.add(node)

    @staticmethod
    def normalize(x,axis=0):
        # the function normalizes the inputted vector based on the inputetd axis
        return x/np.linalg.norm(x,axis=axis)

    @staticmethod
    def interpolate(x,val1,val2):
        # the function interpolates between (val1) and (val2) based on (x)
        return val1*x+val2*(1-x)
