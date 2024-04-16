import sys
import numpy as np
import matplotlib as plt
import nnfs

nnfs.init()

X = [[1,2,3,2.5],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]]

class Layer_dense:
  def __init__(self,n_inputs,n_neurons):
    self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
    self.biases = np.zeros((1,n_neurons))
  def forward(self,inputs):
    self.output = np.dot(inputs,self.weights) + self.biases
#print(np.random.randn(4,3))

class Activation_ReLU:
  def forward(self,inputs):
    self.output = np.maximum(0,inputs)

layer1 = Layer_dense(4,5)
layer2 = Layer_dense(5,2)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)


def create_data(points,classes):
  X = np.zeros((points*classes,2))
  y = np.zeros(points*classes, dtype='unit8')
  for class_number in range(classes):
    ix = range(points*class_number,points*(class_number+1))
    r = np.linspace(0.0,1,points) #radius
    t = np.linspace(class_number*4,(class_number+1)*4,points)+np.random.randn(points)*0.2
    