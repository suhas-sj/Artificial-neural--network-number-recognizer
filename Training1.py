import sys
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

np.random.seed(0)

# X = [[1,2,3,2.5],
#      [2.0,5.0,-1.0,2.0],
#      [-1.5,2.7,3.3,-0.8]]

# X,y = spiral_data(100,3)

nnfs.init()

class Layer_dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        inputs = np.array(inputs)
        self.output = np.dot(inputs, self.weights) + self.biases

#print(np.random.randn(4,3))

class Activation_ReLU:
  def forward(self,inputs):
    self.output = np.maximum(0,inputs)

class Activation_SoftMax:
   def forward(self,inputs):
      exp_values= np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
      probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)
      self.output = probabilities

class Loss:
   def calculate(self,output,y):
      sample_losses = self.forward(output,y)
      data_loss = np.mean(sample_losses)
      return data_loss

class Loss_CategoricalCrossEntropy(Loss):
   def forward(self,y_pred,y_true):
      samples = len(y_pred)
      y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

      if len(y_true.shape) == 1:
         correct_confidences = y_pred_clipped[range(samples),y_true]
      elif len(y_true.shape) == 2:
         correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
      negative_log_likelyhoods = -np.log(correct_confidences)
      return negative_log_likelyhoods
   
    
# layer1 = Layer_dense(2,5)
# layer2 = Layer_dense(5,2)
# activation1 = Activation_ReLU()
# layer1.forward(X)
# activation1.forward(layer1.output)
# print(activation1.output)
# layer2.forward(layer1.output)
# print(layer2.output)


# softmax_outputs = np.array([[1,2,3],[2,3,4],[1,5,2]])
# class_targets = [1,0,0]

# for o,c in zip(softmax_outputs,class_targets):
#   print(o[c])

# categorical cross entropy
# print(softmax_outputs[[0,1,2],class_targets])
# print(softmax_outputs[range(len(softmax_outputs)),class_targets])
# print(-np.log(softmax_outputs[range(len(softmax_outputs)),class_targets]))


X,y = spiral_data(samples=100,classes=3)

dense1 = Layer_dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_dense(3,3)
activation2 = Activation_SoftMax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output,y)

print(loss)