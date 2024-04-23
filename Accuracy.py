import numpy as np

softmax_outputs = np.array([[1,2,3],[2,3,4],[1,5,2]])
class_targets = [2,2,1]

predictions = np.argmax(softmax_outputs,axis=1)
accuracy = np.mean(predictions == class_targets)

print("Accuracy is : ",accuracy*100)

'''

To check for accuracy what we do it find the maximum output in the softmax_outputs and then compare
it with the class_target value , if equal then we consider it as true value else false , then we find the mean of 
the obtained values to find out the accuracy of the output

However we are more considerd about the loss here rather than the accuracy of the predictions

'''