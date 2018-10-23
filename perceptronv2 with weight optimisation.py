# -*- coding: utf-8 -*-
"""
Perceptron Script	

activation = sum(weight_i * x_i) + bias
 	
prediction = 1.0 if activation > 0.0 else 0.0

gradient descent weight changes
weight_i = weight_i + learning_rate * (expected_output - prediction) * x_i

"""

# Make a prediction with weights
def predict(Row, weights):
    activation = weights[0]
    for i in range(numColumns):
        activation += weights[i + 1] * dataset[Row][i]
    return 1.0 if activation > 0.0 else 0.0
 
    # Estimate Perceptron weights using stochastic gradient descent
def train_weights(dataset, l_rate, n_epoch):
    for epoch in range(n_epoch):
        #setting total error to zero
        total_error=0
        for Row in range(numRows):
            #We are calling the prediction function 
            prediction = predict(Row, weights)
            print("Expected=%d, Predicted=%d" % (output[Row], prediction))
            #Is there an error?
            error = output[Row] - prediction
            #Running total of all errors in this epoch
            total_error += abs(error)
            #time to change the weights 
            weights[0] = weights[0] + l_rate * error
            for i in range(numColumns):
                weights[i + 1] = weights[i + 1] + l_rate * error * dataset[Row][i]
        print('>epoch=%d, lrate=%.3f, total error=%.3f' % (epoch+1, l_rate, total_error))
        print(weights)
        #let's stop changing weights if the total error in the epoch was zero
        if total_error == 0:
                return weights
    return weights

# Input data
#dataset can be set to anything or read from file
dataset = [[0.1, 0.9],
[0.2, 0.8],
[0.3, 0.75],
[0.5, 0.75],
[0.7, 0.65],
[0.8, 0.6],
[0.9, 0.7],
[0.1, 0.05],
[0.2, 0.1],
[0.3, 0.15],
[0.4, 0.2],
[0.5, 0.3],
[0.6, 0.55]]
#output can be set to anything or read from file but should be be equal to the number of rows of teh dataset
output = [1,1,1,1,1,1,1,0,0,0,0,0,0,0]
#weights can be set to anything to start but needs to have one more column then the dataset due to w0
#as the treshold actication either gives a zero or one as output the mean is 0.5 so setting the weights at 0.5 may be better than setting them to zero 
weights = [0.5,0.5,0.5]
#Learning rate
l_rate = 0.1
#Iterations to calculate the weight changes. can be a high number as the script stops when total error is zero which is easy if possible for a linear activation functions like the treshold function 
n_epoch = 100
#this returns the number of rows in the dataset
numRows = len(dataset)
#this returns the number of columns in the dataset
numColumns= len(dataset[0])


#calling the function for updating the weights so that prediction is close to given Output
weights = train_weights(dataset, l_rate, n_epoch)
print(weights)
