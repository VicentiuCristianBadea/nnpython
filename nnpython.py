import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import os, sys, time

print("Initializing hyperparameters for Neural Network")

# Size of rows (_R) and columns (_C) for weight matrices 
THETA1_R, THETA1_C = int(sys.argv[1]), int(sys.argv[2])
THETA2_R, THETA2_C = int(sys.argv[3]), int(sys.argv[4])
THETA3_R, THETA3_C = int(sys.argv[5]), int(sys.argv[6])

BATCH_SIZE = int(sys.argv[7])
LABEL = np.arange(10)
EPOCHS = int(sys.argv[8])
LEARNING_RATE = float(sys.argv[9])

# Loading the data 
TRAIN_DATA = pd.read_csv("mnist_train.csv")
TRAIN_DATA = np.array(TRAIN_DATA, float)
TEST_DATA = pd.read_csv("mnist_test.csv")
TEST_DATA = np.array(TEST_DATA, float)

# Normalizing training and testing data
TRAIN_DATA[:, 1:] = np.divide(TRAIN_DATA[:, 1:].astype(float), 255.0)
TEST_DATA[:, 1:] = np.divide(TEST_DATA[:, 1:].astype(float), 255.0)
learning_lines = []
testing_lines = []

parent_dir = "modelData"

vals = list([THETA1_R, THETA1_C, THETA2_R, THETA2_C, THETA3_R, THETA3_C, BATCH_SIZE, EPOCHS, LEARNING_RATE])
vals = [str(x) for x in vals]
vals_joined = "_".join(vals)
directory = os.path.join(parent_dir,"resultsForParameters_"+vals_joined)
os.mkdir(directory)
#os.chmod(directory, 0o777)

learning_data_file = "trainingResults_" + vals_joined+".csv"
learning_data_file = os.path.join(directory, learning_data_file)

testing_data_file = "testingResults_" + vals_joined+".csv"
testing_data_file = os.path.join(directory, testing_data_file)
print("Data is loaded.")

# Initialize matrix of weight values for hidden layer 1 
THETA1 = np.random.uniform(low=-0.15, high=0.15, size=(THETA1_R, THETA1_C))
print("The shape of the THETA1 weights is:", THETA1.shape)

# Initialize matrix of weight values for hidden layer 2 
THETA2 = np.random.uniform(low=-0.15, high=0.15, size=(THETA2_R, THETA2_C))
print("The shape of the THETA2 weights is:", THETA2.shape)

# Initialize matrix of weight values for output layer
THETA3 = np.random.uniform(low=-0.15, high=0.15, size=(THETA3_R, THETA3_C))
print("the shape of the THETA3 weights is:", THETA3.shape)

#Shuffle the data
np.random.shuffle(TRAIN_DATA)
print("Data is shuffled.")

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def dsigmoid(x):
    return np.multiply(x, 1.0-x)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def forwardProp(i, batch, BATCH_SIZE):
	a_1 = np.array([TRAIN_DATA[i+batch*BATCH_SIZE, 1:]]) #input data into input_layer
	a_1 = np.array([np.insert(a_1, 0, 1)])
	z_2 = np.dot(a_1, THETA1) #Hidden layer 1: multiply by weights and add biases
	a_2 = sigmoid(z_2) #activation function
	a_2 = np.array([np.insert(a_2, 0, 1)])
	z_3 = np.dot(a_2, THETA2) #Hidden layer 2: multiply by weights and add biases
	a_3 = sigmoid(z_3) #activation function
	a_3 = np.array([np.insert(a_3, 0, 1)])
	a_4_out = np.dot(a_3, THETA3)
	a_4 = sigmoid(a_4_out) #10 classes
	output = a_4 #predicted output
	y = np.equal(TRAIN_DATA[i+batch*BATCH_SIZE, 0], LABEL).astype(int)  
	return output, y, a_1, a_2, a_3  

def backProp(output, y, a_1, a_2, a_3, MSE, DELTA3, DELTA2, DELTA1):
	sigma4 = output-y # Error for output layer
	sigma3 = np.multiply(np.dot(sigma4, np.transpose(THETA3)), dsigmoid(a_3))
	sigma2 = np.multiply(np.dot(sigma3[:, 1:], np.transpose(THETA2)), dsigmoid(a_2))
	MSE += sigma4**2 # Mean Squared Error
	DELTA3 += np.dot(np.transpose(a_3), sigma4) # Error for Theta 3
	DELTA2 += np.transpose(np.dot(np.transpose(a_2), sigma3[:, 1:])) # Error for Theta 2
	DELTA1 += np.transpose(np.dot(np.transpose(a_1), sigma2[:, 1:])) # Error for Theta 1
	return DELTA1, DELTA2, DELTA3, MSE

def updateWeights(MSE, DELTA1, DELTA2, DELTA3, THETA1, THETA2, THETA3):
	MSE /= 2 # Mean Squared Error
	MSE = np.array(MSE)
	MSE_error = np.sum(MSE)
	MSE_error /= float(BATCH_SIZE) 
	DELTA3 /= float(BATCH_SIZE) 
	DELTA2 /= float(BATCH_SIZE)
	DELTA1 /= float(BATCH_SIZE)
	THETA3 -= LEARNING_RATE*DELTA3 # Multiplying errors by learning rate
	THETA2 -= LEARNING_RATE*np.transpose(DELTA2)
	THETA1 -= LEARNING_RATE*np.transpose(DELTA1)
	return THETA1, THETA2, THETA3 

def test_data_func(TEST_DATA, THETA1, THETA2, THETA3):
	PREDICTIONS = 0.0
	GOOD_PRED = 0.0
	for i in tqdm(range(int(len(TEST_DATA)))):
		a_1 = np.array(TEST_DATA[i, 1:]) # Use usr_input to choose input data
		a_1 = np.array([np.insert(a_1, 0, 1)])
		z_2 = np.dot(a_1, THETA1) # Hidden layer 1: multiply by weights and add biases
		a_2 = sigmoid(z_2) # Activation function
		a_2 = np.array([np.insert(a_2, 0, 1)])
		z_3 = np.dot(a_2, THETA2) # Hidden layer 2: multiply by weights and add biases
		a_3 = sigmoid(z_3) # Activation function
		a_3 = np.array([np.insert(a_3, 0, 1)])
		a_4_out = np.dot(a_3, THETA3)
		a_4 = sigmoid(a_4_out) # 10 classes
		output = a_4 # Predicted output
		y = np.equal(TEST_DATA[i, 0], LABEL).astype(int)
		PREDICTIONS += 1
		if(np.argmax(output) == TEST_DATA[i, 0]):
			GOOD_PRED += 1
		ACCURACY = '%.3f'%((GOOD_PRED/PREDICTIONS)*100)
		line = [ACCURACY, i, np.argmax(output), TEST_DATA[i,0]]
		testing_lines.append(line)
	with open(testing_data_file, "w") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for l in testing_lines:
			writer.writerow(l)


def train_data_func(TRAIN_DATA, THETA1, THETA2, THETA3):
	print("\nTraining Neural Network...")
	for epoch in range(EPOCHS): # Repeat training for number of epochs
		PREDICTIONS = 0.0
		GOOD_PRED = 0.0
		np.random.shuffle(TRAIN_DATA) # Shuffle data before running new epoch
		for batch in tqdm(range(int((len(TRAIN_DATA)/BATCH_SIZE)))): # Run through all training data
			DELTA3 = 0
			DELTA2 = 0
			DELTA1 = 0
			MSE = 0
			for i in range(BATCH_SIZE):
				output, y, a_1, a_2, a_3 = forwardProp(i, batch, BATCH_SIZE) 
				DELTA1, DELTA2, DELTA3, MSE = backProp(output, y, a_1, a_2, a_3, MSE, DELTA3, DELTA2, DELTA1)
				PREDICTIONS += 1
				if(np.argmax(output) == TRAIN_DATA[i+batch*BATCH_SIZE, 0]):
					GOOD_PRED += 1
				ACCURACY = '%.3f'%((GOOD_PRED/PREDICTIONS)*100)
				if batch%1000 == 0:
					line = [ACCURACY, batch, epoch]
					learning_lines.append(line)
					#print(lines)
			THETA1, THETA2, THETA3 = updateWeights(MSE, DELTA1, DELTA2, DELTA3, THETA1, THETA2, THETA3)           
		print("\n ACCURACY:", ACCURACY+"%", "    ", " CURRENT EPOCH:", (str(epoch+1))+"/"+(str(EPOCHS)), "                 ", "CURRENT BATCH")
	with open(learning_data_file, "w") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for l in learning_lines:
			writer.writerow(l)
		
	return THETA1, THETA2, THETA3

THETA1_TEST, THETA2_TEST, THETA3_TEST = train_data_func(TRAIN_DATA, THETA1, THETA2, THETA3)

print("Network trained, currently testing the model.")
test_data_func(TEST_DATA, THETA1_TEST, THETA2_TEST, THETA3_TEST)
time.sleep(3)
