# Author: Eddie Carrizales
# Date: March 20, 2024

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

url = "https://drive.google.com/file/d/12k1AVTnReDPuhcrtlPO069xfDwrwWOEj/view?usp=sharing"

# Medical Insurance Cost Prediction using Neural Network built from scratch
# Dataset Source: https://www.kaggle.com/datasets/rahulvyasm/medical-insurance-cost-prediction

class Neural_Network:

    def load_dataset(self):
        url = "https://drive.google.com/file/d/12k1AVTnReDPuhcrtlPO069xfDwrwWOEj/view?usp=sharing"
        file_id = url.split("/")[-2]
        dwn_url = "https://drive.google.com/uc?id=" + file_id
        original_df = pd.read_csv(dwn_url)  # read the file
        return original_df

    def pre_process_dataset(self, original_df):
        # First letr print our dataframe
        print("Below is our original dataset:")
        print(original_df)
        print("")

        # First lets check if there are any empty rows or values in our data
        # (If there are we will have to remove them or fill them in with generated data)
        empty_rows = original_df.isnull().any(axis=1)
        rows_with_missing_values = original_df[empty_rows]

        # And as we can see, there are no rows with missing values
        #print(rows_with_missing_values)  # shows rows

        # This will display the variable types of our data (As we can see we have many different types of data so we have to convert it all to numerical)
        print("This will display the variable types of our data (As we can see we have many different types of data so we have to convert it all to numerical)")
        print(original_df.dtypes)
        print("")

        # Lets use one-hot encoding to convert the categorical columns to numerical
        numerical_df = pd.get_dummies(original_df, columns=["sex", "smoker", "region"])

        # Now all our data is numerical, I will just move the charges column to the end to keep a nice format
        # Also, because our charges will be what we are trying to predict
        print("Now all our data is numerical, I will just move the charges column to the end to keep a nice format")
        colms = numerical_df.columns.to_list()
        colms.remove("charges")
        colms.append("charges")
        numerical_df = numerical_df[colms]
        print(numerical_df)
        print("")

        # This function provides a summary of various statistics to understand the values in our data
        print("This function provides a summary of various statistics to understand the values in our data")
        print(original_df.describe())
        print("")

        # Now I will normalize all of the data so that it is at the same scale, allowing for better convergence later
        scaler = MinMaxScaler()
        numerical_normalized_df = pd.DataFrame(scaler.fit_transform(numerical_df), columns=numerical_df.columns)

        print("Now I will normalize all of the data so that it is at the same scale, allowing for better convergence later")
        print(numerical_normalized_df)
        print("")

        return numerical_normalized_df


    def train_test_split(self, numerical_normalized_df):
        # Now lets split the data into train and test, I will do 80/20
        # X will hold the features and Y will hold the target variable
        X = numerical_normalized_df.drop(columns=["charges"])  # we take all columns except for "charges"
        y = numerical_normalized_df["charges"]  # We only take "charges" column

        # Splitting the training data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 80/20

        print("This is our data that has been split into 80/20")
        print("X_train:")
        print(X_train)
        print("")

        print("y_train:")
        print(y_train)
        print("")

        print("X_test:")
        print(X_test)
        print("")

        print("y_test:")
        print(y_test)
        print("")

        # Now since we will be working with matrices, we must convert to np.array()
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        return X_train, X_test, y_train, y_test

    def create_network_layers(self, inputs):
        network_layers = []

        lower_bound = -1  # lower bound for random weights
        upper_bound = 1  # upper bound for random weights

        for i in range(1, len(inputs)):
            # Number of nodes in the previous layer
            prev_nodes = inputs[i - 1]

            # Number of nodes in the current layer
            current_nodes = inputs[i]

            # create random weights between -0.1 and 0.1
            weights = np.random.uniform(lower_bound, upper_bound, size=(prev_nodes, current_nodes))

            # initialize the biases with 1's
            biases = np.ones((1, current_nodes))

            # add the weights and biases to the network layers list
            network_layers.append([weights, biases])

            print("This is the network of layers we have created with random weights and biases 4 layers (1 input, 2 hidden, 1 output) [11, 5, 5, 1]:")
            print(network_layers)
            print("")

        return network_layers

    def forward_pass(self, batch, network_layers, activation_function):
        hidden_states = [batch.copy()]

        # We will loop through each of our layers from input layer -> to hidden layers -> to output layer
        for i in range(len(network_layers)):
            # weighted_sum of current layer = (multiply currenty layer inputs) * (with current layer weights) + (and add bias)
            batch = np.matmul(batch, network_layers[i][0]) + network_layers[i][1]  # i.e WX + b

            # if we are not at the output layer (last layer), we apply our activation function
            if i < len(network_layers) - 1:
                # applying activation function (I added conditional here allowing user to choose between 3 activation functions)
                if activation_function == "sigmoid":
                    batch = 1 / (1 + np.exp(-batch))  # sigmoid
                elif activation_function == "tanh":
                    batch = np.tanh(batch)  # tanh
                elif activation_function == "relu":
                    batch = np.maximum(batch, 0)  # relu
                else:
                    print("Invalid Activation function. Please choose between sigmoid, tanh, or relu.")
                    break

            # we store the results/hidden states so we can use them to calculate gradients during backpropagation
            hidden_states.append(batch.copy())

        return batch, hidden_states

    def backward_pass(self, network_layers, hidden_states, delta, targets, learning_rate, activation_function):
        targets = targets.reshape(-1, 1)  # Reshape targets to match the shape of hidden_states[i+1]

        # We will loop through each of our layers backwards: from output layer -> to hidden layers -> input layer (note this loop will exclude input layer)
        for i in range(len(network_layers) - 1, -1, -1):
            # The delta equation is different for output layer and hidden layer:

            # CASE I: if we are at the output layer
            if i == len(network_layers) - 1:
                # conditional statements what will calculate based on the activation function chosen (sigmoid, tanh, relu)
                if activation_function == "sigmoid":
                    # sigmoid equation for backpropagation (derrivative)
                    delta = (hidden_states[i + 1] - targets) * hidden_states[i + 1] * (1 - hidden_states[
                        i + 1])  # S = (t - O) * O * (1 - O) where O represents the ouput with the sigmoid activation function
                elif activation_function == "tanh":
                    # tanh equation for backpropagation (derrivative)
                    delta = (hidden_states[i + 1] - targets) * (1 - np.power(hidden_states[i + 1],
                                                                             2))  # S = (t - O) * (1 - O^2) where O represents the ouput with the tanh activation function
                elif activation_function == "relu":
                    # relu equation for backpropagation (derrivative)
                    delta = (hidden_states[i + 1] - targets) * np.heaviside(hidden_states[i + 1],
                                                                            0)  # S = (t - O) * Relu' where relu' is 1 for x > 0 and 0, otherwise.
                else:
                    print("Invalid Activation function. Please choose between sigmoid, tanh, or relu.")
                    break

            # CASE II: if we are at any hidden layer
            else:
                # conditional statements what will calculate based on the activation function chosen (sigmoid, tanh, relu)
                if activation_function == "sigmoid":
                    # sigmoid equation for backpropagation (derrivative)
                    delta = np.multiply(delta, hidden_states[i + 1] * (
                                1 - hidden_states[i + 1]))  # S = O * (1 - O) sum (Sk * W)
                elif activation_function == "tanh":
                    # tanh equation for backpropagation (derrivative)
                    # delta = (1 - np.power(hidden_states[i+1], 2)) * np.dot(delta, network_layers[i+1][0].T) # S = (1 - O^2) * sum (Sk * W)
                    delta = np.multiply(delta, (1 - np.power(hidden_states[i + 1], 2)))  # S = (1 - O^2) * sum (Sk * W)
                elif activation_function == "relu":
                    # relu equation for backpropagation (derrivative)
                    # delta = np.heaviside(hidden_states[i+1], 0) * np.dot(delta, network_layers[i+1][0].T) # S = relu' * sum (Sk * W) where relu' is 1 for x > 0 and 0, otherwise.
                    delta = np.multiply(delta, np.heaviside(hidden_states[i + 1], 0))
                else:
                    print("Invalid Activation function. Please choose between sigmoid, tanh, or relu.")
                    break

            # Calculate delta W
            w_update = hidden_states[i].T @ delta  # deltaW = S * W.T
            b_update = np.mean(delta, axis=0)

            # Update weights and biases in our matrix
            network_layers[i][0] -= w_update * learning_rate  # deltaW * n
            network_layers[i][1] -= b_update * learning_rate

            delta = delta @ network_layers[i][0].T

        return network_layers

    def mse(self, actual, predicted):
        return (actual - predicted) ** 2

    def mse_grad(self, actual, predicted):
        return predicted - actual

    def train_dataset(self, learning_rate, epochs, batch_size, activation_function, layers_to_create, X_train, y_train):
        # ---- Train our dataset -----
        # Create network layers with specified architecture
        network_layers = self.create_network_layers(layers_to_create)

        print("Now we will train our dataset, these are the current Hyperparameters:")
        print("Learning Rate:", learning_rate)
        print("Epochs:", epochs)
        print("Batch Size:", batch_size)
        print("Activation Function:", activation_function)
        print("Layers to Create:", layers_to_create)
        print("")

        # Iterating our neural network for training
        for epoch in range(epochs):
            epoch_loss = 0  # loss across the whole epoch (forward + backward pass)

            # iterating over rows in our X_train, our step for the loop is the batch_size (so going over x samples every loop)
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[
                          i: (i + batch_size)]  # basically grabbing 8 data samples from our X_train, every loop
                y_batch = y_train[i: (i + batch_size)]  # grabbing 8 targets from our y_train, every loop

                predictions, hidden_states = self.forward_pass(X_batch, network_layers, activation_function)

                # ---optimization using stochastic gradient descent---
                # After the forward pass, we need to calculate the error/loss/cost
                # To do this we can use the mean squared error
                loss = self.mse_grad(y_batch, predictions)
                epoch_loss += np.mean(loss ** 2)  # Accumulate the loss for each batch

                # pass the loss as delta (i.e gradient for optimization)
                network_layers = self.backward_pass(network_layers, hidden_states, loss, y_batch,
                                                         learning_rate,
                                                         activation_function)

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / (X_train.shape[0] / batch_size)

            print("Epoch: " + str(epoch) + " Train MSE: " + "{:.4f}".format(avg_epoch_loss))

        return network_layers

    def test_dataset(self, network_layers, batch_size, activation_function, X_test, y_test):
        test_predictions = []

        for i in range(0, X_test.shape[0], batch_size):
            X_batch = X_test[i: (i + batch_size)]
            y_batch = y_test[i: (i + batch_size)]

            # Forward pass
            prediction, _ = self.forward_pass(X_batch, network_layers, activation_function)
            test_predictions.append(prediction[0])

        # Convert the list of predictions into a numpy array
        test_predictions = np.concatenate(test_predictions)

        # Calculate the test MSE
        test_mse = np.mean((test_predictions - y_test) ** 2)
        print("")
        print("Test MSE: " + "{:.4f}".format(test_mse))
        print("")

        return test_predictions

    def prediction_results(self, test_predictions, numerical_normalized_df, y_test):
        # lets do an inverse transformation on our predictions to get back the charges in $

        # Firs we need the min and max of charges which we have from our numerical_df
        charges_min = numerical_normalized_df['charges'].min()
        charges_max = numerical_normalized_df['charges'].max()

        # Inverse transform predictions to original scale
        predictions_original_scale = test_predictions * (charges_max - charges_min) + charges_min

        # Inverse transform y_test to original scale
        y_test_original_scale = y_test * (charges_max - charges_min) + charges_min

        predictions_df = pd.DataFrame()
        predictions_df['Actual Charges'] = y_test_original_scale
        predictions_df['Predicted Charges'] = predictions_original_scale
        print(predictions_df)
        print("")
        print("Note: Due to vanishing gradient, sometimes we get Nan as values, but running the script a couple of times will give results since random weights and biases will change")

def main():
    # creates an instance of neural network class
    neural_network = Neural_Network()

    # loads our dataset url
    original_df = neural_network.load_dataset()

    # pre-processes our original dataset and gives various statistics about it
    numerical_normalized_df = neural_network.pre_process_dataset(original_df)

    # Splits dataset into train and test (80/20 split)
    X_train, X_test, y_train, y_test = neural_network.train_test_split(numerical_normalized_df)

    # Trains our neural network with the hyperparameters provided
    trained_network = neural_network.train_dataset(learning_rate=1e-6, epochs=10, batch_size=8, activation_function="sigmoid", layers_to_create=[11, 5, 5, 1], X_train = X_train, y_train = y_train)

    # Test our model
    y_predictions = neural_network.test_dataset(network_layers = trained_network, batch_size = 8, activation_function = "sigmoid", X_test = X_test, y_test = y_test)

    # creates actual and prediction table
    neural_network.prediction_results(y_predictions, numerical_normalized_df, y_test)


if __name__ == "__main__":
    main()