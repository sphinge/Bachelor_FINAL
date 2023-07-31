// NeuralNetwork.cpp
#include "Neural.hpp"
#include <cmath> 

// Constructor of the neural network class
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate)
{
    this->learningRate = learningRate;
    for (uint i = 0; i < topology.size(); i++) {
        // Initialize neuron layers
        if (i == topology.size() - 1)
            neuronLayers.push_back(new RowVector(topology[i]));
        else
            neuronLayers.push_back(new RowVector(topology[i] + 1));

        // Initialize cache and delta vectors
        cacheLayers.push_back(new RowVector(neuronLayers[i]->size()));
        deltas.push_back(new RowVector(neuronLayers[i]->size()));

        // Add bias neuron (set to 1.0) to all layers except the output layer
        if (i != topology.size() - 1) {
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(topology[i]) = 1.0;
        }

        // Initialize weights matrix
        if (i > 0) {
            if (i != topology.size() - 1) {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                weights.back()->setRandom();
                weights.back()->col(topology[i]).setZero();
                weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
            } else {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                weights.back()->setRandom();
            }
        }
    }
}

// Activation function (tanh)
Scalar activationFunction(Scalar x)
{
    return tanhf(x);
}

// Derivative of the activation function (tanh)
Scalar activationFunctionDerivative(Scalar x)
{
    return 1 - tanhf(x) * tanhf(x);
}

// Function for forward propagation of data
void NeuralNetwork::propagateForward(RowVector& input)
{
    // Set the input to the input layer
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

    // Propagate the data forward and apply the activation function to your network
    for (uint i = 1; i < neuronLayers.size(); i++) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
        neuronLayers[i]->block(0, 0, 1, neuronLayers[i]->size()).unaryExpr(std::ptr_fun(activationFunction));
    }
}

// Function to calculate errors made by neurons in each layer
void NeuralNetwork::calcErrors(RowVector& output)
{
    // Calculate the errors made by neurons of the last layer
    (*deltas.back()) = output - (*neuronLayers.back());

    // Error calculation of hidden layers
    for (uint i = deltas.size() - 2; i > 0; i--) {
        (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
        for (uint j = 0; j < deltas[i]->size(); j++) {
            deltas[i]->coeffRef(j) *= activationFunctionDerivative(neuronLayers[i]->coeffRef(j));
        }
    }
}


// Function to update the weights of connections
void NeuralNetwork::updateWeights()
{
    double lambda = 0.1;
    for (uint i = 0; i < weights.size(); i++) {
        for (uint c = 0; c < weights[i]->cols(); c++) {
            for (uint r = 0; r < weights[i]->rows(); r++) {
                // Consider the bias neuron (always set to 1.0) for all layers
                Scalar delta = learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                Scalar regularizationTerm = learningRate * lambda * (*weights[i])(r, c); // This is the L2 regularization term
                weights[i]->coeffRef(r, c) += delta - regularizationTerm;  // subtract the regularization term
            }
        }
    }
}


// Function for backward propagation of errors made by neurons
void NeuralNetwork::propagateBackward(RowVector& output)
{
    calcErrors(output);
    updateWeights();
}



// Function to train the neural network with the training data and labels
void NeuralNetwork::train(std::vector<RowVector*>& data, std::vector<RowVector*>& labels)
{
    for (uint i = 0; i < data.size(); i++) {
        //std::cout << "Input to neural network is: " << *data[i] << std::endl;
        propagateForward(*data[i]);
        //std::cout << "Output produced is: " << *neuronLayers.back() << std::endl;
        propagateBackward(*labels[i]); // Use labels for supervised learning
        //std::cout << "Train Labels: " << *labels[i] << std::endl;
        std::cout << "i: " << i << " MSE: " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl;
    }
}

// Function to evaluate the neural network on the test data
void NeuralNetwork::test(std::vector<RowVector*>& data, std::vector<RowVector*>& labels)
{
    Scalar totalSquaredError = 0.0;
    int numSamples = data.size();

    for (int i = 0; i < numSamples; i++) {
        propagateForward(*data[i]);

        // Calculate the squared error between the predicted output and the true label
        RowVector& predictedOutput = *neuronLayers.back();
        RowVector& trueLabel = *labels[i];
        RowVector error = predictedOutput - trueLabel;
        Scalar squaredError = error.squaredNorm();

        totalSquaredError += squaredError;
        //std::cout << "i: " << i << " Predicted: " << predictedOutput << " True Label: " << trueLabel << " Error: " << squaredError << std::endl;
    }

    // Calculate the Mean Squared Error (MSE) over the test data
    Scalar mse = totalSquaredError / numSamples;

    std::cout << "Mean Squared Error (MSE) on test data: " << mse << std::endl;
}
