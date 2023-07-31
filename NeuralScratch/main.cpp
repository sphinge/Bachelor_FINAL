//g++ main.cpp Neural.cpp -o neural_network -std=c++11
// main.cpp
#include "Neural.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Function to read the Iris dataset from a CSV file
void ReadIrisData(std::string filename, std::vector<RowVector*>& data, std::vector<RowVector*>& labels)
{
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Read data from the CSV file
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        RowVector* input = new RowVector(4); // 4 features in Iris dataset
        RowVector* label = new RowVector(3); // 3 classes (one-hot encoded)

        // Read features (sepal length, sepal width, petal length, petal width)
        for (int i = 0; i < 4; ++i) {
            float value;
            char delimiter;
            ss >> value;
            input->coeffRef(0, i) = value;
            if (i < 3) ss >> delimiter; // Read the comma delimiter
        }

        // Read label (species) and convert it to one-hot encoded vector
        std::string species;
        ss >> species;
        //std::cout << "Read species: '" << species << "', length: " << species.length() << std::endl;  // Add this line
        label->setZero();  // Set all classes to 0 by default
        if (species == ",Setosa") {
            label->coeffRef(0) = 1.0; // Set the appropriate class to 1
        } else if (species == ",Versicolor") {
            label->coeffRef(1) = 1.0;
        } else if (species == ",Virginica") {
            label->coeffRef(2) = 1.0;
        }


        // Add the data and labels to the vectors
        data.push_back(input);
        //std::cout << "Label: " << label << std::endl;
        labels.push_back(label);
    }

    file.close();
}

typedef std::vector<RowVector*> data;


int main()
{
    // Create a neural network with topology: 4 input neurons, 3 hidden neurons, 3 output neurons (one-hot encoded)
    NeuralNetwork n({4, 3, 3});

    // Read training data from CSV files
    data train_data, train_labels;
    ReadIrisData("iris.csv", train_data, train_labels);

    // Train the neural network with the training data
    n.train(train_data, train_labels);

    // Read test data from CSV files
    data test_data, test_labels;
    ReadIrisData("iris_test.csv", test_data, test_labels);

    // Evaluate the neural network on the test data
    n.test(test_data, test_labels);

    // Free memory for training data and labels
    for (RowVector* data : train_data) {
        delete data;
    }
    for (RowVector* label : train_labels) {
        delete label;
    }

    // Free memory for test data and labels
    for (RowVector* data : test_data) {
        delete data;
    }
    for (RowVector* label : test_labels) {
        delete label;
    }

    return 0;
}
