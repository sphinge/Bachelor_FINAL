// g++ eigen_wine.cpp -I/usr/include/eigen3  -o wine
// TODO: make it batch wise
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

using namespace Eigen;
using namespace std;

std::ofstream acc_file("wine_plots/acc.csv");
std::ofstream conf_file("wine_plots/confusion.csv");

class NeuralNetwork
{
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize)
    {
        // Initialize weights and biases
        W1 = MatrixXd::Random(hiddenSize, inputSize);
        b1 = VectorXd::Zero(hiddenSize);
        W2 = MatrixXd::Random(outputSize, hiddenSize);
        b2 = VectorXd::Zero(outputSize);
    }

    VectorXd relu(const VectorXd &z)
    {
        return z.array().max(0.0);
    }
    VectorXd softmax(const VectorXd &z)
    {
        VectorXd expZ = z.array().exp();
        return expZ / expZ.sum();
    }
    VectorXd relu_derivative(const VectorXd &z)
    {
        return (z.array() > 0.0).cast<double>();
    }
    VectorXd forward(const VectorXd &x)
    {
        hiddenLayer = relu(W1 * x + b1);
        outputLayer = softmax(W2 * hiddenLayer + b2);
        return outputLayer;
    }

    void train(const MatrixXd &X, const MatrixXd &y, const MatrixXd &X_test, const MatrixXd &y_test, double learningRate, int epochs)
    {
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            for (int i = 0; i < X.rows(); ++i)
            {
                // Forward pass
                VectorXd x = X.row(i);
                VectorXd target = y.row(i);
                forward(x);

                // Backpropagation
                VectorXd dL_dz2 = outputLayer - target;
                VectorXd dL_dh = W2.transpose() * dL_dz2;
                VectorXd dL_dz1 = dL_dh.array() * relu_derivative(hiddenLayer).array();
                ;

                // Update weights and biases
                W2 -= learningRate * dL_dz2 * hiddenLayer.transpose();
                b2 -= learningRate * dL_dz2;
                W1 -= learningRate * dL_dz1 * x.transpose();
                b1 -= learningRate * dL_dz1;
            }

            // Evaluate the accuracy at the end of each epoch and print it
            double accuracy = evaluateAccuracy(X_test, y_test);
            acc_file << epoch + 1 << "," << accuracy * 100 << endl;
        }
    }

    int predict(const VectorXd &x)
    {
        VectorXd probabilities = forward(x);
        int predictedClass = 0;
        double maxProbability = probabilities[0];

        for (int i = 1; i < probabilities.size(); ++i)
        {
            if (probabilities[i] > maxProbability)
            {
                maxProbability = probabilities[i];
                predictedClass = i;
            }
        }

        return predictedClass;
    }

    double evaluateAccuracy(const MatrixXd &X, const MatrixXd &y) {
        int correct = 0;
        for (int i = 0; i < X.rows(); ++i) {
            int real_label = -1;
            for (int j = 0; j < y.cols(); ++j) {
                if (y(i, j) == 1.0) {
                    real_label = j;
                    break;
                }
            }
            int predicted_label = predict(X.row(i));
            if (real_label == predicted_label) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / X.rows();
    }

private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    MatrixXd W1, W2;
    VectorXd b1, b2;
    VectorXd hiddenLayer, outputLayer;
};

pair<MatrixXd, MatrixXd> readCSV(const string &filename, int numFeatures, int numLabels)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    vector<vector<double>> data;
    string line;
    while (getline(file, line))
    {
        stringstream ss(line);
        vector<double> row;
        double val;
        while (ss >> val)
        {
            row.push_back(val);
            if (ss.peek() == ',')
            {
                ss.ignore();
            }
        }
        data.push_back(row);
    }

    int numRows = data.size();
    MatrixXd X(numRows, numFeatures);
    MatrixXd y(numRows, numLabels);
    y.setZero();

    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numFeatures; ++j)
        {
            X(i, j) = data[i][j];
        }

        int labelIndex = static_cast<int>(data[i].back()); // Last value is the label
        y(i, labelIndex - 1) = 1.0; // Subtract 1 to accommodate the new label system
    }

    return make_pair(X, y);
}


int main()
{
    string filename_train = "../data/wine_train.csv"; // Replace with your CSV file name
    int numFeatures = 13; // Number of features
    int numLabels = 3;  // Number of label classes (you should change this based on your dataset)

    pair<MatrixXd, MatrixXd> data_train = readCSV(filename_train, numFeatures, numLabels);
    MatrixXd X_train = data_train.first;
    MatrixXd y_train = data_train.second;

    string filename_test = "../data/wine_test.csv"; // Replace with your CSV file name
    pair<MatrixXd, MatrixXd> data_test = readCSV(filename_test, numFeatures, numLabels);
    MatrixXd X_test = data_test.first;
    MatrixXd y_test = data_test.second;

    // Create and train the neural network
    int inputSize = X_train.cols();
    int hiddenSize = 10;
    int outputSize = y_train.cols();
    NeuralNetwork nn(inputSize, hiddenSize, outputSize);

    double learningRate = 0.001;
    int epochs = 100;

    nn.train(X_train, y_train, X_test, y_test, learningRate, epochs);

    for (int i = 0; i < X_test.rows(); ++i) {
        VectorXd x = X_test.row(i);
        int activatedNeurons = nn.predict(x);
        int real_label = -1;
        for (int j = 0; j < y_test.cols(); ++j) {
            if (y_test(i, j) == 1.0) {
                real_label = j;
                break;
            }
        }
        conf_file << "(" << activatedNeurons << "," << real_label << "),";
    }

    return 0;
}