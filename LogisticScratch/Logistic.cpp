#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

// Sigmoid function
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Logistic Regression class
class LogisticRegression {
private:
    vector<double> weights;
    double learningRate;

public:
    LogisticRegression(int numFeatures, double learningRate)
        : learningRate(learningRate) {
        // Initialize weights with random values close to 0
        weights.resize(numFeatures);
        for (int i = 0; i < numFeatures; ++i) {
            weights[i] = 0.01 * (rand() % 100 - 50);
        }
    }

    // Train the model using gradient descent
    void train(const vector<vector<double>>& features, const vector<double>& labels, int numEpochs) {
        int numSamples = features.size();
        int numFeatures = features[0].size();

        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            double totalError = 0.0;

            for (int i = 0; i < numSamples; ++i) {
                double predicted = predict(features[i]);
                double error = labels[i] - predicted;
                totalError += error * error;

                // Update weights using gradient descent
                for (int j = 0; j < numFeatures; ++j) {
                    weights[j] += learningRate * error * features[i][j];
                }
            }

            // Print the average error for this epoch
            cout << "Epoch " << (epoch + 1) << ", Average Error: " << (totalError / numSamples) << endl;
        }
    }

    // Predict the output for a single sample
    double predict(const vector<double>& input) {
        double z = 0.0;
        int numFeatures = input.size();

        for (int i = 0; i < numFeatures; ++i) {
            z += weights[i] * input[i];
        }

        return sigmoid(z);
    }
};

struct IrisData {
    double sepal_length;
    double sepal_width;
    double petal_length;
    double petal_width;
    double label; // Change the label to a double type
};

double convertLabelToDouble(const string& label) {
    if (label == "Setosa")
        return 0.0;
    else if (label == "Versicolor")
        return 1.0;
    else if (label == "Virginica")
        return 2.0;
    else {
        cout << "Error: Unknown label - " << label << endl;
        return -1.0; // You can choose to handle this differently if needed
    }
}

// Function to read data from CSV file
vector<IrisData> readCSV(const string& filename) {
    ifstream file(filename);
    vector<IrisData> data;

    if (!file.is_open()) {
        cout << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        IrisData row;
        char comma; // To read the comma separator
        // Read numeric values
        ss >> row.sepal_length >> comma >> row.sepal_width >> comma
           >> row.petal_length >> comma >> row.petal_width >> comma;

        // Read the label (string) and convert it to a double value
        string label_str;
        getline(ss, label_str, ',');
        row.label = convertLabelToDouble(label_str);

        data.push_back(row);
    }

    file.close();
    return data;
}

int main() {
    // Read training data from CSV
    vector<IrisData> trainingData = readCSV("iris.csv");

    // Extract features and labels from training data
    vector<vector<double>> trainFeatures;
    vector<double> trainLabels;

    for (const auto& data : trainingData) {
        vector<double> features = {data.sepal_length, data.sepal_width, data.petal_length, data.petal_width};
        double label = data.label;

        trainFeatures.push_back(features);
        trainLabels.push_back(label);
    }

    int numFeatures = trainFeatures[0].size();
    double learningRate = 0.001;
    int numEpochs = 1000;

    LogisticRegression model(numFeatures, learningRate);

    // Train the logistic regression model
    model.train(trainFeatures, trainLabels, numEpochs);

    // Read test data from CSV
    vector<IrisData> testData = readCSV("iris_test.csv");

    // Extract features from test data
    vector<vector<double>> testFeatures;
    for (const auto& data : testData) {
        vector<double> features = {data.sepal_length, data.sepal_width, data.petal_length, data.petal_width};
        testFeatures.push_back(features);
    }

    // Test the trained model
    for (size_t i = 0; i < testFeatures.size(); ++i) {
        const auto& sample = testFeatures[i];
        double realLabel = testData[i].label; // Get the real label from the testData

        double predicted = model.predict(sample);

        // Print the real label, predicted probability, and the predicted class
        cout << "Real Label: " << realLabel << ", Predicted Probability: " << predicted;
        cout << ", Predicted Class: " << (predicted < 0.5 ? 0 : 1) << endl;
    }

    return 0;
}
