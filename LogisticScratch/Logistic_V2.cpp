#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>

using namespace std;

// Simple vector and matrix data structures with basic operations
struct Vector {
    vector<double> data;
    Vector() {}
    explicit Vector(int size) : data(size, 0.0) {}
    Vector(int size, double value) : data(size, value) {} // Added constructor
    int size() const { return data.size(); }
    double& operator[](int i) { return data[i]; }
    const double& operator[](int i) const { return data[i]; }
};

struct Matrix {
    vector<Vector> data;
    int rows() const { return data.size(); }
    int cols() const { return data[0].size(); }
    Vector& operator[](int i) { return data[i]; }
    const Vector& operator[](int i) const { return data[i]; }
};

// Sigmoid function
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Dot product of two vectors
double dot(const Vector& a, const Vector& b) {
    double result = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Vector addition
Vector add(const Vector& a, const Vector& b) {
    Vector result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// Scalar multiplication of a vector
Vector multiply(const Vector& v, double scalar) {
    Vector result(v.size());
    for (int i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}

// Matrix multiplication
Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix result;
    result.data.resize(a.rows());
    for (int i = 0; i < a.rows(); ++i) {
        result[i].data.resize(b.cols());
        for (int j = 0; j < b.cols(); ++j) {
            result[i][j] = dot(a[i], b[j]);
        }
    }
    return result;
}

// Transpose of a matrix
Matrix transpose(const Matrix& a) {
    Matrix result;
    result.data.resize(a.cols());
    for (int i = 0; i < a.cols(); ++i) {
        result[i].data.resize(a.rows());
        for (int j = 0; j < a.rows(); ++j) {
            result[i][j] = a[j][i];
        }
    }
    return result;
}
// Logistic Regression class
class LogisticRegression {
private:
    std::vector<Vector> weights; // Three sets of weights for three binary classifiers
    double learningRate;
    std::vector<double> errors;
    std::vector<double> accuracy;

public:
    explicit LogisticRegression(int numFeatures, double learningRate)
        : learningRate(learningRate), weights(3, Vector(numFeatures)) {}

    // Public getter for accessing the weights of a specific classifier
    const Vector& getWeights(int classLabel) const { return weights[classLabel]; }

    // Public setter for updating the weights of a specific classifier
    void setWeights(int classLabel, const Vector& newWeights) { weights[classLabel] = newWeights; }

    // Public getter for accessing the learning rate
    double getLearningRate() const { return learningRate; }

    // Public setter for updating the learning rate
    void setLearningRate(double newLearningRate) { learningRate = newLearningRate; }

    double calculateAccuracy(const Matrix& features, const Vector& labels) {
        int numSamples = features.rows();
        int numCorrect = 0;

        for (int i = 0; i < numSamples; ++i) {
            double predictedClass = predict(features[i]);
            if (predictedClass == labels[i]) {
                numCorrect++;
            }
        }

        return static_cast<double>(numCorrect) / numSamples;
    }

    std::vector<double>& getErrors() { return errors; }
    std::vector<double>& getAccuracy() { return accuracy; }

    // Train the model using gradient descent for a specific class label
    void train(const Matrix& features, const Vector& labels, int numEpochs, int classLabel) {
        int numSamples = features.rows();
        int numFeatures = features.cols();

        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            Vector gradient(numFeatures, 0.0); // Initialize the gradient vector with zeros
            double totalError = 0.0; 
            int numCorrect = 0;

            for (int i = 0; i < numSamples; ++i) {
                // Convert multiclass labels to binary labels (1 for classLabel, 0 for other classes)
                double binaryLabel = (labels[i] == classLabel) ? 1.0 : 0.0;

                double predicted = sigmoid(dot(weights[classLabel], features[i]));
                double error = binaryLabel - predicted;

                totalError += abs(error);

                // Accumulate gradients for weight update
                for (int j = 0; j < numFeatures; ++j) {
                    gradient[j] += features[i][j] * error;
                }

                double roundedPrediction = (predicted > 0.5) ? 1.0 : 0.0;

                if (roundedPrediction == binaryLabel) {
                    ++numCorrect;
                }  
            }
            accuracy.push_back(static_cast<double>(numCorrect) / numSamples);
            errors.push_back(totalError / numSamples);

            // Update weights using gradient descent after processing all samples in the epoch
            Vector update = multiply(gradient, learningRate);
            weights[classLabel] = add(weights[classLabel], update);

            //cout << "Epoch: " << epoch << "Error: " << error << endl;

        }
        
    }

    double predictProbability(const Vector& input, int classLabel) {
        double z = dot(weights[classLabel], input);
        return sigmoid(z);
    }


    // Predict the output for a single sample and return the class label with the highest probability
    double predict(const Vector& input) {
        double maxProbability = -1.0;
        int maxLabel = -1;

        for (int classLabel = 0; classLabel < 3; ++classLabel) {
            double z = dot(weights[classLabel], input);
            double probability = sigmoid(z);

            if (probability > maxProbability) {
                maxProbability = probability;
                maxLabel = classLabel;
            }
        }

        return maxLabel; // Return the predicted class label
    }

    



};


void readCSV(const string& filename, Matrix& X, Vector& y) {
    ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        double val;
        Vector features;
        while (ss >> val) {
            if (ss.peek() == ',') {
                ss.ignore();
            }
            features.data.push_back(val);
        }
        double label = features.data.back(); // The last value is the label
        features.data.pop_back(); // Remove the label from the feature vector
        X.data.push_back(features);
        y.data.push_back(label);
    }
}



// Function to extract features and labels from training data
void extractFeaturesAndLabels(const Matrix& data, Matrix& features, Vector& labels) {
    int numFeatures = data.cols() - 1;
    for (int i = 0; i < data.rows(); ++i) {
        Vector featureVector;
        for (int j = 0; j < numFeatures; ++j) {
            featureVector.data.push_back(data[i][j]);
        }
        features.data.push_back(featureVector);
        labels.data.push_back(data[i][numFeatures]);
    }
}

// Standardization of the data
void standardizeData(Matrix& data) {
    int numFeatures = data.cols();
    for (int j = 0; j < numFeatures; ++j) {
        double mean = 0.0;
        double variance = 0.0;

        for (int i = 0; i < data.rows(); ++i) {
            mean += data[i][j];
        }
        mean /= data.rows();

        for (int i = 0; i < data.rows(); ++i) {
            variance += (data[i][j] - mean) * (data[i][j] - mean);
        }
        variance /= data.rows();

        double stdDev = sqrt(variance);

        for (int i = 0; i < data.rows(); ++i) {
            data[i][j] = (data[i][j] - mean) / stdDev;
        }
    }
}

std::string runPythonCommand(const std::string& command)
{
    std::vector<char> buffer(128); // Use a vector instead of std::array
    std::string result;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    return result;
}

void saveLearningCurveData(const std::vector<std::size_t>& iterations, const std::vector<double>& errors)
{
    // Save the data to a CSV file
    std::ofstream file("learning_curve.csv");
    file << "Iterations,Error" << std::endl;
    for (std::size_t i = 0; i < iterations.size(); ++i) {
        file << iterations[i] << "," << errors[i] << std::endl;
    }
    file.close();
}

int main() {
    // Read training data from CSV
    Matrix train_data; // Matrix to hold the raw data from CSV
    Vector train_labels;
    try {
        readCSV("iris.csv", train_data, train_labels);
    } catch (const exception& e) {
        cerr << "Error reading CSV: " << e.what() << endl;
        return 1;
    }

    // Separate features and labels
    Matrix train_features;
    try {
        extractFeaturesAndLabels(train_data, train_features, train_labels);
    } catch (const exception& e) {
        cerr << "Error extracting features and labels: " << e.what() << endl;
        return 1;
    }

    // Read test data from CSV
    Matrix test_data; // Matrix to hold the raw data from CSV
    Vector test_labels;
    try {
        readCSV("iris_test.csv", test_data, test_labels);
    } catch (const exception& e) {
        cerr << "Error reading CSV: " << e.what() << endl;
        return 1;
    }

    // Separate features and labels for test data
    Matrix test_features;
    try {
        extractFeaturesAndLabels(test_data, test_features, test_labels);
    } catch (const exception& e) {
        cerr << "Error extracting features and labels: " << e.what() << endl;
        return 1;
    }

    // Convert data to Matrix and Vector
    Matrix X_train_std; // You need to fill this matrix with the standardized training features
    Matrix X_test_std;  // You need to fill this matrix with the standardized test features

    // Standardize the training features
    X_train_std = train_features;
    standardizeData(X_train_std);

    // Standardize the test features
    X_test_std = test_features;
    standardizeData(X_test_std);

    int numFeatures = X_train_std.cols();
    double learningRate = 0.001;
    int numEpochs = 1000;

    // Create three binary logistic regression classifiers
    LogisticRegression model0(numFeatures, learningRate);
    LogisticRegression model1(numFeatures, learningRate);
    LogisticRegression model2(numFeatures, learningRate);

    // Train the binary logistic regression classifiers for each class label
    model0.train(X_train_std, train_labels, numEpochs, 0);
    model1.train(X_train_std, train_labels, numEpochs, 1);
    model2.train(X_train_std, train_labels, numEpochs, 2);

    // Get the errors and accuracies
    std::vector<double> errors0 = model0.getErrors();
    std::vector<double> accuracies0 = model0.getAccuracy();

    std::vector<double> errors1 = model1.getErrors();
    std::vector<double> accuracies1 = model1.getAccuracy();

    std::vector<double> errors2 = model2.getErrors();
    std::vector<double> accuracies2 = model2.getAccuracy();

    // Print the errors and accuracies
    for (int i = 0; i < numEpochs; ++i) {
        std::cout << "Epoch " << i << " - Model 0 Error: " << errors0[i] << ", Accuracy: " << accuracies0[i] << std::endl;
        std::cout << "Epoch " << i << " - Model 1 Error: " << errors1[i] << ", Accuracy: " << accuracies1[i] << std::endl;
        std::cout << "Epoch " << i << " - Model 2 Error: " << errors2[i] << ", Accuracy: " << accuracies2[i] << std::endl;
    }


    // Calculate accuracy on test data
    int correctPredictions = 0;
    for (int j = 0; j < test_features.rows(); ++j) {
        double predictedProb0 = model0.predictProbability(X_test_std[j], 0);
        double predictedProb1 = model1.predictProbability(X_test_std[j], 1);
        double predictedProb2 = model2.predictProbability(X_test_std[j], 2);

        // Assign the class label with the highest probability
        double predictedClass = (predictedProb0 >= predictedProb1 && predictedProb0 >= predictedProb2) ? 0.0 :
                                    (predictedProb1 >= predictedProb0 && predictedProb1 >= predictedProb2) ? 1.0 :
                                                                                                            2.0;
                                                                                        2.0;
        //cout << "predicted: " << predictedClass <<  endl;
        //cout << "real: " << test_labels[j] << endl;
        if (predictedClass == test_labels[j]) {
            correctPredictions++;
        }
    }    

    double accuracy = static_cast<double>(correctPredictions) / test_features.rows();
    cout << "Accuracy on Test Data: " << accuracy << endl;

    return 0;
}
