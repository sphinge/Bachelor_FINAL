//g++ -o cancer binary_cancer.cpp
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

std::ofstream acc_file("cancer_plots/acc.csv");
std::ofstream conf_file("cancer_plots/confusion.csv");

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

using namespace std;

class LogisticRegression {
public:
    vector<double> weights;
    double learning_rate;
    int epochs;
    double bias = 0.0;

    LogisticRegression(int num_features, double learning_rate, int epochs) {
        this->weights.resize(num_features, 0.5);
        this->learning_rate = learning_rate;
        this->epochs = epochs;

    }

    double predict(vector<double> &features) {
        double z = 0.0;
        for (int i = 0; i < features.size(); i++) {
            z += features[i] * weights[i];
        }
        z += bias;
        return sigmoid(z);
    }

    double evaluateModel(LogisticRegression& model, vector<vector<double>>& X, vector<double>& y) {
        int correct = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            double predicted = round(model.predict(X[i]));  // Assuming predict returns a probability
            if (predicted == y[i]) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / y.size();
    }

    void train(vector<vector<double>> &X, vector<double> &y, vector<vector<double>>& X_test, vector<double>& y_test) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < X.size(); i++) {
                double prediction = predict(X[i]);
                for (int j = 0; j < weights.size(); j++) {
                    weights[j] += learning_rate * (y[i] - prediction) * X[i][j];
                }
                bias += learning_rate * (y[i] - prediction);
            }
            // Evaluate model after each epoch
            double accuracy = evaluateModel(*this, X_test, y_test);
            cout << "Epoch " << epoch + 1 << ", Accuracy: " << accuracy << endl;
            acc_file << epoch+1 << "," << accuracy * 100 << endl;

        }
    }
};

double evaluateModel(LogisticRegression& model, vector<vector<double>>& X, vector<double>& y) {
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        double predicted = round(model.predict(X[i]));  // Assuming predict returns a probability
        conf_file << "(" << predicted << "," << y[i] << "),";
        if (predicted == y[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / y.size();
}

vector<double> split(const string &s, char delimiter) {
    vector<double> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(stod(token));
    }
    return tokens;
}

int main() {
    vector<vector<double>> X;
    vector<double> y;
    
    ifstream file("../data/cancer_train.csv");
    string line;
    
    if (file.is_open()) {
        while (getline(file, line)) {
            vector<double> values = split(line, ',');
            if (values.size() > 1) {
                y.push_back(values[0]);
                values.erase(values.begin());
                X.push_back(values);
            }

        }
    } else {
        cout << "Could not open the file." << endl;
        return 1;
    }

    vector<vector<double>> X_test;
    vector<double> y_test;

    ifstream file_test("../data/cancer_test.csv");
    string line_test;
    
    if (file_test.is_open()) {
        while (getline(file_test, line_test)) {
            vector<double> values_test = split(line_test, ',');
            if (values_test.size() > 1) {
                y_test.push_back(values_test[0]);
                values_test.erase(values_test.begin());
                X_test.push_back(values_test);
            }
        }
    } else {
        cout << "Could not open the file." << endl;
        return 1;
    }

    LogisticRegression model(30, 0.0001, 80);
    model.train(X, y, X_test, y_test);

    double accuracy = evaluateModel(model, X_test, y_test);
    
    cout << "Model accuracy: " << accuracy << endl;

    return 0;
}