//g++ -o wine multiclass_wine.cpp
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

std::ofstream acc_file("wine_plots/acc.csv");
std::ofstream conf_file("wine_plots/confusion.csv");

using namespace std;

vector<double> softmax(vector<double> &z) {
    vector<double> exp_values(z.size());
    double exp_sum = 0.0;
    for (int i = 0; i < z.size(); ++i) {
        exp_values[i] = exp(z[i]);
        exp_sum += exp_values[i];
    }
    for (int i = 0; i < exp_values.size(); ++i) {
        exp_values[i] /= exp_sum;
    }
    return exp_values;
}

class SoftmaxRegression {
public:
    vector<vector<double>> weights; // 2D array
    double learning_rate;
    int epochs;

    SoftmaxRegression(int num_classes, int num_features, double learning_rate, int epochs) {
        this->weights.resize(num_classes, vector<double>(num_features, 0.0));
        this->learning_rate = learning_rate;
        this->epochs = epochs;
    }

    vector<double> predict(vector<double> &features) {
        vector<double> z(weights.size(), 0.0);
        for (int i = 0; i < weights.size(); ++i) {
            for (int j = 0; j < features.size(); ++j) {
                z[i] += features[j] * weights[i][j];
            }
        }
        return softmax(z);
    }

    double evaluateModel(SoftmaxRegression& model, vector<vector<double>>& X, vector<int>& y) {
        int correct = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            vector<double> probs = model.predict(X[i]);
            int predicted = max_element(probs.begin(), probs.end()) - probs.begin();
            //cout << "predicted:" << predicted << " real:" << y[i] << endl;
            if (predicted == y[i]) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / y.size();
    }

    void train(vector<vector<double>> &X, vector<int> &y, vector<vector<double>>& X_test, vector<int>& y_test) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < X.size(); ++i) {
                vector<double> prediction = predict(X[i]);
                for (int c = 0; c < weights.size(); ++c) {
                    for (int f = 0; f < weights[0].size(); ++f) {
                        weights[c][f] += learning_rate * ((y[i] == c ? 1.0 : 0.0) - prediction[c]) * X[i][f];
                    }
                }
            }
            // Evaluate and print training accuracy after each epoch
            double train_accuracy = evaluateModel(*this, X_test, y_test); // Evaluate on test set
            cout << "Epoch " << (epoch+1) << ": Test accuracy = " << train_accuracy << endl;
            acc_file << epoch+1 << "," << train_accuracy * 100 << endl;
        }
    }
};

double evaluateModel(SoftmaxRegression& model, vector<vector<double>>& X, vector<int>& y) {
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        vector<double> probs = model.predict(X[i]);
        int predicted = max_element(probs.begin(), probs.end()) - probs.begin();
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
    vector<int> y;
    
    ifstream file("../data/wine_train.csv");
    string line;
    
    if (file.is_open()) {
        while (getline(file, line)) {
            vector<double> values = split(line, ',');
            if (values.size() == 14) { // 4 features + 1 class label
                y.push_back(static_cast<int>(values.back())-1);
                values.pop_back();
                X.push_back(values);
            }
        }
    } else {
        cout << "Could not open the file." << endl;
        return 1;
    }

    vector<vector<double>> X_test;
    vector<int> y_test;

    ifstream file_test("../data/wine_test.csv");
    string line_test;
    
    if (file_test.is_open()) {
        while (getline(file_test, line_test)) {
            vector<double> values_test = split(line_test, ',');
            if (values_test.size() == 14) { // 4 features + 1 class label
                y_test.push_back(static_cast<int>(values_test.back())-1);
                values_test.pop_back();
                X_test.push_back(values_test);
            }
        }
    } else {
        cout << "Could not open the file." << endl;
        return 1;
    }

    SoftmaxRegression model(3 /* num_classes */, 13 /* num_features */, 0.001, 100);
    model.train(X, y, X_test, y_test);  // passing test set for evaluation

    double accuracy = evaluateModel(model, X_test, y_test);
    cout << "Model accuracy: " << accuracy << endl;

    return 0;
}