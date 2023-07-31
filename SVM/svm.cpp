#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
using namespace std;

struct IrisData {
    double sepal_length;
    double sepal_width;
    double petal_length;
    double petal_width;
    double label; // Change the label to a double type

    vector<double> getFeatures() const {
        return {sepal_length, sepal_width, petal_length, petal_width};
    }

    int getLabel() const {
        return static_cast<int>(label);
    }
};

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
        char comma;
        ss >> row.sepal_length >> comma >> row.sepal_width >> comma
           >> row.petal_length >> comma >> row.petal_width >> comma >> row.label;
        data.push_back(row);
    }

    file.close();
    return data;
}

double getHingeLoss(vector<double> &x, int &y, vector<double> &w, double &b)
{
    double dot_product = 0;
    for(size_t i = 0; i < x.size(); i++)
        dot_product += w[i] * x[i];
    double loss = 1 - y * (dot_product + b);
    return (loss > 0) ? loss : 0;
}

int predict(const vector<double>& x, const vector<vector<double>>& W, const vector<double>& b)
{
    double max_val = -1e9;  // Initialize to a very small value
    int max_k = -1;
    for (int k = 0; k < W.size(); k++) {
        double val = 0;
        for (int i = 0; i < W[k].size(); i++) {
            val += W[k][i] * x[i];
        }
        val += b[k];
        if (val > max_val) {
            max_val = val;
            max_k = k;
        }
    }
    return max_k;
}

double getAccuracy(const vector<vector<double>>& X, const vector<int>& y, 
                   const vector<vector<double>>& W, const vector<double>& b) 
{
    int correct = 0;
    for (int i = 0; i < X.size(); i++) {
        if (predict(X[i], W, b) == y[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / X.size();
}

void trainSVM(vector<vector<double>> &X, vector<int> &y, vector<vector<double>> &W, vector<double> &b, vector<double> &db, vector<vector<double>> &dW, int &n_features, int &n_classes, double &lrate)

{
    
    int iter = 0;
    int n = X.size();
    
    while(true)
    {
        vector<double> cost(n_classes, 0.0);
        
        for(int k = 0; k < n_classes; k++)
        {
            for(int i = 0; i < n; i++)
            {
                int y_i = (y[i] == k) ? 1 : -1;
                double loss = getHingeLoss(X[i], y_i, W[k], b[k]);
                //std::cout << "Loss: " << loss << std::endl;
                cost[k] += loss;
                if(loss > 0)
                {
                    for(int j = 0; j < n_features; j++)
                    {
                        dW[k][j] += (-X[i][j] * y_i);
                    }
                    db[k] += -y_i;
                }
            }
            
            cost[k] /= n;
            for(int j = 0; j < n_features; j++)
            {
                dW[k][j] /= n;
            }
            db[k] /= n;
        }

        // Calculate average cost for this iteration
        double avg_cost = std::accumulate(cost.begin(), cost.end(), 0.0) / n_classes;

        if(iter++ > 15000)
        {
            for(int k = 0; k < n_classes; k++)
            {
                cout << "y = ";
                for(int j = 0; j < n_features; j++)
                {
                    cout << W[k][j] << " * x" << (j+1);
                    if(j < n_features - 1) 
                        cout << " + ";
                }
                cout << " + " << b[k] << "\n";
            }
            break;
        }

        for(int k = 0; k < n_classes; k++)
        {
            for(int j = 0; j < n_features; j++)
            {
                W[k][j] -= lrate * dW[k][j];
            }
            b[k] -= lrate * db[k];
        }

        if (iter % 100 == 0) {
            cout << "Iteration " << iter << ", training accuracy: " << getAccuracy(X, y, W, b) << ", average error: " << avg_cost << "\n";
        }
    }
}



void evaluateModel(const vector<vector<double>>& X_test, const vector<int>& Y_test, 
                   const vector<vector<double>>& W, const vector<double>& b) 
{
    int correct = 0;
    for (int i = 0; i < X_test.size(); i++) {
        if (predict(X_test[i], W, b) == Y_test[i]) {
            correct++;
        }
    }
    double accuracy = static_cast<double>(correct) / X_test.size();
    cout << "Accuracy: " << accuracy << endl;
}



int main() {
    vector<IrisData> trainingData = readCSV("iris.csv");

    vector<vector<double>> X;
    vector<int> Y;
    for (const auto& data : trainingData) {
        X.push_back(data.getFeatures());
        Y.push_back(data.getLabel());
    }

    int n_features = 4;
    int n_classes = 3;
    double lrate = 0.0005;
    
    vector<double> db(n_classes, 0.0);
    vector<vector<double>> dW(n_classes, vector<double>(n_features, 0.0));
    vector<vector<double>> W(n_classes, vector<double>(n_features, 1.0));
    vector<double> b(n_classes, 0.0);

    trainSVM(X, Y, W, b, db, dW, n_features, n_classes, lrate);

    vector<IrisData> testData = readCSV("iris_test.csv");
    
    vector<vector<double>> X_test;
    vector<int> Y_test;
    for (const auto& data : testData) {
        X_test.push_back(data.getFeatures());
        Y_test.push_back(data.getLabel());
    }

    // Evaluate the model
    evaluateModel(X_test, Y_test, W, b);



    return 0;
}
