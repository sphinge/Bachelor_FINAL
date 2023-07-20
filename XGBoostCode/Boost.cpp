#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <xgboost/c_api.h>

struct IrisData {
    std::vector<std::vector<float>> features;
    std::vector<int> class_labels; // Change the class_labels type to int
};

IrisData loadIrisData(const std::string& filename) {
    IrisData iris_data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return iris_data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row_features;
        std::istringstream iss(line);
        std::string value;

        // Read the four feature values (sepal_length, sepal_width, petal_length, petal_width)
        for (int i = 0; i < 4; i++) {
            if (!std::getline(iss, value, ',')) {
                std::cerr << "Error reading feature value from line: " << line << std::endl;
                file.close();
                return iris_data;
            }
            row_features.push_back(std::stof(value));
        }

        // Read the class label (species)
        if (!std::getline(iss, value, ',')) {
            std::cerr << "Error reading class label from line: " << line << std::endl;
            file.close();
            return iris_data;
        }
        iris_data.class_labels.push_back(std::stoi(value)); // Convert label to int
        iris_data.features.push_back(row_features);
    }

    file.close();
    return iris_data;
}


int main() {
    // Load Iris dataset from CSV
    IrisData training_data = loadIrisData("iris.csv");
    IrisData test_data = loadIrisData("iris_test.csv");

    // Check if the data was loaded successfully for training data
    if (training_data.features.empty() || training_data.class_labels.empty() ||
        training_data.features[0].size() != 4 || training_data.features.size() != training_data.class_labels.size()) {
        std::cerr << "Error loading training Iris dataset." << std::endl;
        return 1;
    }

    // Convert training data to DMatrix
    DMatrixHandle h_train[1];
    XGDMatrixCreateFromMat((float*)training_data.features.data(), training_data.features.size(), 4, -1, &h_train[0]);

    // Convert the class labels to unsigned int
    std::vector<unsigned int> uint_class_labels(training_data.class_labels.begin(), training_data.class_labels.end());

    // Load the labels for training data
    XGDMatrixSetUIntInfo(h_train[0], "label", uint_class_labels.data(), uint_class_labels.size());

    // Create the booster and load some parameters
    BoosterHandle h_booster;
    XGBoosterCreate(h_train, 1, &h_booster);
    XGBoosterSetParam(h_booster, "booster", "gbtree");
    XGBoosterSetParam(h_booster, "objective", "multi:softmax"); // Set the objective function for multi-class classification
    XGBoosterSetParam(h_booster, "num_class", "3"); // Set the number of classes (3 classes: 0, 1, 2)
    XGBoosterSetParam(h_booster, "max_depth", "5");
    XGBoosterSetParam(h_booster, "eta", "0.1");
    XGBoosterSetParam(h_booster, "min_child_weight", "1");
    XGBoosterSetParam(h_booster, "subsample", "0.5");
    XGBoosterSetParam(h_booster, "colsample_bytree", "1");
    XGBoosterSetParam(h_booster, "num_parallel_tree", "1");

    // Perform 200 learning iterations
    for (int iter = 0; iter < 200; iter++)
        XGBoosterUpdateOneIter(h_booster, iter, h_train[0]);

    // Convert test data to DMatrix
    DMatrixHandle h_test;
    XGDMatrixCreateFromMat((float*)test_data.features.data(), test_data.features.size(), 4, -1, &h_test);
    bst_ulong out_len = 0;
    const float *f = nullptr;
    XGBoosterPredict(h_booster, h_test, 0, 0, 0, &out_len, &f);

    // Print the predictions for the test data
    for (unsigned int i = 0; i < out_len; i++)
        std::cout << "prediction[" << i << "]=" << static_cast<int>(f[i]) << std::endl; // Convert the output to int

    // Free XGBoost internal structures
    XGDMatrixFree(h_train[0]);
    XGDMatrixFree(h_test);
    XGBoosterFree(h_booster);

    return 0;
}
