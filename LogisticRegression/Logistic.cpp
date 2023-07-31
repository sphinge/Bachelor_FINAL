#include <iostream>
#include <shark/Data/Csv.h>
#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/GradientDescent/LBFGS.h>
#include <shark/Algorithms/Trainers/LogisticRegression.h> // Assuming the provided code is in this header file

using namespace shark;
using namespace std;

int main() {
    // Load training data from CSV file (assuming you have a CSV file with features)
    Data<RealVector> data;
    importCSV(data, "iris.csv");
    //std::cout << data.numberOfElements() << " training examples" << std::endl;
    // Data<RealVector> features = data;

    // // Print the data
    // std::cout << "Printing Data<RealVector> object:" << std::endl;
    // for (std::size_t i = 0; i < features.numberOfElements(); ++i)
    // {
    //     std::cout << "Data point " << i + 1 << ": ";
    //     for (std::size_t j = 0; j < features.element(i).size(); ++j)
    //     {
    //         std::cout << features.element(i)[j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Load labels from CSV file (assuming you have a CSV file with labels)
    Data<unsigned int> labels;
    importCSV(labels, "iris_labels.csv");
    // std::cout << labels.numberOfElements() << " labels" << std::endl;

    // Load test data from CSV file (assuming you have a CSV file with test features)
    Data<RealVector> testData;
    importCSV(testData, "iris_test.csv");

    size_t numFeatures = 4; // Number of input features
    size_t numClasses = 2; // Number of classes (binary classification in this case)

    // Create a logistic regression model
    // LinearClassifier<RealVector> logisticModel(numFeatures, numClasses ); // numClasses - 1 for binary classification
    // LogisticRegression<> logisticModel(0, 0, true, 1.e-8);
    LinearClassifier<> logisticModel;
    // Create a training dataset from the loaded data
    ClassificationDataset trainingData(data, labels);

    // Train the logistic regression model using the training data
    LogisticRegression<RealVector> logisticTrainer; // Perform 100 iterations
    logisticTrainer.train(logisticModel, trainingData);

    // Evaluate the model on the test data
    Data<unsigned int> predictedLabels = logisticModel(testData);

    // Calculate the number of test data points
    size_t numTestDataPoints = testData.numberOfElements();

    // Load test labels from CSV file (assuming you have a CSV file with test labels)
    Data<unsigned int> testLabels;
    importCSV(testLabels, "iris_test_labels.csv");

    // Calculate the accuracy of the model on the test data
    size_t correctPredictions = 0;
    for (size_t i = 0; i < numTestDataPoints; ++i) {
        if (predictedLabels.element(i) == testLabels.element(i))
            correctPredictions++;
    }

    double accuracy = static_cast<double>(correctPredictions) / numTestDataPoints;
    std::cout << "Accuracy on test data: " << accuracy << std::endl;

    return 0;
}
