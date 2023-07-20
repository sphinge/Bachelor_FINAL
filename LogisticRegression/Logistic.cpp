#include <iostream>
#include <shark/Data/Csv.h>
#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/GradientDescent/LBFGS.h>
#include <shark/Algorithms/Trainers/LogisticRegression.h> // Assuming the provided code is in this header file

using namespace shark;

int main() {
    // Create a synthetic classification dataset
    ClassificationDataset dataset;
    dataset.shuffle();
    size_t numFeatures = 4; // Number of input features
    size_t numClasses = 2;  // Number of classes (binary classification in this case)
    
    // Load data from CSV file (assuming you have a CSV file with features and labels)
    Data<RealVector> data;
    Data<unsigned int> labels;
    importCSV(data, "iris.csv");
    importCSV(labels, "iris_labels.csv");
    
    // Split the data into training and test sets
    size_t numDataPoints = data.numberOfElements();
    size_t trainSize = 0.8 * numDataPoints; // 80% for training, 20% for testing
    size_t testSize = numDataPoints - trainSize;
    
    // Split the data and labels into training and test sets
    ClassificationDataset trainingData = createLabeledDataFromRange(data.elements().begin(), data.elements().begin() + trainSize, labels.elements().begin(), numClasses);
    ClassificationDataset testData = createLabeledDataFromRange(data.elements().begin() + trainSize, data.elements().end(), labels.elements().begin() + trainSize, numClasses);
    
    // Create a logistic regression model
    LinearClassifier<RealVector> logisticModel(numFeatures, numClasses);
    LogisticRegression<RealVector> logisticTrainer;
    
    // Train the logistic regression model using the training data
    logisticTrainer.train(logisticModel, trainingData);
    
    // Evaluate the model on the test data
    Data<RealVector> testInputs = testData.inputs();
    Data<unsigned int> testLabels = testData.labels();
    Data<unsigned int> predictedLabels = logisticModel(testInputs);

    // Calculate the accuracy of the model on the test data
    size_t correctPredictions = 0;
    for (size_t i = 0; i < testSize; ++i) {
        if (predictedLabels.element(i) == testLabels.element(i))
            correctPredictions++;
    }

    double accuracy = static_cast<double>(correctPredictions) / testSize;
    std::cout << "Accuracy on test data: " << accuracy << std::endl;

    return 0;
}