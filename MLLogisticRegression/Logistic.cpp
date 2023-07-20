#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <ensmallen.hpp>
#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack;

int main()
{
    // Load training data and labels.
    arma::mat predictors; // Training data (features)
    arma::Row<size_t> responses; // Labels

    data::Load("data.csv", predictors, true);
    data::Load("labels.csv", responses, true);

    // Set the number of classes in the dataset.
    size_t numClasses = responses.max() + 1;

    // Create and train the softmax regression model.
    double lambda = 0.0001; // L2-regularization parameter
    SoftmaxRegression<> sr(predictors, responses, numClasses, lambda);

    // Load test data (if needed).
    arma::mat testDataset; // Test data
    data::Load("test.csv", testDataset, true);

    // Classify new data points (if needed).
    arma::Row<size_t> labels; // Predicted labels for each test point
    sr.Classify(testDataset, labels);

    // Compute Accuracy (Optional - if you have ground truth labels for the test data).
    // double accuracy = sr.ComputeAccuracy(testDataset, groundTruthLabels);

    // Print the predicted labels for the test data.
    for (size_t i = 0; i < labels.n_elem; ++i)
    {
        std::cout << "Predicted label for test point " << i << ": " << labels[i] << std::endl;
    }

    return 0;
}
