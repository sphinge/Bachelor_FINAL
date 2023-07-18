#include <iostream>
#include <shark/Data/Csv.h>
#include <shark/Algorithms/Trainers/LDA.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

using namespace shark;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_csv_file>" << std::endl;
        return 1;
    }

    // Create a dataset
    ClassificationDataset data;
    try {
        importCSV(data, argv[1], LAST_COLUMN, ' ');
    } catch (...) {
        std::cerr << "Unable to read data from file " <<  argv[1] << std::endl;
        return 1;
    }

    // Split the data into training and testing sets (80% - training, 20% - testing)
    ClassificationDataset test = splitAtElement(data, static_cast<std::size_t>(0.8 * data.numberOfElements()));

    // Create a linear classifier and LDA trainer
    LinearClassifier<> classifier;
    LDA lda;

    // Train the model
    lda.train(classifier, data);

    // Evaluate the model using the Zero-One Loss function
    ZeroOneLoss<> loss;
    Data<unsigned int> predictions = classifier(test.inputs());
    double error = loss(test.labels(), predictions);

    // Print the results
    std::cout << "RESULTS:" << std::endl;
    std::cout << "========" << std::endl;
    std::cout << "test data size: " << test.numberOfElements() << std::endl;
    std::cout << "error rate: " << error << std::endl;

    return 0;
}
