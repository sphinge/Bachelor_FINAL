#include <shark/Data/Csv.h>
#include <shark/Algorithms/Trainers/LogisticRegression.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>

#include <iostream>

using namespace shark;
using namespace std;

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "usage: " << argv[0] << " (file with inputs/independent variables) (file with outputs/dependent variables)" << endl;
        exit(EXIT_FAILURE);
    }

    LabeledData<RealVector, unsigned int> data;
    try {
        importCSV(data, argv[1], std::string(" "));
    } catch (...) {
        cerr << "unable to read input data from file " << argv[1] << endl;
        exit(EXIT_FAILURE);
    }

    for (std::size_t i = 0; i < data.numberOfElements(); ++i) {
        data.label(i) = data.label(i) > 0 ? 1 : 0; // Convert class labels to binary values (0 and 1)
    }

    // trainer and model
    LogisticRegression<> trainer;
    LinearClassifier<RealVector> model;

    // train model
    trainer.train(model, data);

    // show model parameters
    cout << "intercept: " << model.offset() << endl;
    cout << "weights: " << model.weights() << endl;

    CrossEntropy<unsigned int, RealVector> loss;
    Data<RealVector> prediction = model(data.inputs());
    cout << "cross-entropy loss: " << loss(data.labels(), prediction) << endl;
}
