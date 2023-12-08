
#include "WB.pb.h"  // Include the generated protobuf header
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <string>
// g++ main.cpp WB.pb.cc -I/usr/include/eigen3 -lprotobuf -o chk

// Protobuf serialized size: 381 bytes
using namespace Eigen;
using namespace std;

int main() {
    // Initialize weights and biases
    int inputSize = 5;
    int hiddenSize = 10;
    int outputSize = 3;

    MatrixXf W1 = MatrixXf::Random(hiddenSize, inputSize);
    VectorXf b1 = VectorXf::Zero(hiddenSize);
    MatrixXf W2 = MatrixXf::Random(outputSize, hiddenSize);
    VectorXf b2 = VectorXf::Zero(outputSize);

    // Create protobuf message
    WeightsBiases weightsBiasesMessage;
    for (int i = 0; i < W1.size(); ++i) {
        weightsBiasesMessage.add_w1(W1.data()[i]);
    }
    for (int i = 0; i < b1.size(); ++i) {
        weightsBiasesMessage.add_b1(b1.data()[i]);
    }
    for (int i = 0; i < W2.size(); ++i) {
        weightsBiasesMessage.add_w2(W2.data()[i]);
    }
    for (int i = 0; i < b2.size(); ++i) {
        weightsBiasesMessage.add_b2(b2.data()[i]);
    }

    // Serialize to string
    std::string serializedData;
    weightsBiasesMessage.SerializeToString(&serializedData);

    // Write to file
    std::ofstream protobufFile("weights_biases_protobuf.bin", std::ios::binary);
    protobufFile.write(serializedData.c_str(), serializedData.size());
    protobufFile.close();

    std::cout << "Protobuf serialized size: " << serializedData.size() << " bytes" << std::endl;

    return 0;
}
