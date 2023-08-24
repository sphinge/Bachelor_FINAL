#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// Define a simple struct to hold Iris data
struct IrisSample {
    double sepal_length;
    double sepal_width;
    double petal_length;
    double petal_width;
    int label;
};

// Function to apply noise augmentation
IrisSample applyNoiseAugmentation(const IrisSample& original) {
    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> distribution(0.0, 0.05); // Mean 0, Standard Deviation 0.05
    
    IrisSample augmented = original;
    augmented.sepal_length += distribution(gen);
    augmented.sepal_width += distribution(gen);
    augmented.petal_length += distribution(gen);
    augmented.petal_width += distribution(gen);
    
    return augmented;
}

int main() {
    std::ifstream file("iris.csv");
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }
    
    std::string line;
    std::vector<IrisSample> originalData;
    
    // Read and parse the CSV file
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        IrisSample sample;
        char comma;
        iss >> sample.sepal_length >> comma
            >> sample.sepal_width >> comma
            >> sample.petal_length >> comma
            >> sample.petal_width >> comma
            >> sample.label;
        originalData.push_back(sample);
    }
    
    file.close();
    
    // Apply data augmentation
    std::vector<IrisSample> augmentedData;
    for (const IrisSample& original : originalData) {
        augmentedData.push_back(original);
        for (int i = 0; i < 5; ++i) { // Augment with 5 noisy samples
            IrisSample augmented = applyNoiseAugmentation(original);
            augmentedData.push_back(augmented);
        }
    }
    
    // Shuffle the augmented data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(augmentedData.begin(), augmentedData.end(), gen);
    
    // Save both the original and augmented data to a new CSV file
    std::ofstream output("iris_augmented.csv");
    if (!output.is_open()) {
        std::cerr << "Failed to create output file." << std::endl;
        return 1;
    }
    
    // Save the original data
    for (const IrisSample& sample : originalData) {
        output << sample.sepal_length << ","
               << sample.sepal_width << ","
               << sample.petal_length << ","
               << sample.petal_width << ","
               << sample.label << "\n";
    }
    
    // Save the augmented data
    for (const IrisSample& sample : augmentedData) {
        output << sample.sepal_length << ","
               << sample.sepal_width << ","
               << sample.petal_length << ","
               << sample.petal_width << ","
               << sample.label << "\n";
    }
    
    output.close();
    
    return 0;
}
