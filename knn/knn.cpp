#include "knn.h"
#include <map>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <ctime>
#include <limits>


using namespace std;
vector<Data> readCSV(const string& filename) {
    ifstream file(filename);
    vector<Data> data;

    if (!file.is_open()) {
        cout << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        Data row;
        row.fields = new double[4];
        row.size = 4;
        char comma;
        ss >> row.fields[0] >> comma >> row.fields[1] >> comma
           >> row.fields[2] >> comma >> row.fields[3] >> comma >> row.cls;
        data.push_back(row);
    }

    file.close();
    return data;
}

void normalize(std::vector<Data>& data)
{
	double max = std::numeric_limits<double>::max();
	double min = std::numeric_limits<double>::min();
	double *mins = new double[data[0].size];
	double *maxes = new double[data[0].size];
	std::fill_n(mins, data[0].size, max);
	std::fill_n(maxes, data[0].size, min);

	for (auto& row : data) { // using reference to modify the original data
		for (size_t i = 0; i < row.size; i++) {
			if (row.fields[i] > maxes[i])
				maxes[i] = row.fields[i];
			if (row.fields[i] < mins[i])
				mins[i] = row.fields[i];
		}
	}
	for (auto& row : data) { // using reference to modify the original data
		for (size_t i = 0; i < 4; i++) {
			row.fields[i] = (row.fields[i] - mins[i]) / (maxes[i] - mins[i]);
		}
	}
}


std::string Knn::getNeighbours(Data d, std::vector<Data> data_list, size_t k)
{
	double *top_k_distances = new double[k];
	int *top_k_indexes = new int[k];
	std::fill_n(top_k_indexes, k, -1);
	std::fill_n(top_k_distances, k, std::numeric_limits<double>::max()); // Initialize top_k_distances with max double

	for (size_t index = 0; index < data_list.size(); index++)
	{
		double distance = d.distance(data_list[index]);
		size_t ind = 0;
		while (ind < k && top_k_indexes[ind] != -1 && top_k_distances[ind] < distance)
		{
			ind++;
		}
		if (ind < k)
		{
			int shift_index = k - 1;
			while (shift_index > ind)
			{
				top_k_indexes[shift_index] = top_k_indexes[shift_index - 1];
				top_k_distances[shift_index] = top_k_distances[shift_index - 1];
				shift_index--;
			}
			top_k_indexes[ind] = index;
			top_k_distances[ind] = distance;
		}
	}
	 
	size_t index = 0; 
	std::map<std::string, int> top_k_class_map;
	while (index < k && top_k_indexes[index] != -1)
	{
		top_k_class_map[data_list[top_k_indexes[index]].cls]++;
		index++;
	}
	int max = -1;
	std::string maxClass;
	for (auto pair : top_k_class_map)
	{
		if (pair.second > max)
		{
			max = pair.second;
			maxClass = pair.first;
			//std::cout << max << std::endl;
			//std::cout << maxClass << std::endl;
		}
	}

	delete[] top_k_distances;
	delete[] top_k_indexes;

	return maxClass;
}


void run_knn(std::vector<Data> test, std::vector<Data> training, size_t k)
{
	Knn *knn = new Knn();
	size_t correct = 0;
	for (auto test_data : test)
	{
		std::string maxClass = knn->getNeighbours(test_data, training, k);
		cout << "max class: " << maxClass << " test: " << test_data.cls << endl;
		if (maxClass.compare(test_data.cls) == 0)
		{
			correct++;
		}
	}
	std::cout << 
		correct << " of "<< test.size() << 
		" (" << 
		static_cast<double>(100.0) *correct/static_cast<double>(test.size()) <<
		")" << std::endl;
	
}

int main(int argc, char **argv)
{
	auto dataset = readCSV("iris.csv");
	//std::cout << "Test" << std::endl;
	normalize(dataset);
	std::random_shuffle(dataset.begin(), dataset.end());
	auto test_dataset = readCSV("iris_test.csv");
	run_knn(test_dataset, dataset, 10);
	return 0;
}