// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

pair<MatrixXd, MatrixXd> readCSV(const string &filename, int numFeatures, int numLabels);

#endif
