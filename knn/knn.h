#ifndef KNN_H
#define KNN_H
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

struct Data
{
	double *fields;
	size_t size;

	std::string cls;

	double distance(const Data& d) const
	{
		double total = 0;
		size_t min_size = std::min(size, d.size);
		for (size_t i = 0; i < min_size; i++)
		{
			total += (std::pow(d.fields[i] - fields[i], 2));
		}
		return sqrt(total);
	}

};

class Knn
{
public:
	std::string getNeighbours(Data d, std::vector<Data> data_list, size_t k);

private:
};

#endif