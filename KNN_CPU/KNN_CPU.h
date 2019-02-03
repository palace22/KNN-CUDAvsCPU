#pragma once 
#include <math.h>
#include <algorithm>
#include <map>
#include "../Dataset.h"

class KNN_CPU
{
public:
	KNN_CPU(int dataset_size, int data_size) :  data_size(data_size), dataset_size(dataset_size) {};
	~KNN_CPU();

	vector< pair< float, int >> calculate(float* query, float** dataset, int k = 1);
	vector< pair< float, int >> calculateInLine(float* query, vector<float> dataset, int k = 1);
	vector< pair< float, int >> minDinstanceCPU(vector<float> dataset, vector<float> query, int k);
	void print_result(vector< pair< float, int >> result, vector<int> groundtruth_dataset, int query_size, int k, int k_groudtruth, bool print);
private:
	float dinstance(float *a, float *b);

	int data_size;
	int dataset_size;
	int k;
	vector<float> dataset;
};

