#pragma once 
#include <math.h>
#include <algorithm>
#include <map>
#include "Dataset.h"

class KNN
{
public:
	KNN(int dataset_size, int data_size) :  data_size(data_size), dataset_size(dataset_size) {};
	~KNN();

	vector< pair< float, int >> calculate(float* query, float** dataset, int k = 1);
	vector< pair< float, int >> calculateInLine(float* query, vector<float> dataset, int k = 1);
	vector< pair< float, int >> minDinstanceCPU(vector<float> dataset, vector<float> query, int k);

private:
	float dinstance(float *a, float *b);
	float dinstanceInLine(float *a, float *b);

	int data_size;
	int dataset_size;
	int k;
	vector<float> dataset;
};

