#pragma once 
#include <string>
#include <fstream>
#include <random>
#include <iostream>
#include <vector>
#include <iterator>
#include <omp.h>
#include <sstream>

using namespace std;

class Dataset
{
public:
	Dataset() {};
	vector<float> generateInMemoryInLine(float mean, float variance); //to generate in memory dataset
	vector<float> readInLineDataset(bool COALESCED); //to read .txt file dataset
	vector<int> readInLine(vector<vector<int>> v);
	vector<float> readInLine(vector<vector<float>> v);
	vector<vector<float>> generateCPUCUDAdataset(vector<vector<float>> dataset);
	int getDatasetSize() { return dataset_size; }
	int getDatSize() { return data_size; }
	~Dataset();

private:
	ifstream ds;
	string name = "";
	int dataset_size = 0;
	int data_size = 0;
};

