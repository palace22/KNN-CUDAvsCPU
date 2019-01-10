#include "Dataset.h"

vector<float> Dataset::generateInMemoryInLine(float mean, float variance) {

	default_random_engine generator;
	normal_distribution <float> distribution(mean, variance);

	vector<float> dataset;
	dataset.resize(this->dataset_size*this->data_size);

	double start = omp_get_wtime();
	for (int i = 0; i < this->dataset_size*this->data_size; i++)
			dataset[i] = distribution(generator);
	printf_s("GenerateInMemory Dataset = %.16g", omp_get_wtime() - start);
	cout << endl;

	return dataset;
}

vector<float> Dataset::readInLineDataset(bool COALESCED) {

	ifstream inFile;
	inFile.open(this->name.c_str());

	vector<float> dataset;
	dataset.resize(this->dataset_size * this->data_size);

	cout << endl << "######### ... READING ... #########" << endl;
	if (inFile.is_open())
	{
		double start = omp_get_wtime();
		if(!COALESCED)
			for (int a = 0; a < this->dataset_size * this->data_size; a++)
				inFile >> dataset[a];
		else {
			for (int a = 0; a < this->data_size; a++) {
				for (int b = 0; b < this->dataset_size; a++) {
					inFile >> dataset[b*dataset_size + a];
				}
			}
		}
		printf_s("Read Dataset = %.16g", omp_get_wtime() - start);
		cout << endl;
		inFile.close();
	}
	else {
		std::cout << "Can't find input file " << endl;
	}
	return dataset;
}

vector<float> Dataset::readInLine(vector<vector<float>> v) {

	int data_size = v[0].size();
	int vector_size = v.size();
	vector<float> inLine(data_size*vector_size);

	cout << endl << "######### ... READING IN LINE ... #########" << endl;
	
	for (int i = 0; i < vector_size; i++) {
		for (int j = 0; j < data_size; j++) {
			inLine[i*data_size + j] = v[i][j];
		}
	}
	cout << endl;

	return inLine;
}

vector<int> Dataset::readInLine(vector<vector<int>> v) {

	int data_size = v[0].size();
	int vector_size = v.size();
	vector<int> inLine(data_size*vector_size);

	cout << endl << "######### ... READING IN LINE ... #########" << endl;

	for (int i = 0; i < vector_size; i++) {
		for (int j = 0; j < data_size; j++) {
			inLine[i*data_size + j] = v[i][j];
		}
	}
	cout << endl;

	return inLine;
}

vector<vector<float>> Dataset::generateCPUCUDAdataset( vector<vector<float>> tmpDataset) {

	int dataset_query_size = tmpDataset.size();
	int query_size = tmpDataset[0].size();

	vector< vector<float>> dataset(2);

	dataset[0].resize(dataset_query_size*query_size);
	dataset[1].resize(dataset_query_size*query_size);

	for (int i = 0; i < dataset_query_size; i++) 
		for (int j = 0; j < query_size; j++) 
			dataset[0][i*query_size + j] = tmpDataset[i][j];
		
		//out << "i: " << i << endl;
	
	for (int i = 0; i < dataset_query_size; i++) 
		for (int j = 0; j < query_size; j++) 
			dataset[1][j*dataset_query_size + i] = tmpDataset[i][j];
	
	return dataset;
}

/*
[ 1 2 3]
[ 4 5 6]
[ 7 8 9]
[ 1 1 1]

[ 1 4 7 1 ][ 2 5 8 1 ]...
*/

Dataset::~Dataset()
{
}
