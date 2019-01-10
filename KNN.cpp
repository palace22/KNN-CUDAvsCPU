#include "KNN.h"

float KNN::dinstance(float *a, float *b) {
	float d = 0;
	for (int i = 0; i < data_size; i++) {
		d += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return d;
}

float KNN::dinstanceInLine(float *a, float *b) {
	float d = 0;
	for (int i = 0; i < data_size; i++) {
		d += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return d;
}


vector< pair< float, int >> KNN::calculate(float* query, float** dataset, int k ) {
	vector< pair< float, int >> dd;
	dd.resize(dataset_size);
	float* d = new float[dataset_size];
	vector< pair< float, int >> kMin;
	kMin.resize(k);
	float dTmp = 0;

	for (int i = 0; i < dataset_size; i++) {
		dd[i] = make_pair(dinstance( query, dataset[i]), i);
	}
	
	sort(dd.begin(), dd.end());

	for (int i = 0; i < k; i++) {
		kMin[i].first = dd[i].first;
		kMin[i].second = dd[i].second;
	}


	return kMin;

}

vector< pair< float, int >> KNN::calculateInLine(float* query, vector<float> dataset, int k) {
	vector< pair< float, int >> dd;
	dd.resize(dataset_size);
	dataset.resize(dataset_size*data_size);
	float* d = new float[dataset_size];
	vector< pair< float, int >> kMin;
	kMin.resize(k);
	float dTmp = 0;
	//for (int i = 0; i < k; i++)
		//d[i] = 99999999999.9;

	for (int i = 0; i < dataset_size; i++) {
		dd[i] = make_pair(dinstance(query, &dataset[i*data_size]), i);
		//std::cout << "D sqrt: " << d << std::endl;
	}

	sort(dd.begin(), dd.end());

	for (int i = 0; i < k; i++) {
		kMin[i].first = dd[i].first;
		kMin[i].second = dd[i].second;
	}


	return kMin;

}

vector< pair< float, int >> KNN::minDinstanceCPU(vector<float> dataset, vector<float> query, int k) {
	
	int querySize = query.size() / data_size;
	//int querySize = 1;
	vector< pair< float, int >> dd;
	dd.resize(dataset_size*querySize);
	dataset.resize(dataset_size*data_size);

	float* d = new float[dataset_size];
	vector< pair< float, int >> kMin;
	kMin.resize(k);

	float dTmp = 0;
	//for (int i = 0; i < k; i++)
		//d[i] = 99999999999.9;
	vector< pair< float, int >>::iterator it = dd.begin();

	for (int q = 0; q < querySize; q++) {
		for (int i = 0; i < dataset_size; i++) {
			dd[i + q*dataset_size] = make_pair(dinstance(&query[q*data_size], &dataset[i*data_size]), i%dataset_size);
			//std::cout << "D sqrt: " << d << std::endl;
		}
		sort( it+q*dataset_size, it+(q+1)*dataset_size);

	}
	/*
	for (int i = 0; i < k; i++) {
		kMin[i].first = dd[i].first;
		kMin[i].second = dd[i].second;
		//cout << "First:" << kMin[i].first << "Second:" << kMin[i].second << endl;
	}*/

	return dd;

}

KNN::~KNN()
{}
