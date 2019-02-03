#include "KNN_CPU.h"

float KNN_CPU::dinstance(float *a, float *b) {
	float d = 0;
	for (int i = 0; i < data_size; i++)
		d += (a[i] - b[i]) * (a[i] - b[i]);
	
	return d;
}

vector< pair< float, int >> KNN_CPU::calculate(float* query, float** dataset, int k ) {
	vector< pair< float, int >> dd;
	dd.resize(dataset_size);
	float* d = new float[dataset_size];
	vector< pair< float, int >> kMin;
	kMin.resize(k);
	float dTmp = 0;

	for (int i = 0; i < dataset_size; i++) 
		dd[i] = make_pair(dinstance( query, dataset[i]), i);
	
	sort(dd.begin(), dd.end());

	for (int i = 0; i < k; i++) {
		kMin[i].first = dd[i].first;
		kMin[i].second = dd[i].second;
	}

	return kMin;

}

vector< pair< float, int >> KNN_CPU::calculateInLine(float* query, vector<float> dataset, int k) {
	vector< pair< float, int >> dd;
	dd.resize(dataset_size);
	dataset.resize(dataset_size*data_size);
	float* d = new float[dataset_size];
	vector< pair< float, int >> kMin;
	kMin.resize(k);
	float dTmp = 0;

	for (int i = 0; i < dataset_size; i++) 
		dd[i] = make_pair(dinstance(query, &dataset[i*data_size]), i);

	sort(dd.begin(), dd.end());

	for (int i = 0; i < k; i++) {
		kMin[i].first = dd[i].first;
		kMin[i].second = dd[i].second;
	}

	return kMin;

}

vector< pair< float, int >> KNN_CPU::minDinstanceCPU(vector<float> dataset, vector<float> query, int k) {
	
	int querySize = query.size() / data_size;
	vector< pair< float, int >> dd;
	dd.resize(dataset_size*querySize);
	dataset.resize(dataset_size*data_size);

	float* d = new float[dataset_size];
	vector< pair< float, int >> kMin;
	kMin.resize(k);

	float dTmp = 0;

	vector< pair< float, int >>::iterator it = dd.begin();

	for (int q = 0; q < querySize; q++) {
		for (int i = 0; i < dataset_size; i++) 
			dd[i + q*dataset_size] = make_pair(dinstance(&query[q*data_size], &dataset[i*data_size]), i%dataset_size);

		sort( it+q*dataset_size, it+(q+1)*dataset_size);
	}
	return dd;

}

void KNN_CPU::print_result(vector< pair< float, int >> result, vector<int> groundtruth_dataset, int query_size, int k, int k_groudtruth, bool print) {
	if (print) {
		for (int q = 0; q < query_size; q++) {
			cout << "############# QUERY " << q << "#############" << endl;
			for (int i = 0; i < k; i++) {
				cout << "  MY INDEX : " << result[i + q * dataset_size].second << endl;
				cout << "GROUNDTRUTH: " << groundtruth_dataset[i + q * k_groudtruth] << endl << endl;
			}
		}
	}
}


KNN_CPU::~KNN_CPU()
{}
