#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "Dataset.h"
#include "utils.h"
#include <fstream>
#include <string>
#include <omp.h>
#include "KNN.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/advance.h>
#include <thrust/device_vector.h>

using namespace std;

#define VERBOSE 1// 0: print only time; 1: print comparison result index: GROUNDTRUTH - RESULT; 
#define DATA_SIZE 128
#define k 100 // Number of firsts min index

__global__ void minDistance(float* query, float* dataset, float* results, int* results_index, int datasetSize, int querySize) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	float d = 0;

	if (row < datasetSize) {
		/*
		for (int i = 0; i < DATA_SIZE; i++)
			p[i] = dataset[row * DATA_SIZE + i];

		__syncthreads();*/
		float v;
		if (col < querySize) {
			for (int i = 0; i < DATA_SIZE; i++) {
				v = dataset[row * DATA_SIZE + i] - query[i + col * DATA_SIZE];
				d += v * v;
			}
			results[row + col * datasetSize] = d;
			results_index[row + col * datasetSize] = row;
		}
		
	}
}

__global__ void minDistanceCoalesced(float* query, float* dataset, float* results, int* results_index, int datasetSize, int querySize) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	float p[DATA_SIZE];
	float d = 0;

	if (row < datasetSize) {
		for (int i = 0; i < DATA_SIZE; i++)
			p[i] = dataset[row + i * datasetSize];

		__syncthreads();
		
		float v;
		if (col < querySize){
			for (int i = 0; i < DATA_SIZE; i++) {
				//v = p[i] - query[i + col * DATA_SIZE];
				v = p[i] - query[i + col * DATA_SIZE];
				d += v * v;
			}
			results[row + col * datasetSize] = d;
			results_index[row + col * datasetSize] = row;
		}
	}
}

int main()
{
	Dataset* d = new Dataset();
	
	// ##########  ANN_SIFT10K  ##########
	
	vector<vector<float>> dataset = pqtable::ReadTopN("siftsmall/siftsmall_base.fvecs", "fvecs");
	vector<vector<float>> _queryDataset = pqtable::ReadTopN("siftsmall/siftsmall_query.fvecs", "fvecs");
	vector<vector<int>> _groundtruthDataset = pqtable::ReadTopNI("siftsmall/siftsmall_groundtruth.ivecs", "ivecs");
	
	// ##########  ANN_SIFT1M  ##########
	/*
	vector<vector<float>> dataset = pqtable::ReadTopN("sift/sift_base.fvecs", "fvecs");
	vector<vector<float>> _queryDataset = pqtable::ReadTopN("sift/sift_query.fvecs", "fvecs");
	vector<vector<int>> _groundtruthDataset = pqtable::ReadTopNI("sift/sift_groundtruth.ivecs", "ivecs");
	*/

	int datasetSize = dataset.size();
	int querySize = _queryDataset.size();
	//int querySize = 10; // for 1M dataset
	int dataSize = dataset[0].size();
	int kNumber = _groundtruthDataset[0].size();

	cout << "DATASET SIZE: " << dataset.size() << endl;
	cout << "ELEMENT SIZE: " << dataset[0].size() << endl << endl;
	cout << "QUERY: " << querySize << endl;
	cout << "K:" << kNumber;
	
	vector<vector<float>> tmpDataset = d->generateCPUCUDAdataset(dataset);
	vector<float> queryDataset = d->readInLine(_queryDataset);
	vector<int> groundtruthDataset = d->readInLine(_groundtruthDataset);
	vector<float> cpuDataset = tmpDataset[0];
	vector<float> cudaDataset = tmpDataset[1]; // For minDinstanceCoalesced( ... )

	cout << endl << "######### INITIALIZE CUDA #########" << endl;

	thrust::device_vector<float> deviceQueryDataset = queryDataset;
	thrust::device_vector<float> deviceResult(datasetSize*querySize);
	thrust::device_vector<int> deviceResultIndex(datasetSize*querySize);
	thrust::device_vector<float> deviceCudaDataset = cudaDataset;
	thrust::device_vector<float> deviceCudaDataset2 = cpuDataset;

	float* ds_ptr = thrust::raw_pointer_cast(&deviceCudaDataset[0]);
	float* query_ptr = thrust::raw_pointer_cast(&deviceQueryDataset[0]);
	float* result_ptr = thrust::raw_pointer_cast(&deviceResult[0]);
	int* result_index_ptr = thrust::raw_pointer_cast(&deviceResultIndex[0]);

	dim3 dimGrid(ceil((float)datasetSize / dataSize), ceil((float)querySize/8));
	dim3 dimBlock(DATA_SIZE, 8);

	cout << endl << "######### KNN GPU #########" << endl;
	double startGPU = omp_get_wtime();
	minDistanceCoalesced <<< dimGrid, dimBlock >>> (query_ptr, ds_ptr, result_ptr, result_index_ptr, datasetSize, querySize);
	cudaDeviceSynchronize();
	printf_s("TIME GPU COALESCED= %.16g", omp_get_wtime() - startGPU);
	cout << endl;

	thrust::device_vector<float>::iterator deviceResultIt = deviceResult.begin();
	thrust::device_vector<float>::iterator deviceResultEndIt = deviceResult.begin();
	thrust::advance(deviceResultEndIt, datasetSize);
	thrust::device_vector<int>::iterator deviceResultIndexIt = deviceResultIndex.begin();
	
	for (int i = 0; i < querySize; i++) {
		thrust::sort_by_key(deviceResultIt, deviceResultEndIt, deviceResultIndexIt);
		thrust::advance(deviceResultIt, datasetSize);
		thrust::advance(deviceResultEndIt, datasetSize);
		thrust::advance(deviceResultIndexIt, datasetSize);
	}
	printf_s("TIME GPU COALOSCED SORT= %.16g", omp_get_wtime() - startGPU);
	cout << endl;

	float* ds_ptr2 = thrust::raw_pointer_cast(&deviceCudaDataset2[0]);
	query_ptr = thrust::raw_pointer_cast(&deviceQueryDataset[0]);
	result_ptr = thrust::raw_pointer_cast(&deviceResult[0]);
	result_index_ptr = thrust::raw_pointer_cast(&deviceResultIndex[0]);

	startGPU = omp_get_wtime();
	minDistance << < dimGrid, dimBlock >> > (query_ptr, ds_ptr2, result_ptr, result_index_ptr, datasetSize, querySize);
	cudaDeviceSynchronize();
	printf_s("TIME GPU = %.16g", omp_get_wtime() - startGPU);
	cout << endl;

	deviceResultIt = deviceResult.begin();
	deviceResultEndIt = deviceResult.begin();
	thrust::advance(deviceResultEndIt, datasetSize);
	deviceResultIndexIt = deviceResultIndex.begin();

	for (int i = 0; i < querySize; i++) {
		thrust::sort_by_key(deviceResultIt, deviceResultEndIt, deviceResultIndexIt);
		thrust::advance(deviceResultIt, datasetSize);
		thrust::advance(deviceResultEndIt, datasetSize);
		thrust::advance(deviceResultIndexIt, datasetSize);
	}
	printf_s("TIME GPU SORT= %.16g", omp_get_wtime() - startGPU);
	cout << endl;

	if (VERBOSE) {
		thrust::host_vector<int> resultIndexOut = deviceResultIndex;
		for (int i = 0; i < querySize; i++) {
			cout << "###################### QUERY" << i << " ###################### " << endl;
			for (int j = 0; j < kNumber; j++) {
				//cout << "MY VALUE : " << resultOut[i*datasetSize + j] << endl;
				cout << "MY INDEX : " << resultIndexOut[i*datasetSize + j] << endl;
				cout << "GROUNDTRUTH: " << groundtruthDataset[i*kNumber + j] << endl << endl;
			}
			cout << endl << endl;
		}
	}
	
	//INITIALIZE FOR CPU
	KNN* knnCpu = new KNN(datasetSize, DATA_SIZE);

	cout << endl << "######### KNN CPU #########" << endl;
	double start = omp_get_wtime();
	vector< pair< float, int >> kMin = knnCpu->minDinstanceCPU(cpuDataset, queryDataset, k);
	printf_s("TIME CPU = %.16g", omp_get_wtime() - start);
	cout << endl;
	
	if (VERBOSE) {
		for (int q = 0; q < querySize; q++) {
			cout << "############# QUERY " << q << "#############" << endl;
			for (int i = 0; i < k; i++) {
				cout << "MY INDEX : "<< kMin[i + q * datasetSize].second << endl;
				cout << "GROUNDTRUTH: "<< groundtruthDataset[i + q * k] << endl << endl;
			}
		}
	}	
	cout << endl << "######### FINISH #########" << endl;
	exit(0);
	return 0;
}
