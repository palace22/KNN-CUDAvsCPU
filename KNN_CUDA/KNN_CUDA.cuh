#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define DATA_SIZE 128

using namespace std;


void insertion_sort_cuda(float* result, int* index, int dataset_size, int query_size);
void print_result(thrust::host_vector<int> result_index_out, vector<int> groundtruth_dataset, int dataset_size, int query_size, int k, bool p);

void min_dinstance_cuda(float* dataset, float* query, float* result, int* result_index, int dataset_size, int data_size, int query_size, int block_size);
void min_dinstance_coalesced_cuda(float* dataset, float* query, float* result, int* result_index, int dataset_size, int data_size, int query_size, int block_size);
void min_dinstance_coalesced_shQ_cuda(float* dataset, float* query, float* result, int* result_index, int dataset_size, int data_size, int query_size, int block_size);
void min_dinstance_coalesced_constQ_cuda(float* dataset, vector<float> query, thrust::device_vector<float> result, int* result_index, int dataset_size, int data_size, int query_size, int block_size);
void min_dinstance_coalesced_shDS_constQ_cuda(float* dataset, vector<float> query, thrust::device_vector<float> result, int* result_index, int dataset_size, int data_size, int query_size, int block_size);



