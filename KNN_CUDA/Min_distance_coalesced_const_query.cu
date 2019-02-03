#include "KNN_CUDA.cuh"

/*
========================== min_distance_coalesced_constant_memory: TRANSOPNSE DATASET   ==========================
																   QUERIES IN CONSTANT MEMORY
 i: vector's element
 j: dataset's vector
 dataset[i + data_size * j] is in dataset[j + i * dataset_size]

 =================================================================================================================
 */

__constant__ float constant_query[100 * DATA_SIZE];

__global__ void min_distance_coalesced_const(float* dataset, float* results, int dataset_size, int data_size, int query_size) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0;
	float v;

	if (row < dataset_size) {
		for (int j = 0; j < query_size; ++j) {
			for (int i = 0; i < data_size; ++i) {
				v = dataset[row + i * dataset_size] - constant_query[i + data_size * j];
				d += v * v;
			}
			results[row + j * dataset_size] = d;
			d = 0;
			__syncthreads();
		}
	}
}

void min_dinstance_coalesced_constQ_cuda(float* dataset, vector<float> query, thrust::device_vector<float> result, int* result_index, int dataset_size, int data_size, int query_size, int block_size) {

	cout << "================== MIN DISTANCE COALESCED CONST QUERY ==================" << endl << endl;

	double startGPU;
	float* constant_query_ptr;

	int phase = ceil(query_size / 100);
	int query_size_tmp = 0;
	float* result_ptr;

	startGPU = omp_get_wtime();
	for (int i = 0; i < phase; i++) {
		if (query_size_tmp + 100 <= query_size)
			query_size_tmp += 100;
		else
			query_size_tmp += query_size% 100;

		result_ptr = thrust::raw_pointer_cast(&result[i * 100 * dataset_size]);
		constant_query_ptr = &query[i * 100 * data_size];
		cudaMemcpyToSymbol(constant_query, constant_query_ptr, 100 * data_size * sizeof(float));
		min_distance_coalesced_const<< <ceil(float(dataset_size) / block_size), block_size >> > (dataset, result_ptr, dataset_size, data_size, query_size);
		cudaDeviceSynchronize();
	}
	result_ptr = thrust::raw_pointer_cast(&result[0]);
	insertion_sort_cuda(result_ptr, result_index, dataset_size, query_size);
	cudaDeviceSynchronize();
	printf_s("TIME: %.16g\n\n", omp_get_wtime() - startGPU);
}