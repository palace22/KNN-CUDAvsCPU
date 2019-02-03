#include "KNN_CUDA.cuh"

/*
========================== min_distance_coalesced: TRANSOPNSE DATASET   ==========================

 i: vector's element
 j: dataset's vector
 dataset[i + data_size * j] is in dataset[j + i * dataset_size]

 =================================================================================================
 */

__global__ void min_distance_coalesced(float* dataset, float* query, float* results, int dataset_size, int data_size, int query_size) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0;
	float v;

	if (row < dataset_size) {
		for (int query_index = 0; query_index < query_size; ++query_index) {
			for (int element_index = 0; element_index < data_size; ++element_index) {
				v = dataset[row + element_index * dataset_size] - query[element_index + query_index * data_size];
				d += v * v;
			}
			results[row + query_index * dataset_size] = d;
			d = 0;
			__syncthreads();
		}
	}
}

void min_dinstance_coalesced_cuda(float* dataset, float* query, float* result, int* result_index, int dataset_size, int data_size, int query_size, int block_size) {
	
	cout << "======================== MIN DISTANCE COALESCED ========================" << endl << endl;

	double startGPU = omp_get_wtime();
	min_distance_coalesced << <ceil(float(dataset_size) / block_size), block_size >> > (dataset, query, result, dataset_size, data_size, query_size);
	cudaDeviceSynchronize();
	insertion_sort_cuda(result, result_index, dataset_size, query_size);
	cudaDeviceSynchronize();

	printf_s("TIME: %.16g\n\n", omp_get_wtime() - startGPU);

}