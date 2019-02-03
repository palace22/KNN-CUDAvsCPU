#include "KNN_CUDA.cuh"

/*
========================== min_distance: NAIVE SOLUTION ==========================
*/

__global__ void min_distance(float* dataset, float* query, float* results, int dataset_size, int data_size, int query_size) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0;
	float v;

	if (row < dataset_size) {
		for (int i = 0; i < query_size; ++i) {
			for (int j = 0; j < data_size; ++j) {
				v = dataset[j + row * data_size] - query[j + i * data_size];
				d += v * v;
			}
			results[row + i * dataset_size] = d;
			d = 0;
			__syncthreads();
		}
	}
}

void min_dinstance_cuda(float* dataset, float* query, float* result, int* result_index, int dataset_size, int data_size, int query_size, int block_size) {


	cout << "========================== MIN DISTANCE NAIVE ==========================" << endl << endl;

	double startGPU = omp_get_wtime();
	min_distance <<<ceil(float(dataset_size) / block_size), block_size>> > (dataset, query, result, dataset_size, data_size, query_size);
	cudaDeviceSynchronize();
	insertion_sort_cuda(result, result_index, dataset_size, query_size );
	printf_s("TIME: %.16g\n\n", omp_get_wtime() - startGPU);



}