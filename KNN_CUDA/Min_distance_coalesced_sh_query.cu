#include "KNN_CUDA.cuh"

/*
========================== min_distance_coalesced_sh: TRANSOPNSE DATASET   ==========================
													  QUERIES IN SHARED MEMORY
 i: vector's element
 j: dataset's vector
 dataset[i + data_size * j] is in dataset[j + i * dataset_size]

 ====================================================================================================
 */

__global__ void min_distance_coalesced_sh(float* dataset, float* query, float* results, int dataset_size, int data_size, int query_size) {

	const unsigned TILE = 25;
	int iteration = query_size / TILE;
	__shared__ float sh_query[TILE][DATA_SIZE];

	int row = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0;
	float v;

	if (row < dataset_size) {
		for (int phase = 0; phase < iteration; ++phase) {
			if (threadIdx.x < TILE)
				for (int j = 0; j < data_size; ++j)
					sh_query[threadIdx.x][j] = query[j + data_size * (threadIdx.x + TILE * phase)];

			for (int j = 0; j < TILE; ++j) {
				d = 0;
				for (int i = 0; i < data_size; ++i) {
					v = dataset[row + i * dataset_size] - sh_query[j][i];
					d += v * v;
				}
				results[row + (j + TILE * phase) * dataset_size] = d;
			}
			__syncthreads();
		}
	}
}

void min_dinstance_coalesced_shQ_cuda(float* dataset, float* query, float* result, int* result_index, int dataset_size, int data_size, int query_size, int block_size) {

	cout << "=================== MIN DISTANCE COALESCED SH QUERY ====================" << endl << endl;

	double startGPU = omp_get_wtime();
	min_distance_coalesced_sh << <ceil(float(dataset_size) / block_size), block_size >> > (dataset, query, result, dataset_size, data_size, query_size);
	cudaDeviceSynchronize();
	insertion_sort_cuda(result, result_index, dataset_size, query_size);
	cudaDeviceSynchronize();

	printf_s("TIME: %.16g\n\n", omp_get_wtime() - startGPU);

}
