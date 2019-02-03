#include "KNN_CUDA.cuh"

__constant__ float constant_query[100 * DATA_SIZE];

__global__ void min_distance_coalesced_sh_query_constant(float* dataset, float* results, int dataset_size, int data_size, int query_size) {

	const unsigned TILE = query_size;
	const unsigned Q = 100;
	int iteration = DATA_SIZE;

	__shared__ float sh_dataset[Q];
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	float d[Q];
	float v;

	if (row < dataset_size) {
		for (int t = 0; t < query_size; t++)
			d[t] = 0;
		for (int phase = 0; phase < iteration; ++phase) {
			sh_dataset[threadIdx.x] = dataset[row + dataset_size * phase];
			__syncthreads();

			for (int i = 0; i < TILE; ++i) {
				v = sh_dataset[threadIdx.x] - constant_query[phase + i * data_size];
				d[i] += v * v;
			}
			__syncthreads();
		}

		for (int i = 0; i < query_size; i++) {
			results[row + i * dataset_size] = d[i];
		}
	}
}

void min_dinstance_coalesced_shDS_constQ_cuda(float* dataset, vector<float> query, thrust::device_vector<float> result, int* result_index, int dataset_size, int data_size, int query_size, int block_size) {

	cout << "============= MIN DISTANCE COALESCED SH DATASET CONST QUERY ============" << endl << endl;

	dim3 dim_grid_min_dist = ceil((float)dataset_size / 100);
	dim3 dim_block_min_dist = 100;

	double startGPU;
	float* constant_query_ptr;
	int phase = ceil(query_size / 100);
	int query_size_tmp = 0;
	float* result_ptr;
	int b;
	startGPU = omp_get_wtime();

	for (int i = 0; i < phase; i++) {
		if (query_size_tmp + 100 <= query_size)
			query_size_tmp += 100;
		else {
			query_size_tmp += query_size % 100;
			b = query_size_tmp;
			dim_grid_min_dist = ceil((float)dataset_size / b);
			dim_block_min_dist = b;
		}

		result_ptr = thrust::raw_pointer_cast(&result[i * 100 * dataset_size]);
		constant_query_ptr = &query[i * 100 * data_size];
		cudaMemcpyToSymbol(constant_query, constant_query_ptr, 100 * data_size * sizeof(float));
		min_distance_coalesced_sh_query_constant << < dim_grid_min_dist, dim_block_min_dist >> > (dataset, result_ptr, dataset_size, data_size, query_size_tmp);
		cudaDeviceSynchronize();
	}
	result_ptr = thrust::raw_pointer_cast(&result[0]);
	insertion_sort_cuda(result_ptr, result_index, dataset_size, query_size);
	cudaDeviceSynchronize();
	printf_s("TIME: %.16g\n\n", omp_get_wtime() - startGPU);
}