#include "KNN_CUDA.cuh"

__global__ void cu_insertion_sort(float *dist, int *ind, int dataset_size, int query_size, int k) {

	// Variables
	int l, i, j;
	float *p_dist;
	int   *p_ind;
	float curr_dist, max_dist;
	int   curr_row, max_row;
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (xIndex < query_size) {
		// Pointer shift, initialization, and max value
		p_dist = dist + xIndex * dataset_size;
		p_ind = ind + xIndex * dataset_size;
		max_dist = p_dist[0];
		p_ind[0] = 1;

		// Part 1 : sort kth firt elementZ
		for (l = 1; l < k; l++) {
			curr_row = l;
			curr_dist = p_dist[curr_row];
			if (curr_dist < max_dist) {
				i = l - 1;
				for (int a = 0; a < l - 1; a++) {
					if (p_dist[a] > curr_dist) {
						i = a;
						break;
					}
				}
				for (j = l; j > i; j--) {
					p_dist[j] = p_dist[j - 1];
					p_ind[j] = p_ind[j - 1];
				}
				p_dist[i] = curr_dist;
				p_ind[i] = l + 1;
			}
			else {
				p_ind[l] = l + 1;
			}
			max_dist = p_dist[curr_row];
		}

		// Part 2 : insert element in the k-th first lines
		max_row = k - 1;
		for (l = k; l < dataset_size; l++) {
			curr_dist = p_dist[l];
			if (curr_dist < max_dist) {
				i = k - 1;
				for (int a = 0; a < k - 1; a++) {
					if (p_dist[a] > curr_dist) {
						i = a;
						break;
					}
				}
				for (j = k - 1; j > i; j--) {
					p_dist[j] = p_dist[(j - 1)];
					p_ind[j] = p_ind[(j - 1)];
				}
				p_dist[i] = curr_dist;
				p_ind[i] = l;
				max_dist = p_dist[max_row];
			}
		}
	}
}

void insertion_sort_cuda(float* result, int* index, int dataset_size, int query_size) {

	dim3 dim_grid_sort(1);
	dim3 dim_block_sort(query_size);

	cu_insertion_sort << < dim_grid_sort, dim_block_sort >> > (result, index, dataset_size, query_size, 50);
	cudaDeviceSynchronize();

}

void print_result(thrust::host_vector<int> device_result_index, vector<int> groundtruth_dataset, int dataset_size, int query_size, int k, bool p) {
	thrust::host_vector<int> host_result_index = device_result_index;

	if (p) {
		for (int i = 0; i < query_size; i++) {
			cout << "###################### QUERY" << i << " ###################### " << endl;
			for (int j = 0; j < k; j++) {
				cout << " MY INDEX  : " << host_result_index[j + i * dataset_size] << endl;
				cout << "GROUNDTRUTH: " << groundtruth_dataset[j + i * 100] << endl << endl;
			}
			cout << endl << endl;
		}
	}
}

