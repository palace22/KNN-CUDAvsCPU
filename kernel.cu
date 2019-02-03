#include "Dataset.h"
#include "utils.h"
#include "KNN_CUDA/KNN_CUDA.cuh"
#include "KNN_CPU/KNN_CPU.h"


#define k 10
#define QUERY_SIZE 100
#define DATA_SIZE 128

int main()
{
	cout << "========================================================================" << endl;
	cout << "                            KNN: CPU vs CUDA " << endl;
	cout << "========================================================================" << endl << endl;

	// ##########  ANN_SIFT10K  ##########

	string  dataset_name = "ANN_SIFT10K";
	vector<vector<float>> dataset = pqtable::ReadTopN("siftsmall/siftsmall_base.fvecs", "fvecs");
	vector<vector<float>> _query_dataset = pqtable::ReadTopN("siftsmall/siftsmall_query.fvecs", "fvecs");
	vector<vector<int>> _groundtruth_dataset = pqtable::ReadTopNI("siftsmall/siftsmall_groundtruth.ivecs", "ivecs");

	// ##########  ANN_SIFT1M  ##########
	
	//string dataset_name = "ANN_SIFT1M";
	//vector<vector<float>> dataset = pqtable::ReadTopN("sift/sift_base.fvecs", "fvecs");
	//vector<vector<float>> _query_dataset = pqtable::ReadTopN("sift/sift_query.fvecs", "fvecs");
	//vector<vector<int>> _groundtruth_dataset = pqtable::ReadTopNI("sift/sift_groundtruth.ivecs", "ivecs");
	
	
	Dataset* d = new Dataset();
	
	int dataset_size = dataset.size();
	int data_size = dataset[0].size();
	int k_number = _groundtruth_dataset[0].size();
	cout << "========================================================================" << endl;
	cout << "DATASET:      " << dataset_name << endl;
	cout << "DATASET SIZE: " << dataset.size() << endl;
	cout << "DATA SIZE:    " << dataset[0].size() << endl;
	cout << "QUERY:        " << _query_dataset.size() << endl;
	cout << "K:            " << k_number << endl << endl;

	cout << "QUERY USED:   " << QUERY_SIZE << endl;
	cout << "K USED:       " << k << endl;
	cout << "========================================================================" << endl << endl;


	vector<vector<float>> tmp_dataset = d->generateCPUCUDAdataset(dataset);
	vector<float> query_dataset = d->readInLine(_query_dataset);
	vector<int> groundtruth_dataset = d->readInLine(_groundtruth_dataset);
	vector<float> cpu_dataset = tmp_dataset[0];
	vector<float> cuda_dataset = tmp_dataset[1]; // TRANSOPNSE DATASET
	vector<int> result_index(dataset_size*QUERY_SIZE);
	query_dataset.resize(QUERY_SIZE*data_size);

	thrust::device_vector<float> device_query_dataset = query_dataset;
	thrust::device_vector<float> device_result(dataset_size*QUERY_SIZE);
	thrust::device_vector<int> device_result_index = result_index;
	thrust::device_vector<float> device_cuda_dataset = cuda_dataset;
	thrust::device_vector<float> device_cuda_dataset_naive = cpu_dataset;

	float* ds_ptr = thrust::raw_pointer_cast(&device_cuda_dataset[0]);
	float* ds_ptr_naive = thrust::raw_pointer_cast(&device_cuda_dataset_naive[0]);
	float* query_ptr = thrust::raw_pointer_cast(&device_query_dataset[0]);
	float* result_ptr = thrust::raw_pointer_cast(&device_result[0]);
	int* result_index_ptr = thrust::raw_pointer_cast(&device_result_index[0]);

	int block_size = 960;

	min_dinstance_cuda(ds_ptr_naive, query_ptr, result_ptr, result_index_ptr, dataset_size, DATA_SIZE, QUERY_SIZE, block_size);
	print_result(device_result_index, groundtruth_dataset, dataset_size, QUERY_SIZE, k, false);

	device_result_index = result_index; //reset index vector

	min_dinstance_coalesced_cuda(ds_ptr, query_ptr, result_ptr, result_index_ptr, dataset_size, DATA_SIZE, QUERY_SIZE, block_size);
	print_result(device_result_index, groundtruth_dataset, dataset_size, QUERY_SIZE, k, false);

	device_result_index = result_index; //reset index vector

	min_dinstance_coalesced_shQ_cuda(ds_ptr, query_ptr, result_ptr, result_index_ptr, dataset_size, DATA_SIZE, QUERY_SIZE, block_size);
	print_result(device_result_index, groundtruth_dataset, dataset_size, QUERY_SIZE, k, true);

	device_result_index = result_index; //reset index vector

	min_dinstance_coalesced_constQ_cuda(ds_ptr, query_dataset, device_result, result_index_ptr, dataset_size, DATA_SIZE, QUERY_SIZE, block_size);
	print_result(device_result_index, groundtruth_dataset, dataset_size, QUERY_SIZE, k, false);

	device_result_index = result_index; //reset index vector

	min_dinstance_coalesced_shDS_constQ_cuda(ds_ptr, query_dataset, device_result, result_index_ptr, dataset_size, DATA_SIZE, QUERY_SIZE, block_size);
	print_result(device_result_index, groundtruth_dataset, dataset_size, QUERY_SIZE, k, false);

	KNN_CPU* knn_cpu = new KNN_CPU(dataset_size, DATA_SIZE);

	cout << "=========================== MIN DISTANCE CPU ===========================" << endl << endl;
	
	device_result_index = result_index; //reset index vector

	double start = omp_get_wtime();
	vector< pair< float, int >> kMin = knn_cpu->minDinstanceCPU(cpu_dataset, query_dataset, k);
	printf_s("TIME: %.16g\n", omp_get_wtime() - start);
	knn_cpu->print_result(kMin, groundtruth_dataset, QUERY_SIZE, k, k_number, true);

}