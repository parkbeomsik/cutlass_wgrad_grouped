#include "cutlass_wgrad_grouped.h"
#include "cutlass_error.h"
#include "base_operation.h"
#include "cuda_runtime.h"
#include "cutlass_error.h"
#include "cuda_error_helper.h"
#include <vector>

int main() {

    using namespace cutlass_wgrad_grouped;

    cutlass_wgrad_grouped::initialize();

    std::vector<Conv2dConfig> configs(4);
    
    configs.at(0) = {1, 16, 16, 32, 32, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1};
    configs.at(1) = {1, 8, 8, 32, 32, 3, 3, 8, 8, 1, 1, 1, 1, 1, 1};
    configs.at(2) = {1, 16, 16, 32, 32, 3, 3, 8, 8, 1, 1, 2, 2, 1, 1};
    configs.at(3) = {1, 16, 16, 32, 32, 1, 1, 16, 16, 0, 0, 1, 1, 1, 1};

    cutlass_wgrad_grouped::initialize_problems(configs);

    void * host_ptr_A[4];
    void * host_ptr_B[4];
    void * host_ptr_C[4];
    void * host_ptr_D[4];

    for (int i = 0; i < configs.size(); ++i) {
        checkCudaErrors(cudaMalloc(&host_ptr_A[i], sizeof(float)*configs[i].N*configs[i].K*configs[i].P*configs[i].Q));
        checkCudaErrors(cudaMalloc(&host_ptr_B[i], sizeof(float)*configs[i].N*configs[i].C*configs[i].H*configs[i].W));
        checkCudaErrors(cudaMalloc(&host_ptr_C[i], sizeof(float)*configs[i].C*configs[i].K*configs[i].R*configs[i].S));
        checkCudaErrors(cudaMalloc(&host_ptr_D[i], sizeof(float)*configs[i].C*configs[i].K*configs[i].R*configs[i].S));
    }

    void ** device_ptr_A;
    void ** device_ptr_B;
    void ** device_ptr_C;
    void ** device_ptr_D;
    checkCudaErrors(cudaMalloc(&device_ptr_A, sizeof(float *) * 4));
    checkCudaErrors(cudaMalloc(&device_ptr_B, sizeof(float *) * 4));
    checkCudaErrors(cudaMalloc(&device_ptr_C, sizeof(float *) * 4));
    checkCudaErrors(cudaMalloc(&device_ptr_D, sizeof(float *) * 4));

    checkCudaErrors(cudaMemcpy(device_ptr_A, host_ptr_A, sizeof(float *) * 4,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ptr_B, host_ptr_B, sizeof(float *) * 4,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ptr_C, host_ptr_C, sizeof(float *) * 4,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ptr_D, host_ptr_D, sizeof(float *) * 4,
                                cudaMemcpyHostToDevice));

    OperationWithWorkspace best_operation = get_best_operation(device_ptr_A, device_ptr_B, device_ptr_C, device_ptr_D);

    printf("%p\n", best_operation.operation);

    Status status = run(best_operation);

    printf("%s\n", cutlassGetStatusString(status));


    cutlass_wgrad_grouped::finalize();
    
    return 0;
}