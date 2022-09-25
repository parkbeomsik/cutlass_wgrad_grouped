#include <vector>
// #include "base_operation.h"
// #include "cutlass_error.h"
#include "cutlass_wgrad_grouped.h"
#include "initialize_all.h"
#include "wgrad_grouped_operation.h"

#include "cuda_runtime.h"
#include "cuda_error_helper.h"

#include "cutlass/conv/conv2d_problem_size.h"


namespace cutlass_wgrad_grouped {

void * _device_problems;

std::vector<Operation *> operations;

std::vector<void *> device_workspaces;
// std::vector<void *> host_workspaces;

std::vector<OperationWithWorkspace> operations_with_workspaces;

void initialize() {

    initialize_all(operations);
}

void initialize_problems(std::vector<Conv2dConfig> const & host_configs) {

    using namespace cutlass::conv;
    
    int problem_count = host_configs.size();

    // Set problem sizes in host memory first
    std::vector<Conv2dProblemSize> host_problems;
    for (int i = 0; i < problem_count; ++i) {
        Conv2dConfig host_config = host_configs.at(i);

        // Set single problem in host
        Conv2dProblemSize problem(host_config.N, 
                                  host_config.H, host_config.W, host_config.C,
                                  host_config.K, host_config.R, host_config.S,
                                  host_config.P, host_config.Q,
                                  host_config.pad_h, host_config.pad_w,
                                  host_config.stride_h, host_config.stride_w,
                                  host_config.dilation_h, host_config.dilation_w,
                                  Mode::kCrossCorrelation);

        host_problems.push_back(problem);
    }

    assert(host_problems.size() == problem_count);

    // Set problems in device memory
    checkCudaErrors(cudaMalloc(&_device_problems, 
                                (size_t)sizeof(Conv2dProblemSize)*problem_count));
    checkCudaErrors(cudaMemcpy((void *)_device_problems, (void *)host_problems.data(), 
                               (size_t)sizeof(Conv2dProblemSize)*problem_count,
                               cudaMemcpyHostToDevice));

    // Set problems of operations (will set tensor data ptrs later)
    wGradGroupedConfig wgrad_config = {(Conv2dProblemSize *)_device_problems, problem_count, NULL, NULL, NULL, NULL};
    for (auto operation : operations) {
        void * host_workspace = malloc(operation->get_host_workspace_size());
        operation->initialize(&wgrad_config, host_workspace);
        operations_with_workspaces.push_back(OperationWithWorkspace({operation, host_workspace}));
    }

    assert(operations_with_workspaces.size() == operations.size());

}

void finalize() {
    for (auto device_workspace : device_workspaces){
        checkCudaErrors(cudaFree(device_workspace));
    }
    for (auto operation_with_workspace : operations_with_workspaces) {
        free(operation_with_workspace.host_workspace);
        free(operation_with_workspace.operation);
    }
    
    device_workspaces.clear();
    operations_with_workspaces.clear();
    operations.clear();
}

OperationWithWorkspace get_best_operation(void ** ptr_A,
                              void ** ptr_B,
                              void ** ptr_C,
                              void ** ptr_D) {

    assert(operations_with_workspaces.size() == operations.size());

    std::vector<float> runtime_ms_list;
    runtime_ms_list.resize(operations.size());

    for (int i = 0; i < operations.size(); ++i) {
        auto operation = operations.at(i);
        auto host_workspace = operations_with_workspaces.at(i).host_workspace;

        operation->update_ptrs(ptr_A, ptr_B, ptr_C, ptr_D, host_workspace);

        // Warm up
        for (int iter = 0; iter < 3; ++iter) { 
            operation->run(host_workspace);
        }

        checkCudaErrors(cudaDeviceSynchronize());

        // Measure runtime

        cudaEvent_t events[2];

        for (auto & event : events) {
            checkCudaErrors(cudaEventCreate(&event));
        }

        // Record an event at the start of a series of GEMM operations
        checkCudaErrors(cudaEventRecord(events[0]));

        Status result;
        for (int iter = 0; iter < 20; ++iter) {
            result = operation->run(host_workspace);
        }

        checkCudaErrors(cudaEventRecord(events[1]));
        checkCudaErrors(cudaEventSynchronize(events[1]));

        float runtime_ms;
        checkCudaErrors(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));

        if (result == Status::kSuccess) {
            runtime_ms_list.at(i) = runtime_ms;
        } else {
            runtime_ms_list.at(i) = 100000.0;
        }
        // 
    }

    assert(runtime_ms_list.size() == operations.size());

    float min_runtime_ms = 10000.0;
    OperationWithWorkspace best_operation {NULL, NULL};
    for (int i = 0; i < runtime_ms_list.size(); ++i) {
        if (runtime_ms_list.at(i) < min_runtime_ms) {
            min_runtime_ms = runtime_ms_list.at(i);
            best_operation = operations_with_workspaces.at(i);
        }
    }

    return best_operation;
}

Status run(OperationWithWorkspace operation_with_workspace) {
    return operation_with_workspace.operation->run(operation_with_workspace.host_workspace);
}

Status update_ptrs(OperationWithWorkspace operation_with_workspace,
                   void ** ptr_A,
                   void ** ptr_B,
                   void ** ptr_C,
                   void ** ptr_D,
                   int problem_count) {
    return operation_with_workspace.operation->update_ptrs(ptr_A, ptr_B, ptr_C, ptr_D, problem_count, operation_with_workspace.host_workspace);
}


} // namespace cutlass_wgrad_grouped