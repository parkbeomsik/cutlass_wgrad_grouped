
/*
    Generated by generate_cutlass_code.py - Do not edit.
*/

#include "wgrad_grouped_operation.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution_grouped.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad_grouped.h"
#include "cutlass/conv/device/implicit_gemm_convolution_grouped.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

////////////////////////////////////////////////////////////////////

using cutlass_simt_swgrad_grouped_optimized_32x32x4_16x32x4_1x1x1_5_nhwc_base = typename cutlass::conv::kernel::DefaultConv2dWgradGrouped<
    float, 
    cutlass::layout::TensorNHWC,
    float,
    cutlass::layout::TensorNHWC,
    float, cutlass::layout::TensorNHWC,
    float, 
    cutlass::arch::OpClassSimt, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 4>,
    cutlass::gemm::GemmShape<16, 32, 4>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        float, 1,
        float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
    5,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
    >::Conv2dWgradKernel;

// Derived class
struct cutlass_simt_swgrad_grouped_optimized_32x32x4_16x32x4_1x1x1_5_nhwc : 
  public cutlass_simt_swgrad_grouped_optimized_32x32x4_16x32x4_1x1x1_5_nhwc_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_wgrad_grouped {

// Initialize all instances
void initialize_cutlass_simt_swgrad_grouped_optimized_32x32x4_16x32x4_1x1x1_5_nhwc(std::vector<Operation *> &operation_list) {


  using Operation_cutlass_simt_swgrad_grouped_optimized_32x32x4_16x32x4_1x1x1_5_nhwc = cutlass::conv::device::ImplicitGemmConvolutionGrouped<
    cutlass_simt_swgrad_grouped_optimized_32x32x4_16x32x4_1x1x1_5_nhwc>;

  operation_list.push_back(new Conv2dOperation<
    Operation_cutlass_simt_swgrad_grouped_optimized_32x32x4_16x32x4_1x1x1_5_nhwc>(
      "cutlass_simt_swgrad_grouped_optimized_32x32x4_16x32x4_1x1x1_5_nhwc"));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
    