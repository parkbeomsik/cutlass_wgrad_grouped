
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

using cutlass_simt_swgrad_grouped_optimized_8x64x32_8x64x32_1x1x4_3_nhwc_base = typename cutlass::conv::kernel::DefaultConv2dWgradGrouped<
    float, 
    cutlass::layout::TensorNHWC,
    float,
    cutlass::layout::TensorNHWC,
    float, cutlass::layout::TensorNHWC,
    float, 
    cutlass::arch::OpClassSimt, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<8, 64, 32>,
    cutlass::gemm::GemmShape<8, 64, 32>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombination<
        float, 1,
        float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
    >::Conv2dWgradKernel;

// Derived class
struct cutlass_simt_swgrad_grouped_optimized_8x64x32_8x64x32_1x1x4_3_nhwc : 
  public cutlass_simt_swgrad_grouped_optimized_8x64x32_8x64x32_1x1x4_3_nhwc_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_wgrad_grouped {

// Initialize all instances
void initialize_cutlass_simt_swgrad_grouped_optimized_8x64x32_8x64x32_1x1x4_3_nhwc(std::vector<Operation *> &operation_list) {


  using Operation_cutlass_simt_swgrad_grouped_optimized_8x64x32_8x64x32_1x1x4_3_nhwc = cutlass::conv::device::ImplicitGemmConvolutionGrouped<
    cutlass_simt_swgrad_grouped_optimized_8x64x32_8x64x32_1x1x4_3_nhwc>;

  operation_list.push_back(new Conv2dOperation<
    Operation_cutlass_simt_swgrad_grouped_optimized_8x64x32_8x64x32_1x1x4_3_nhwc>(
      "cutlass_simt_swgrad_grouped_optimized_8x64x32_8x64x32_1x1x4_3_nhwc"));


}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
    