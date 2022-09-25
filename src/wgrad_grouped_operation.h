/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/* \file
  \brief Defines operations for all CONV operation kinds in CUTLASS Library.
*/

#pragma once
#include <iostream>

// #include "base_operation.h"
#include "cutlass_wgrad_grouped.h"

#include "cutlass/cutlass.h"
#include "cutlass/conv/conv2d_problem_size.h"

#include "cutlass/core_io.h"

#include "cuda_runtime.h"
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_wgrad_grouped {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conv2d library operation class for cutlass profiler
//
///////////////////////////////////////////////////////////////////////////////////////////////////

struct wGradGroupedConfig {
  /// Conv2d problem size 
  //  contains strictly conv2d size (N,H,W,C,K,R,S,P,Q,padding,stride,dilation,mode)
  //  also includes (split_k_slices, groups)
  cutlass::conv::Conv2dProblemSize* problem_sizes;
  int problem_count;

  void * ref_A;
  void * ref_B;
  void * ref_C;
  void * ref_D;
};

template <typename Operator_>
class Conv2dOperation : public Operation {
public:

  using Operator = Operator_;

  using OperatorArguments = typename Operator::Arguments;

public:
    /// Constructor
  Conv2dOperation(std::string name = "unknown_conv2d_fprop") : Operation(name) {
  }

protected:  
  /// Gets the host-side workspace
  virtual uint64_t get_host_workspace_size() const {

    return sizeof(Operator);
  }
  
  /// Initializes the workspace
  virtual Status initialize(
    void const *_arguments, 
    void *host_workspace) const {

    wGradGroupedConfig * arguments = (wGradGroupedConfig *) _arguments;

    OperatorArguments args;
    typename Operator::EpilogueOutputOp::Params epilogue_op(1.0, 0.0);

    args.problem_sizes = arguments->problem_sizes;
    args.problem_count = arguments->problem_count;
    args.threadblock_count = 0;
    args.ref_A = (typename Operator::TensorRefA *)arguments->ref_A;
    args.ref_B = (typename Operator::TensorRefB *)arguments->ref_B;
    args.ref_C = (typename Operator::TensorRefC *)arguments->ref_C;
    args.ref_D = (typename Operator::TensorRefC *)arguments->ref_D;

    Operator *op = new (host_workspace) Operator;
    //std::cout << "initialize library::Conv2dOperation" << std::endl;
    //print_operator_args(args);
    return static_cast<Status>(static_cast<int>(op->initialize(args)));

  }

  /// Update pointers of tensor data
  virtual Status update_ptrs(
    void **ptr_A,
    void **ptr_B,
    void **ptr_C,
    void **ptr_D,
    int problem_count,
    void *host_workspace) const {

    Operator *op = new (host_workspace) Operator;

    return static_cast<Status>(
      static_cast<int>(op->update_ptrs(ptr_A,
                                       ptr_B,
                                       ptr_C,
                                       ptr_D)));
  }

  /// Runs the kernel
  virtual Status run(void *host_workspace) const {

    Operator *op = new (host_workspace) Operator;
    //std::cout << "run library::Conv2dOperation" << std::endl;
    //print_operator_args(args);
    return static_cast<Status>(static_cast<int>(op->run()));
  }

};

}  // namespace cutlass
///////////////////////////////////////////////////////////////////////////////////////////////////
