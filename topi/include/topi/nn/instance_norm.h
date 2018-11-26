/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Instance normalization op constructions
 * \file nn/Instance_norm.h
 */
#ifndef TOPI_NN_INSTANCE_NORM_H_
#define TOPI_NN_INSTANCE_NORM_H_

#include <string>

#include "topi/tags.h"
#include "tvm/tvm.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Instance normalization inference operator with NCHW layout
*
* \param x The input tensor. 4-D with shape [batch, channel, height, width]
* \param gamma 1-D with shape [channel]
* \param beta 1-D with shape [channel]
* \param moving_mean 1-D with shape [channel]
* \param moving_var 1-D with shape [channel]
* \param eps Epsilon to prevent div by 0
* \param fix_gamma Fix gamma while training
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the instance normalization operation
*/
inline Tensor instance_norm_inference(const Tensor& x,
                                   const Tensor& gamma,
                                   const Tensor& beta,
                                   float eps,
                                   bool fix_gamma,
                                   std::string name = "tensor",
                                   std::string tag = kBroadcast) {
  CHECK_EQ(x->shape.size(), 4) << "Instance norm requires 4-D input";

  auto batch = x->shape[0];
  auto channel = x->shape[1];
  auto height = x->shape[2];
  auto width = x->shape[3];
  auto rh = tvm::reduce_axis(Range(0, height), "rh");
  auto rw = tvm::reduce_axis(Range(0, width), "rw");
  auto rh2 = tvm::reduce_axis(Range(0, height), "rh2");
  auto rw2 = tvm::reduce_axis(Range(0, width), "rw2");

  Tensor s_mean;
  Tensor s_mean_2;
  Tensor s_var;
  s_mean = tvm::compute(
          {batch, channel},
          [&](Var i, Var j){
            return tvm::sum(x(i, j, rh, rw) / (height * width), {rh, rw});
          }
          );
  s_mean_2 = tvm::compute(
          {batch, channel},
          [&](Var i, Var j){
            return tvm::sum(tvm::pow(x(i, j, rh2, rw2), 2) / (height * width), {rh2, rw2});
          }
          );
  s_var = tvm::compute(
          {batch, channel},
          [&](Var i, Var j){
            return s_mean_2(i, j) - tvm::pow(s_mean(i, j), 2); 
          }
          );

  Tensor out;
  if (fix_gamma) {
    out = tvm::compute(
      x->shape,
      [&](Var b, Var c, Var i, Var j){
        return (x(b, c, i, j) - s_mean(b, c)) / tvm::sqrt(s_var(b, c) + eps) + beta(c);
      }, name, tag);
  } else {
    out = tvm::compute(
      x->shape,
      [&](Var b, Var c, Var i, Var j){
        return (x(b, c, i, j) - s_mean(b, c)) / tvm::sqrt(s_var(b, c) + eps) * gamma(c) + beta(c);
      }, name, tag);
  }
  return out;
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_INSTANCE_NORM_H_
