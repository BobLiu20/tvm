"""TVM operator instance normalization compute."""
from __future__ import absolute_import
import tvm
from .. import tag
from .. import cpp

@tvm.tag_scope(tag=tag.BROADCAST)
def instance_norm_inference(data, gamma, beta, eps, fix_gamma):
    """Instance normalization inference operator in NCHW layout.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, height, width]

    gamma : tvm.Tensor
        1-D with shape [channel]

    beta : tvm.Tensor
        1-D with shape [channel]

    eps : float
        Epsilon to prevent div 0.

    fix_gamma : boolean
        Fix gamma while training

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, height, width]

    mean : tvm.Tensor
        1-D with shape [channel]

    var : tvm.Tensor
        1-D with shape [channel]
    """
    assert len(data.shape) == 4, "only support 4-dim instance norm"
    batch, channel, height, width = data.shape
    rh = tvm.reduce_axis((0, height), name="rh")
    rw = tvm.reduce_axis((0, width), name="rw")
    rh2 = tvm.reduce_axis((0, height), name="rh2")
    rw2 = tvm.reduce_axis((0, width), name="rw2")
    s_mean = tvm.compute((batch, channel), \
            lambda b, c: tvm.sum(data[b, c, rh, rw] / (height * width), axis=[rh, rw]))
    s_mean_2 = tvm.compute((batch, channel), \
            lambda b, c: tvm.sum(tvm.intrin.power(data[b, c, rh2, rw2], 2) / (height * width), axis=[rh2, rw2]))
    s_var = tvm.compute((batch, channel), \
            lambda b, c: s_mean_2[b, c] - tvm.intrin.power(s_mean[b, c], 2))
    if fix_gamma:
        out = tvm.compute((batch, channel, height, width), \
            lambda b, c, h, w: (data[b, c, h, w] - s_mean[b, c]) / \
            tvm.intrin.sqrt(s_var[b, c] + eps) + beta[c])
    else:
        out = tvm.compute((batch, channel, height, width), \
            lambda b, c, h, w: (data[b, c, h, w] - s_mean[b, c]) / \
            tvm.intrin.sqrt(s_var[b, c] + eps) * gamma[c] + beta[c])
    return out
    # return cpp.nn.instance_norm_inference(data, gamma, beta, eps, False)
