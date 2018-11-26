# pylint: disable=invalid-name
"""scheduler functions for cuda backend"""
from __future__ import absolute_import as _abs

import tvm
from .. import generic
from .. import cpp

@generic.schedule_instance_norm.register(["cuda"])
def schedule_instance_norm(outs):
    fin = outs[0].op.input_tensors
    mean = fin[1]
    var = fin[2]
    gamma = fin[3]
    beta = fin[4]
    mean_2 = var.op.input_tensors[0]
    ts = [mean, mean_2, var, outs[0]]

    s = tvm.create_schedule([x.op for x in ts])

    def _schedule(Out):  # Out is tensor here
        num_thread = 8
        block_x = tvm.thread_axis("blockIdx.x")
        block_y = tvm.thread_axis("blockIdx.y")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
        by, ty = s[Out].split(s[Out].op.axis[0], factor=num_thread)
        bx, tx = s[Out].split(s[Out].op.axis[1], factor=num_thread)
        s[Out].reorder(by, bx, ty, tx)
        s[Out].bind(ty, thread_y)
        s[Out].bind(tx, thread_x)
        s[Out].bind(by, block_y)
        s[Out].bind(bx, block_x)

    for x in ts:
        _schedule(x)
    return s

@generic.schedule_lrn.register(["cuda"])
def schedule_lrn(outs):
    """Schedule for LRN

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of LRN
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.cuda.schedule_lrn(cpp_target, outs)

@generic.schedule_l2_normalize.register(["cuda"])
def schedule_l2_normalize(outs):
    """Schedule for L2 normalize

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of L2 normalize
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.cuda.schedule_l2_normalize(cpp_target, outs)
