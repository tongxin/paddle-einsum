# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import re
import paddle

__all__ = ['einsum']


def parse_op_labels(labelstr, operand):
    '''
    Parses labels for an input operand.

    Parameters
    ----------
    labelstr:
        the 
    Returns an extended label string with all missing labels filled as dots
    '''
    # Sanity checks
    assert all(c.isalpha() for c in labelstr.replace('.', '')), \
        f"Invalid equation: a label is expected to be in [a-Z]."

    assert labelstr.replace('...', '', 1).find('.') == -1, \
        f"Invalid equation: `.` is only expected to be included in an ellipsis."

    # Check shape
    ndims = len(operand.shape) # Note, in Paddle a tensor rank is always nonzero
    assert ndims > 0
    
    full_labelstr = labelstr.replace('...', '.' * (ndims - len(labelstr) + 3))

    assert len(full_labelstr) == ndims, \
        f"Invalid equation: the label string '{labelstr}' misses dimensions."

    return full_labelstr

def parse_labels(labelstr, operands):
    '''
    Parse out a list of distinct labels and count their number of occurrences.
    
    Parameters
    ----------
    labelstr:
        The equation's label string
    nop:
        Number of operands
    
    Returns
    -------
    nop_label:
        list of full label strings matching the each operand's dimensino size
    '''

    nop_labels = labelstr.split(',')
    assert len(nop_labels) == len(operands), \
        f"Invalid equation: the number of operands is {len(operands)} but only found {len(nop_labels)} in the label string."
    
    return list(map(parse_op_labels, nop_labels, operands))

def validate_rhs(rhs, input_labels, n_bcast_dims):
    # Sanity check.
    if n_bcast_dims > 0:
        assert '...' in rhs, f"Invalid equation: missing ellipsis in output labels"
    rhs = rhs.replace('...', '')
    rhs_set = set(rhs)
    # Hidden assumption: availble labels don't include '.'
    assert '.' not in input_labels
    # Verify that output labels all come from the set of input labels
    non_input_labels = rhs_set.difference(input_labels)
    assert not non_input_labels, \
        f"Invalid equation: output label '{non_input_labels}' not used by any input."
    # Verify that output labels are not duplicate
    assert len(rhs) == len(rhs_set), \
        f"Invalid equation: duplicate output labels are found."

def count_bcast_dims(extended_labels, operand=None):
    '''
    Returns the number of broadcast dimensions
    '''
    return extended_labels.count('.')

def bcastable_test(args, f=None):
    '''
    Tests if the two operands can perform a broadcast operation on the given ranges of dimensions. 
    We follow the Numpy broadcasting convention which states that, by lining up the shape arrays
    starting from the right most dimension, all the aligned dimensions either have equal sizes or
    one of them is sized one.

    Parameters
    ----------
    args:
        *args unpacks into operand one's axes range, shape, operand two's axes range, shape

    f: 
        if available, is used as a callback for postprocessing the aligned operand dimensions.
    '''
    xran, xshape, yran, yshape = args

    xran_inv, yran_inv = xran[::-1], yran[::-1]

    for xi, yi in zip(xran_inv, yran_inv):
        xs, ys = xshape[xi], yshape[yi]
        cond = xs == ys or xs == 1 or ys == 1
        if not cond:
            return False

    if not f:
        return True

    # Apply the callback to each aligned dimension pair
    for xi, yi in zip(xran_inv, yran_inv):
        f(xi, yi)

def part_labels(nop_labels, rhs, n_bcast_dims):
    '''
    Part labels in two strings, the output label string and the combined label string. 
    Infer output labels in case no explicit right hand side is given. In this case
    the output label string is formed by labels that count only once, in alphabetical order.

    Returns
    -------
    output:
        output label string
    combine:
        combine label string
    count:
        combine labels' count
    '''
    # Put all labels in alphabetical order
    concat = sorted(''.join(nop_labels).replace('.', ''))
    labels, count = [], []
    for a, b in zip(['.'] + concat, concat):
        if a != b:
            labels.append(b)
            count.append(1)
        else:
            count[-1] += 1
    if rhs != None:
        validate_rhs(rhs, labels, n_bcast_dims)
        output = rhs.replace('...', '.' * n_bcast_dims)
    else:
        output = '.' * n_bcast_dims + ''.join(l for l, c in zip(labels, count) if c == 1)
    
    for i in range(len(count))[::-1]:  # Ouch ... it hurts to get confused with [-1:]
        if labels[i] in output:
            labels.pop(i)
            count.pop(i)
    
    combine = ''.join(labels)
    return output, combine, count

def dim_strides(shape):
    '''
    Returns the dimension strides for a tensor shape
    '''
    strides = []
    stride = 1
    for size in shape[::-1]:
        strides.append(stride)
        stride = stride * size
    return strides

def create_op_view(operand, *view_def):
    '''
    Create and materialize a view of an operand.
    
    Parameters
    ----------

    operand:
        the base tensor operand

    view_def: 
        include two lists which define the view's dimension sizes and strides
    '''
    view_sizes, view_strides = view_def
    return operand.create_view(view_sizes, view_strides)    

def has_duplicated_labels(labels):
    '''
    Returns True if there is any duplicate label.
    '''
    labels = labels.replace('.', '')
    return any(l in labels[i+1:] for i, l in enumerate(labels))

def diagonalize(labels, operand):
    '''
    Merges dimensions if there are duplicated labels. 
    
    For those dimensions with duplicate labels, merge them into one dimension
    which represents the diagonal elements. That requires the duplicate labeled 
    dimensions have the same size. The order of dimensions is kept unchanged
    up to the left-most appearance of each label.

    Examples
    -------- 

    'ijj...i' would be merged into 'ij...'

    '''
    if not has_duplicated_labels(labels):
        return labels, operand

    strides = dim_strides(operand.shape)
    shape = operand.shape
    new_labels = []
    new_sizes = []
    new_strides = []

    for ax, l in enumerate(labels):
        new_ax = new_labels.index(l)
        if new_ax < 0 or l == '.':
            # not duplicate
            new_labels.append(l)
            new_strides.append(strides[ax])
            new_sizes.append(shape[ax])
        else:
            # duplicated label
            new_strides[new_ax] += strides[ax]

    # call framework API to build a new tensor
    new_op = create_op_view(operand, new_sizes, new_strides)
    return new_labels, new_op


def dims_index(in_labels, out_labels):
    '''
    Build an inverse map of dimension indices. Following prerequisites must hold to make
    the result meaningful. First, there's no duplicate alphabet labels in either parameters.
    Second, the broadcast dimensions in out_labels, are at least as many as in in_labels.
    Third, indices of broadcast dimension are contiguous.

    Parameters
    ----------
    in_labels:
        The dimension labels to map to
    out_labels:
        The dimension labels to map from
    

    Returns
    -------
    The inverse map from out_labels to in_labels. The length of the inverse map equals that of
    out_labels. -1 is filled if there's no matching intput dimension for a specific label.

    Examples
    --------
    in_labels = 'ij..', out_labels = '..ji'
    inv_map = [2, 3, 1, 0]

    in_labels = 'ij..', out_labels = '..kji'
    inv_map = [2, 3, -1, 1, 0]
    '''
    # print(f"in labels: '{in_labels}'     out labels: '{out_labels}'")

    inv_map = [-1] * len(out_labels)
    
    # First build the broadcast dimension mapping
    # Find the broadcast index range in out_labels
    r = re.search(r'\.+', out_labels)
    if r:
        start, end = r.start(), r.end()
        s = re.search(r'\.+', in_labels)
        # fill the broadcast dimension indices from right to left.
        if s:
            for ax, dim in zip(range(start, end)[::-1], range(s.start(), s.end())[::-1]):
                inv_map[ax] = dim
        
    # Now work on non-broadcast dimensions 
    if r:
        it = itertools.chain(range(start), range(end, len(out_labels)))
    else:
        it = iter(range(len(out_labels)))

    for i in it:
        inv_map[i] = in_labels.find(out_labels[i])

    return inv_map

def verify_shape(axes_list, operands):
    # for axes in axes_list:
        # print(axes)
    op_shapes = [op.shape for op in operands]
    for ax_dims in zip(*axes_list):
        # axes are a column of nop input dimension axes. -1 represents new axis
        # all non size-one dimensions must have the same size
        sizes, ops, op_dims = [], [], []
        for dim, shape, op in zip(ax_dims, op_shapes, operands):
            if dim > -1 and shape[dim] > 1:
                sizes.append(shape[dim])
                ops.append(op)
                op_dims.append(dim)

        for s1, s2, ax1, ax2, op1, op2 in zip(sizes, sizes[1:], op_dims, op_dims[1:], ops, ops[1:]):
            assert s1 == s2, f'Dimension {ax1} in {op1.name} and dimension {ax2} in {op2.name} do not match in size.'

def prod(iter, default=1):
    if len(iter):
        res = 1
        for s in iter:
            res *= s
        return res
    return default

def plan_squeeze(plan, op, op_axes, op_shape, squeeze_axes):
    varname = f'op{op}'
    squeeze_dims = []

    # Update axes and reset mappings for squeezed dims
    for ax in squeeze_axes:
        dim = op_axes[ax]
        squeeze_dims.append(dim)
        op_axes[ax] = -1
    for dim in sorted(squeeze_dims)[-1:]:
        op_shape.pop(dim)

    dims_left = sorted(dim for dim in op_axes if dim >= 0)
    for i in range(len(op_axes)):
        old = op_axes[i]
        if old >= 0:
            op_axes[i] = dims_left.index(old)
    # Be aware that the op label string is not updated yet...

    step = paddle.squeeze, [varname], varname, squeeze_dims
    plan.add_step(step)

def plan_reduce(plan, op, op_axes, op_shape, reduce_axes):
    '''
    Add reduce to the plan
    '''
    varname = f'op{op}'
    reduce_dims = []

    # Update axes and reset mappings for squeezed dims
    for ax in reduce_axes:
        dim = op_axes[ax]
        reduce_dims.append(dim)
        op_axes[ax] = -1
    for dim in sorted(reduce_dims)[-1:]:
        op_shape.pop(dim)

    dims_left = sorted(dim for dim in op_axes if dim >= 0)
    for i in range(len(op_axes)):
        old = op_axes[i]
        if old >= 0:
            op_axes[i] = dims_left.index(old)

    step = paddle.sum, [varname], varname, reduce_dims
    plan.add_step(step)

    # Be aware that the op label string is not updated yet...

def plan_scalar_prod(plan, operands, op1, op2):    
    varnames = [f'op{op1}', f'op{op2}']
    f = lambda var1, var2: var1 * var2
    step = f, varnames, varnames[1]
    plan.add_step(step)

def plan_matmul(plan, op1, op2, op1_axes, op2_axes, op1_shape, op2_shape, I, J1, J2, K):
    '''
    plan matmul
    '''
    # Transpose and re-shape op1 and op2 in I, J1, K and I, J2, K
    # Then apply matmul(x, y, transpose_x=False, tranpose_y=True)
    var1, var2 = f'op{op1}', f'op{op2}'

    # Note, I may index into -1
    I1_dims = [op1_axes[ax] for ax in I if op1_axes[ax] >= 0]
    I2_dims = [op2_axes[ax] for ax in I if op2_axes[ax] >= 0]
    J1_dims = [op1_axes[ax] for ax in J1]
    J2_dims = [op2_axes[ax] for ax in J2]
    K1_dims = [op1_axes[ax] for ax in K]
    K2_dims = [op2_axes[ax] for ax in K]
    I1_shape, J1_shape, K1_shape = [[op1_shape[dim] for dim in dims] for dims in (I1_dims, J1_dims, K1_dims)]
    I2_shape, J2_shape, K2_shape = [[op2_shape[dim] for dim in dims] for dims in (I2_dims, J2_dims, K2_dims)]
    K1_size, J1_size, J2_size = prod(K1_shape), prod(J1_shape), prod(J2_shape)

    perm1 = I1_dims + J1_dims + K1_dims
    perm2 = I2_dims + J2_dims + K2_dims
    
    if any(i != dim for i, dim in enumerate(perm1)):
        # print(f'perm1: {perm1}')
        step = paddle.transpose, [var1], var1, perm1
        plan.add_step(step)
        # update axes index
        # for i, dim in enumerate(op1_axes):
        #     if dim >= 0:
        #         new_dim = perm1.index(dim)
        #         op1_axes[i] = new_dim

    if any(i != dim for i, dim in enumerate(perm2)):
        # print(f'perm2: {perm2}')
        step = paddle.transpose, [var2], var2, perm2
        plan.add_step(step)
        # update axes index
        # for i, dim in enumerate(op2_axes):
        #     if dim >= 0:
        #         new_dim = perm2.index(dim)
        #         op2_axes[i] = new_dim

    # In case no K... dimensions remain, do a broadcast
    if not K:
        # unsqueeze operands include J1...J2... dimensions
        if J2:
            fill_start = len(I2_dims) + len(J1)
            fill_end = fill_start + len(J2)
            fill = list(range(fill_start, fill_end))
            step = paddle.unsqueeze, [var1], var1, fill
            plan.add_step(step)
        if J1:
            fill_start = len(I2_dims)
            fill_end = fill_start + len(J1)
            fill = list(range(fill_start, fill_end))
            step = paddle.unsqueeze, [var2], var2, fill
            plan.add_step(step)
        # make broadcast
        step = paddle.multiply, [var1, var2], var2
        plan.add_step(step)
    # K... are there, let's reason about I... and J...
    # In case I... and J... are empty, do the vector-vector version of matmul
    elif not I and not J1 and not J2:
        # merge K dimensions
        if len(K) > 1:
            for var in var1, var2:
                step = paddle.reshape, [var], var, [K1_size]
                plan.add_step(step)
        # Build vector-vector matmul
        step = paddle.matmul, [var1, var2], var2
        plan.add_step(step)
    # General case, there are K... and some I... and J..., the actual operation will be 
    # matrix-vector or matrix-matrix multiplies, depending on the operands' shapes.
    else:
        # Merge J dims and K dims by reshaping
        merged_shape1 = I1_shape + [J1_size] + [K1_size]
        merged_shape2 = I2_shape + [J2_size] + [K1_size]

        step = paddle.reshape, [var1], var1, merged_shape1
        plan.add_step(step)
        step = paddle.reshape, [var2], var2, merged_shape2
        plan.add_step(step)

        # Matmul
        step = paddle.matmul, [var1, var2], var2, False, True
        plan.add_step(step)

    # The result shape is in I..., J1, J2. Let's reshape back to known dimensions
    # Note, this is static deduction, not by reading the tensor shape at runtime
    result_shape = [1] * len(I)
    for i, ax in enumerate(I):
        dim1, dim2 = op1_axes[ax], op2_axes[ax]
        s = 1 if dim1 < 0 else op1_shape[dim1]
        s = s if dim2 < 0 else max(s, op2_shape[dim2])
        result_shape[i] = s
    if J1:
        result_shape += J1_shape
    if J2:
        result_shape += J2_shape

    # Need a scalar dimension somehow
    if result_shape:
        step = paddle.reshape, [var2], var2, result_shape
        plan.add_step(step)

    # Wrap up, updating auxiliary data
    op2_shape.clear()
    for s in result_shape:
        op2_shape.append(s)

    for ax in range(len(op2_axes)):
        op2_axes[ax] = -1
    dim = 0
    for ax in I:
        op2_axes[ax], dim = dim, dim+1
    for ax in J1 + J2:
        op2_axes[ax], dim = dim, dim+1

def plan_summation(plan, ops, nop_axes, nop_shapes, op1, op2, ndims_bcast, label_count):
    '''
    Plan various kinds of summation
    '''
    op1_axes, op2_axes = [nop_axes[op] for op in (op1, op2)]
    op1_shape, op2_shape = [nop_shapes[op] for op in (op1, op2)]

    ndims = len(op1_axes)
    ndims_out = ndims - len(label_count)

    count = [0] * ndims_out + label_count

    I, K, J1, J2 = list(range(ndims_bcast)), [], [], []

    for ax in range(ndims_bcast, ndims):
        dim1, dim2 = op1_axes[ax], op2_axes[ax]

        if (dim1 != -1) != (dim2 != -1):
            if dim1 != -1:
                J1.append(ax)
            else:
                J2.append(ax)
        elif dim1 != -1:
            if count[ax] == 2:
                # kill this axis
                K.append(ax)
                count[ax] = 0   
            else:
                I.append(ax)
                # Decrement count
                if ax >= ndims_out:
                    label_count[ax - ndims_out] -= 1

    # Now it's OK to merge the K dims as the same shape holds
    # print(f'I: {I}   J1: {J1}    J2: {J2}   K: {K}')

    plan_matmul(plan, op1, op2, op1_axes, op2_axes, op1_shape, op2_shape, I, J1, J2, K)

def rearrange(axes):
    perm, fill = [], []
    for ax, dim in enumerate(axes):
        if dim < 0:
            fill.append(ax)
        else:
            perm.append(dim)
    # Trivial permutation returns []
    if all(i == dim for i, dim in enumerate(perm)):
        perm = []
    
    return perm, fill

def plan_broadcast(plan, operands, nop_axes):
    '''
    Plan broadcast across
    '''
    nop = len(operands)
    varnames = [f'op{i}' for i in range(nop)]

    for i, op_axes in zip(range(nop), nop_axes):
        # Re-arrange the dimesions according to the global layout
        perm, fill = rearrange(op_axes)
        var = varnames[i]
        if perm:
            step = paddle.transpose, [var], var, perm
            plan.add_step(step)
        if fill:
            step = paddle.unsqueeze, [var], var, fill
            plan.add_step(step)

    def f(*args):
        expr = ' * '.join(varnames)
        return eval(expr, dict(zip(varnames, args)))

    step = f, varnames, None
    plan.add_step(step)

def labels_to_axes(labelstr, labels):
    return [i for i, l in enumerate(labelstr) if l in labels]

class Plan:
    def __init__(self):
        self.env = {}
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)

    def get_var(self, varname):
        return self.env[varname] if varname in self.env else None

    def set_var(self, varname, var):
        self.env[varname] = var

    def show(self):
        res = None
        for f, in_varnames, out_varname, *args in self.steps:
            print(repr((out_varname, f, *in_varnames, *args)))
            if out_varname:
                self.set_var(out_varname, res)
        return res

    def execute(self):
        res = None
        for f, in_varnames, out_varname, *args in self.steps:
            res = f(*map(self.get_var, in_varnames), *args)
            if out_varname:
                self.set_var(out_varname, res)
        return res

def reorder_ops(nop_axes):
    nop = len(nop_axes)
    ndim = len(nop_axes[0])
    opis = list(range(nop))
    perm = []

    for i in range(ndim)[::-1]:
        if not opis:
            break
        to_remove = [opi for opi in opis if nop_axes[opi][i] != -1]
        for opi in to_remove:
            perm.append(opi)
            opis.remove(opi)

    return perm

def plan_einsum(operands, nop_axes, ndims_bcast, label_count):
    '''
    Plans the actual execution steps.

    Results
    -------
    the execution plan
    '''
    nop = len(operands)
    ndims = len(nop_axes[0])
    ndims_combine = len(label_count)
    ndims_out = ndims - ndims_combine

    # Initialize a plan with an environment
    plan = Plan()
    op_names = [f'op{i}' for i in range(nop)]
    list(map(plan.set_var, op_names, operands))
    nop_shapes = [list(op.shape) for op in operands]

    # In case no dimensions to combine, do broadcast straight across
    if not ndims_combine:
        plan_broadcast(plan, operands, nop_axes)
        return plan

    # Canonicalize by removing size-1 to-combine dimensions
    for i, op_axes, shape in zip(range(nop), nop_axes, nop_shapes):
        squeeze_axes = []
        for j in range(ndims_out, ndims):
            dim = op_axes[j]
            if dim >= 0 and shape[dim] == 1:
                squeeze_axes.append(j)
                label_count[j-ndims_out] -= 1
        if squeeze_axes:
            plan_squeeze(plan, i, nop_axes[i], nop_shapes[i], squeeze_axes)

    # Reduce dimensions whose label_count == 1
    for i, op_axes in enumerate(nop_axes):
        reduce_axes = []
        for j in range(ndims_out, ndims):
            dim = op_axes[j]
            if dim >= 0 and label_count[j-ndims_out] == 1:
                reduce_axes.append(j)
                label_count[j-ndims_out] = 0
        if reduce_axes:
            plan_reduce(plan, i, nop_axes[i], nop_shapes[i], reduce_axes)

    # Plan the summations over the operand sequence
    for i in range(nop):
        # plan a single step
        
        if i == 0:
            continue

        # We'd like to arrange the dimensions in the following way:
        # [I...  J... K...]
        # [I...  J... K...]
        # where  
        #       I... are aligned and not to be combined immediately 
        #       J... are not aligned and not to be combined immediately
        #       K... are aligned and should be immediately combined
        # At this point the non-trivial broadcast dimensinos in K are already reduced
        # and removed. That means all K dimensions are aligned and their sizes are not 1.
        # We then inspect the layout of I,J,K plus the above observation to make
        # specializatoin decisions.  The current strategy is set as follows:
        #  (1) if I... J... K... are all empty, it's multiplying a scalar
        #  (2) if K... are empty, better use a broadcast
        #  (3) if I... J... empty and K... not empty, a vector-vector multiply (or a dot)
        #  (4) Elsewise, either I... or J... not empty, and K... not empty, use a general matmul

        # Resolve the summation kind: dot, matmul or *
        if all(dim < 0 for dim in nop_axes[i-1]):
            # op1 is a scalar
            plan_scalar_prod(plan, operands, i-1, i)
        else:
            plan_summation(plan, operands, nop_axes, nop_shapes, i-1, i, ndims_bcast, label_count)

    return plan

def einsum(equation, *operands):
    r"""
    Executes the sum of product of provided operands based on the Einstein summation convention.
    Einsum can be used to complete a variety of operations, such as sum, transpose,
    batch matrix multiplication.

    Args:
        equation (`str`):
            The equation uses uncased letters to indicate dimensions. These letters are called 
            dimension labels. The dimension labels are comma separated into groups the number of which
            equals the number of the input operands. The equation uses `->` to indicate 
            explicitly the output dimensions which otherwise would be collapsed as the result of summation.
            In case the explicit output is not given, Einsum will deduce the output dimensions automatically.
            Dimensions with the same label should be broadcastable. The equation uses ellipsis ('...') to 
            specify broadcast dimensions.

        operands (`Tensor`):
            The operands to compute the Einstein sum of. The number of operands should be the same as the
            the operands described in input equation.
    
    Returns:
        `Tensor`: The result of Einstein sum product.
    
    Example:
    .. code-block::

        import paddle
        import paddlenlp
        import numpy as np

        np.random.seed(102)

        x = paddle.to_tensor(np.random.rand(4))
        y = paddle.to_tensor(np.random.rand(5))
        # sum
        print(paddlenlp.ops.einsum('i->', x))
        # Tensor(shape=[], dtype=float64, place=CUDAPlace(0), stop_gradient=True, 2.30369050)

        # dot
        print(paddlenlp.ops.einsum('i,i->', x, x))
        # Tensor(shape=[], dtype=float64, place=CUDAPlace(0), stop_gradient=True, 1.43773247)

        # outer
        print(paddlenlp.ops.einsum("i,j->ij", x, y)),
        # Tensor(shape=[4, 5], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #         [[0.34590188, 0.48353496, 0.09996135, 0.18656330, 0.21392910],
        #         [0.39122025, 0.54688535, 0.11305780, 0.21100591, 0.24195704],
        #         [0.17320613, 0.24212422, 0.05005442, 0.09341929, 0.10712238],
        #         [0.42290818, 0.59118179, 0.12221522, 0.22809690, 0.26155500]])

        A = paddle.to_tensor(np.random.rand(2, 3, 2))
        B = paddle.to_tensor(np.random.rand(2, 2, 3))
        # transpose
        print(paddlenlp.ops.einsum('ijk->kji', A))
        #  Tensor(shape=[2, 3, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #        [[[0.49174730, 0.33344683],
        #          [0.89440989, 0.26162022],
        #          [0.36116209, 0.12241719]],

        #         [[0.49019824, 0.51895050],
        #          [0.18241053, 0.13092809],
        #          [0.81059146, 0.55165734]]])

        # batch matrix multiplication
        print(paddlenlp.ops.einsum('ijk, ikl->ijl', A,B))
        # Tensor(shape=[2, 3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #     [[[0.13654339, 0.39331432, 0.65059661],
        #      [0.07171420, 0.57518653, 0.77629221],
        #      [0.21250688, 0.37793541, 0.73643411]],

        #     [[0.56925339, 0.65859030, 0.57509818],
        #      [0.30368265, 0.25778348, 0.21630400],
        #      [0.39587265, 0.58031243, 0.51824755]]])

        # Ellipsis transpose
        print(paddlenlp.ops.einsum('...jk->...kj', A))
        # Tensor(shape=[2, 2, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #     [[[0.49174730, 0.89440989, 0.36116209],
        #         [0.49019824, 0.18241053, 0.81059146]],

        #         [[0.33344683, 0.26162022, 0.12241719],
        #         [0.51895050, 0.13092809, 0.55165734]]])

        # Ellipsis batch matrix multiplication
        print(paddlenlp.ops.einsum('...jk, ...kl->...jl', A,B))
        # Tensor(shape=[2, 3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        # [[[0.13654339, 0.39331432, 0.65059661],
        #     [0.07171420, 0.57518653, 0.77629221],
        #     [0.21250688, 0.37793541, 0.73643411]],

        #     [[0.56925339, 0.65859030, 0.57509818],
        #     [0.30368265, 0.25778348, 0.21630400],
        #     [0.39587265, 0.58031243, 0.51824755]]])
    """

    # if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        # operands = operands[0]
    nop = len(operands)
    assert nop > 0, "At least one operand is expected."

    # Part the equation to the left hand side and the right hand side of ->
    equation = equation.lower().replace(' ', '')
    lhs, *rhs = equation.split('->')

    assert len(rhs) < 2, "Invalid equation: multiple `->` were found."
    # Note, we distinguish between 'ij->' and 'ij' by setting rhs to '' and None
    rhs = rhs[0] if rhs else None

    # Parse labels for each operand and count the number of occurrences for each alphabet label
    nop_labels = parse_labels(lhs, operands)

    # Diagonalize the operands which have duplicate labels
    nop_labels, operands = list(zip(*map(diagonalize, nop_labels, operands)))

    # To handle broadcasting, we should first know how many dimensions are there
    # We need to use that number to generate output labels
    # e.g. 1 for ['ij', 'i.', '.k']
    n_bcast_dims = max(map(count_bcast_dims, nop_labels, operands))

    # Parse or infer output labels. The broadcasting dimensions should be taken care of.
    # Following the Numpy's rule, the output labels must include the broadcasting dimensions,
    # if there are any. 
    out_labels, combine_labels, combine_count  = part_labels(nop_labels, rhs, n_bcast_dims)
    
    # The label order is now resolved
    all_labels = out_labels + combine_labels

    print(f'labels => output: {out_labels}   combine: {combine_labels}')

    # Build global_index, a data structure that maintains the mapping from all_labels
    # to the dimensions in the remained operands during the summation process.  
    f = lambda labels: dims_index(labels, all_labels)
    global_index = list(map(f, nop_labels))

    # Verify that all aligned dimensions are broadcastable in size across operands
    verify_shape(global_index, operands)

    # Reorder the operands and possibly reduce the summation complexity
    perm = reorder_ops(global_index)

    operands = [operands[i] for i in perm]
    nop_labels = [nop_labels[i] for i in perm]
    global_index = [global_index[i] for i in perm]

    # Now we're ready to build up an execution plan
    args = [operands, global_index, n_bcast_dims, combine_count]
    plan = plan_einsum(*args)
    result = plan.execute()

    return result

if __name__ == '__main__':
    import numpy as np

    x = paddle.rand([1, 5, 2, 2, 3, 4])
    y = paddle.rand([5, 2, 3, 4])
    z = paddle.rand([2, 1, 2])
    t = paddle.rand([1, 5, 2, 3, 4])
    einsum('abcdef, bcef, cad', x, y, z)

    x = np.random.randn(5, 1, 10000)
    y = np.random.randn(100, 10000)

    tx = paddle.to_tensor(x)
    ty = paddle.to_tensor(y)

    equations = [               \
        'ijk, jk',              \
        '...k, ...k->...k',     \
        'ij..., j...',          \
        'ij..., j...->...'      \
    ]

    einsum('ijk->j', tx)
    
    for eqn in equations:
        print(einsum(eqn, tx, ty))

    np.random.seed(102)

    tx = paddle.to_tensor(np.random.rand(4))
    ty = paddle.to_tensor(np.random.rand(5))

    equations =[
        'i,i->'
    ]
    for eqn in equations:
        print(einsum(eqn, tx, tx))

    equations = [
        'i,j->ij',
        'i,j->'
    ]
    for eqn in equations:
        print(einsum(eqn, tx, ty))

    x = np.random.randn(10, 1, 4, 256)
    y = np.random.randn(256, 10, 1)
    
    tx, ty = paddle.to_tensor(x), paddle.to_tensor(y)

    equations = [
        'abcd,dfg->d'
    ]
    for eqn in equations:
        print(einsum(eqn, tx, ty))

    x = np.random.rand(10000, 100, 10)
    tx = paddle.to_tensor(x)

    print(einsum('ijk->', tx))

