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

def parse_and_count_labels(labelstr, operands):
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
    count:
        the number of occurrences for each label
    '''
    # Counters for 26 alphabetical letters
    
    count = {label : 0 for label in 'abcdefghijklmnopqrstuvwxyz'}
    count = {}

    nop_labels = labelstr.split(',')
    assert len(nop_labels) == len(operands), \
        f"Invalid equation: the number of operands is {len(operands)} but only found {len(nop_labels)} in the label string."
    
    nop_labels = list(map(parse_op_labels, nop_labels, operands))

    for labels in nop_labels:
        for c in set(labels.replace('.', '')):
            if c in count:
                count[c] += 1
            else:
                count[c] = 1
    
    return nop_labels, count

def parse_output_labels(rhs, avail_labels, n_bcast_dims):
    '''
    Parse explicit output labels given on the right hand side of '->' and the available
    input labels.

    Parameters
    ----------
    rhs:
        the output label string, given by the right hand side of the einsum equation
    avail_labels:
        the available labels to check with
    n_bcast_dims:
        the number of broadcast dimensions

    Returns
    -------
    The output labels in a string
    '''
    # Sanity check. Only alphabet is allowed if not '.'
    assert all(c in avail_labels for c in rhs.replace('.', '')), f"Invalid equation: output labels must come from the input labels. "

    # Syntax check. Verify there's no duplicate alphabetical labels
    for i, l in enumerate(rhs.replace('.', '')):
        if rhs.find(l, 0, i) >= 0:
            assert False, f"Invalid equation: duplicate output label {l}."

    # Check there's no dots other than in an ellipsis
    assert rhs.replace('...', '', 1).find('.') == -1, \
        f"Invalid equation: `.` is only expected to be included in an ellipsis."
    
    # Check if ellipsis is missing
    assert (n_bcast_dims > 0) == (rhs.find('...') >= 0), \
        f"Invalid equation: there are broadcasting dimensions yet found no '...' in the output labels."

    out_labels = rhs.replace('...', '.' * n_bcast_dims, 1) if n_bcast_dims > 0 else rhs

    return out_labels

def has_bcast_dims(extended_labels, operand=None):
    '''
    Returns whether there are non-labeled dimensions by checking the extended labels 
    '''
    return '.' in extended_labels

def count_bcast_dims(extended_labels, operand=None):
    '''
    Returns the number of broadcast dimensions
    '''
    return extended_labels.count('.')

def get_bcast_dims_indices_and_shape(op_shape, op_labels):
    '''
    Returns the indices and shape of the broadcast dimensions.

    Parameters
    ----------
    op_shape:
        the tensor shape of the operand
    op_labels:
        the extended label string for the operand. Broadcast dimensions are labeled with dots.
    

    Returns
    -------
    indices:
        the indices of the broadcast dimensions
    shape:
        the sizes of the broadcast dimensions
    '''
    assert len(op_shape) == len(op_labels)

    indices, shape = [], []
    for i, size, label in zip(range(len(op_shape)), op_shape, op_labels):
        if label == '.':
            indices.append(i)
            shape.append(size)

    return indices, shape

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

def gather_labels(labels_list, bcast_ndims):
    '''
    Returns a sorted string of all labels in the list including dots 
    '''
    labelset = set()

    for l in labels_list:
        labelset.update(l)

    return ''.join(sorted(labelset)).replace('.', '.' * bcast_ndims)

def gather_singular_labels(labels_list, alphabet_only=True):
    '''
    Returns the labels which only show in one operand
    Parameter alphabet_only indicates whether to count labels in [a-z] only
    '''
    all_labels = sorted(''.join(labels_list))    

    _off = 0
    if alphabet_only:
        for i, l in enumerate(all_labels):
            if l.isalpha():
                _off = i
                break

    all_labels = all_labels[_off:]

    singular_labels = []
    last_label, count = None, 0
    for l in all_labels:
        if (l != last_label):
            # new label, the last label is singular is count is one
            if count == 1:
                singular_labels.append(l)
            label, count = l, 1
        else:
            count += 1
    if count == 1:
        singular_labels.append(all_labels[-1])

    return ''.join(singular_labels)



def infer_output_labels(label_count, n_bcast_dims):
    '''
    Infer output labels in case no explicit output labels are given on the right hand side of '->'.
    The output labels are those that appear only once, put in alphabetical order. 
    Returns the output labels in a string
    '''
    # Broadcast labels come first
    output_labels = '.' * n_bcast_dims
    # Followed by singular labels
    singular_labels = list(l for l, cnt in label_count.items() if cnt == 1)
    output_labels += ''.join(sorted(singular_labels))

    return output_labels

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
    print(f"in labels: '{in_labels}'     out labels: '{out_labels}'")

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
    print(axes_list)
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

def plan_squeeze(plan, op, op_axes, op_shape, squeeze_axes):
    varname = f'op{op}'
    squeeze_dims = []

    # Update axes and reset mappings for squeezed dims
    for ax in squeeze_axes:
        dim = op_axes[ax]
        squeeze_dims.append(dim)
        for i, d in enumerate(op_axes):
            if d > dim:
                op_axes[i] -= 1
        op_axes[ax] = -1
        op_shape.pop(dim)
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
        for i, d in enumerate(op_axes):
            if d > dim:
                op_axes[i] -= 1
        op_axes[ax] = -1
        op_shape.pop(dim)
    # Be aware that the op label string is not updated yet...

    step = paddle.sum, [varname], varname, reduce_dims
    plan.add_step(step)

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

    perm1 = I1_dims + J1_dims + K1_dims
    perm2 = I2_dims + J2_dims + K2_dims
    
    if any(i != dim for i, dim in enumerate(perm1)):
        print(f'perm1: {perm1}')
        step = paddle.transpose, [var1], var1, perm1
        plan.add_step(step)
        # update axes index
        for i, dim in enumerate(op1_axes):
            if dim > 0:
                new_dim = perm1.index(dim)
                op1_axes[i] = new_dim

    if any(i != dim for i, dim in enumerate(perm2)):
        print(f'perm2: {perm2}')
        step = paddle.transpose, [var2], var2, perm2
        plan.add_step(step)
        # update axes index
        for i, dim in enumerate(op2_axes):
            if dim > 0:
                new_dim = perm2.index(dim)
                op2_axes[i] = new_dim

    # As a preparation for matmul, merge multiple dimensions in J and in K
    # If I is empty, meaning no batching is needed, then matmul does vector-vector,
    # matrix-vector and matrix-matrix multiplies, depending on the two operand's shapes.
    # In this case (I is []), we don't create a dummy J dimension if J is empty.
    # However if K is empty we need to expand an extra dimension of size 1 for K.
    tmp = [var1, J1, perm1, op1_shape], [var2, J2, perm2, op2_shape]
    for var, J, perm, shape in tmp:
        new_shape = [shape[dim] for dim in perm]
        
        K_size, J_size = 1, 1
        for _ in K:
            K_size *= new_shape.pop()
        for _ in J:
            J_size *= new_shape.pop()

        new_shape += [J_size, K_size]
        step = paddle.reshape, [var], var, new_shape
        plan.add_step(step)

    # Matmul
    step = paddle.matmul, [var1, var2], var2, False, True
    plan.add_step(step)

    # The result shape is in I..., J1, J2. Let's reshape back to known dimensions
    # Note, this is static deduction, not by reading the tensor shape at runtime
    result_shape = [1] * len(I)
    for ax in I:
        dim1, dim2 = op1_axes[ax], op2_axes[ax]
        s = 1 if dim1 < 0 else op1_shape[dim1]
        s = s if dim2 < 0 else max(s, op2_shape[dim2])
        result_shape[ax] = s
    if J1:
        result_shape += [op1_shape[dim] for dim in J1_dims]
    if J2:
        result_shape += [op2_shape[dim] for dim in J2_dims]

    # Need a scalar dimension somehow
    if not result_shape:
        result_shape = [1]
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
    print(f'I: {I}   J1: {J1}    J2: {J2}   K: {K}')

    # Plan different versions of matmul based on the the shape of I, J, K
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

def plan_broadcast(plan, operands, nop_axes, ndims_bcast):
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
            line = repr((out_varname, f, *in_varnames, *args))
            if out_varname:
                self.set_var(out_varname, res)
        return res

    def execute(self):
        res = None
        for f, in_varnames, out_varname, *args in self.steps:
            print(repr((out_varname, f, *in_varnames, *args)))
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
    ndims_combine = len(label_count)
    ndims_out = len(nop_axes[0]) - ndims_combine

    # Initialize a plan with an environment
    plan = Plan()
    op_names = [f'op{i}' for i in range(nop)]
    list(map(plan.set_var, op_names, operands))
    nop_shapes = [list(op.shape) for op in operands]

    # In case no dimensions to combine, do broadcast straight across
    if not ndims_combine:
        plan_broadcast(plan, operands, nop_axes, ndims_bcast)
        return plan

    # Canonicalize by removing size-1 to-combine dimensions
    for i, op_axes, shape in zip(range(nop), nop_axes, nop_shapes):
        squeeze_axes = []
        for j, dim in enumerate(op_axes[-ndims_combine:]):
            if shape[dim] == 1:
                squeeze_axes.append(ndims_out+j)
                label_count[j] -= 1
        if squeeze_axes:
            plan_squeeze(plan, i, nop_axes[i], nop_shapes[i], squeeze_axes)

    # Reduce dimensions whose label_count == 1
    for i in range(nop):
        reduce_dims = []
        reduce_axes = []
        for j, dim in enumerate(nop_axes[i][-ndims_combine:]):
            if dim >= 0 and label_count[j] == 1:
                reduce_dims.append(dim)
                reduce_axes.append(ndims_out+j)
                label_count[j] = 0

        if reduce_dims:
            plan_reduce(plan, i, nop_axes[i], nop_shapes[i], reduce_axes)

    # Plan the summations over the operand sequence
    for i in range(nop):
        # plan a single step
        
        if i == 0:
            continue

        # We'd like to arrange the dimensions in the following way:
        # [B.... I...  J... K...]
        # [B.... I...  J... K...]
        # where B... represents broadcasting dimensions, 
        #       I... label matched and not yet to be combined dimensions, both output and not output
        #       J... label not matched dimensions and output dimensions
        #       K... label matched and should immediately combined dimensions
        # We then inspect the layout and see if the summation can be specialized.  
        # Current specialization schemes:
        #  (1) if B... I... K... not empty, and J... empty, then the summation can be turned into dot
        #  (2) if J... not empty, K... not empty, then the summation can be turned into matmul
        #  (3) otherwise, make a broadcast *

        # Resolve the summation kind: dot, matmul or *
        if not nop_shapes[i-1]:
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
    nop_labels, label_count = parse_and_count_labels(lhs, operands)

    # Diagonalize the operands which have duplicate labels
    nop_labels, operands = list(zip(*map(diagonalize, nop_labels, operands)))

    # To handle broadcasting, we should first know how many dimensions are there
    # We need to use that number to generate output labels
    # e.g. 1 for ['ij', 'i.', '.k']
    n_bcast_dims = max(map(count_bcast_dims, nop_labels, operands))

    # Parse or infer output labels. The broadcasting dimensions should be taken care of.
    # Following the Numpy's rule, the broadcasting dimensions must be present in the output. 
    if rhs is None:
        output_labels = infer_output_labels(label_count, n_bcast_dims)
    else:
        output_labels = parse_output_labels(rhs, list(label_count.keys()), n_bcast_dims)

    print(f'output_labels:  {output_labels}')

    # The rest labels need to be combined.
    for l in output_labels:
        if l in label_count:
            label_count.pop(l)

    combined_labels = ''.join(label_count.keys())
    
    print(f'labels to combine:  {combined_labels}')

    # Reorder all_labels to be consistent 
    all_labels = output_labels + combined_labels

    # Label counters for combined labels
    label_count = [label_count[l] for l in combined_labels]

    print(label_count)

    # Build global_dims_index, a data structure that maintains the mapping from all_labels
    # to the dimensions in the remained operands during the summation process.  
    f = lambda labels: dims_index(labels, all_labels)
    global_dims_index = list(map(f, nop_labels))

    # Verify that all aligned dimensions are broadcastable in size across operands
    verify_shape(global_dims_index, operands)

    # Reorder the operands and possibly reduce the summation complexity
    perm = reorder_ops(global_dims_index)

    operands = [operands[i] for i in perm]
    nop_labels = [nop_labels[i] for i in perm]
    global_dims_index = [global_dims_index[i] for i in perm]

    # Now we're ready to build up an execution plan
    args = [operands, global_dims_index, n_bcast_dims, label_count]
    plan = plan_einsum(*args)
    result = plan.execute()

    return result

if __name__ == '__main__':
    import numpy as np

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
