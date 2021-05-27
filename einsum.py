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
    
    nop_labels = labelstr.split(',')
    assert len(nop_labels) == len(operands), \
        f"Invalid equation: the number of operands is {len(operands)} but only found {len(nop_labels)} in the label string."
    
    nop_labels = list(map(parse_op_labels, nop_labels, operands))

    for labels in nop_labels:
        for c in set(labels.replace('.', '')):
            count[c] += 1

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
    assert all(c in avail_labels for c in rhs), f"Invalid equation: output labels must come from the input labels. "

    # Syntax check. Verify there's no duplicate alphabetical labels
    for i, l in enumerate(rhs.replace('.', '')):
        if rhs.find(l, 0, i) >= 0:
            assert False, f"Invalid equation: duplicate output label {l}."

    # Check there's no dots other than in an ellipsis
    assert rhs.replace('...', '', 1).find('.') == -1, \
        f"Invalid equation: `.` is only expected to be included in an ellipsis."
    
    # Check if ellipsis is missing
    assert n_bcast_dims > 0 == rhs.find('...') >= 0, \
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

    map(labelset.update, labels_list)
    
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
    inv_map = [-1] * len(out_labels)
    
    # First build the broadcast dimension mapping
    # Find the broadcast index range in out_labels
    r = re.search('\.+', out_labels)
    if r:
        start, end = r.start(), r.end()
        s = re.search('\.+', in_labels)
        # fill the broadcast dimension indices from right to left.
        if s:
            inv_map[end:start:-1] = range(s.end(), s.start())
        
    # Now work on non-broadcast dimensions 
    if start:
        it = itertools.chain(range(start), range(end, len(out_labels)))
    else:
        it = iter(range(len(out_labels)))
        
    for i in it:
        inv_map[i] = in_labels.find(out_labels[i])

    return inv_map

def verify_shape(axes_list, operands):
    op_shapes = [op.shape for op in operands]
    for axes in zip(*axes_list):
        # axes are a column of nop input dimension axes. -1 represents new axis
        # all non size-one dimensions must have the same size
        sizes, ops, op_axes = [], [], []
        for axis, shape, op in zip(axes, op_shapes, ops):
            if axis > 0 and shape[axis] > 1:
                sizes.append(shape[axis])
                ops.append(op)
                op_axes.append(axis)

        for s1, s2, ax1, ax2, op1, op2 in zip(sizes, sizes[1:], op_axes, op_axes[1:], ops, ops[1:]):
            assert s1 == s2, f'Dimension {ax1} in {op1.name} and dimension {ax2} in {op2.name} do not match in size.'


def plan_reduce(plan, op, op_axes, op_shape, reduce_axes):
    '''
    Add reduce to the plan
    '''
    varname = f'op{op}'
    reduce_dims = [op_axes[ax] for ax in reduce_axes]
    step = paddle.sum, [varname], varname, reduce_dims
    plan.add_step(step)
    # Update axes index
    for ax in reduce_axes:
        op_axes[ax] = -1
    for dim in reduce_dims:
        op_shape.pop(dim)

def plan_matmal(plan, op1, op2, op1_axes, op2_axes, op1_shape, op2_shape, I, J1, J2, K):
    '''
    plan matmul
    '''
    # Transpose and re-shape op1 and op2 in I, J1, K and I, J2, K
    # Then apply matmul(x, y, transpose_x=False, tranpose_y=True)
    perm1 = [op1_axes[ax] for ax in I + J1 + K]
    perm2 = [op2_axes[ax] for ax in I + J2 + K]
    
    var1, var2 = f'op{op1}', f'op{op2}'
    step = paddle.transpose, [var1], var1, perm1
    plan.add_step(step)

    step = paddle.transpose, [var2], var2, perm2
    plan.add_step(step)

    # Reshape and merge J and K into single dimensions
    for var, perm, J, shape in zip([var1, var2], [J1, J2], [perm1, perm2], [op1_shape, op2_shape]):
        new_shape = [shape[dim] for dim in perm]
        K_size = sum(new_shape.pop() for _ in K)
        J_size = sum(new_shape.pop() for _ in J)
        K_size = 1 if K_size == 0 else K_size
        new_shape += [J_size, K_size]

        step = paddle.reshape, [var], var, new_shape
        plan.add_step(step)

    # Matmul
    step = paddle.matmul, [var1, var2], var2, False, True
    plan.add_step(step)

    # The result shape is in I..., J1, J2. Let's reshape back to known dimensions
    # Note, this is static deduction, not by reading the tensor shape at runtime
    result_shape = [max(op1_shape[dim1], op2_shape[dim2]) for dim1, dim2 in zip(perm1[:len(I)], perm2[:len(I)])]
    result_shape += [op1_shape[dim] for dim in perm1[len(I):-len(K)]]
    result_shape += [op2_shape[dim] for dim in perm2[len(I):-len(K)]]

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

def plan_summation(plan, ops, nop_axes, nop_shapes, op1, op2, label_count):
    '''
    Plan various kinds of summation
    '''

    op1_axes, op2_axes = [nop_axes[op] for op in (op1, op2)]
    op1_shape, op2_shape = [nop_shapes[op] for op in (op1, op2)]

    ndims = len(op1_axes)
    ndims_out = ndims - len(label_count)

    count = [0] * ndims_out + label_count

    I, K, J1, J2 = [], [], [], []
    op1_reduce_axes, op2_reduce_axes = [], []

    for i, dim1, dim2, in zip(range(ndims), op1_axes, op2_axes, count):
        if (dim1 != -1) != (dim2 != -1):
            if dim1 != -1:
                J1.append(dim1)
            else:
                J2.append(dim2)
        elif dim1 != -1:
            if count[i] == 2:
                shape = op1_shape[dim1], op2_shape[dim2]
                if shape[0] != shape[1]:
                    if shape[0] != 1:
                        op1_reduce_axes.append(i)
                    else:
                        op2_reduce_axes.append(i)
                else:
                    K.append(i)
                # Either case, kill this axis
                count[i] = 0       
            else:
                I.append(i)
                # Decrement count
                if i >= ndims_out:
                    label_count[i - ndims_out] -= 1

    # Reduce the K dimensions
    # Two side effects caused by the reduce's:
    #   1) the killed dims will be replaced with -1 in the axes array,
    #   2) the shape array will be shrinked 
    if op1_reduce_axes:
        plan_reduce(plan, op1, op1_axes, op1_shape, op1_reduce_axes)
        
    if op2_reduce_axes:
        plan_reduce(plan, op2, op2_axes, op2_shape, op1_reduce_axes)

    # Now it's OK to merge the K dims as the same shape holds

    # Plan different versions of matmul based on the the shape of I, J, K
    plan_matmal(plan, op1, op2, op1_axes, op2_axes, op1_shape, op2_shape, I, J1, J2, K)

def labels_to_axes(labelstr, labels):
    return [i for i, l in enumerate(labelstr) if l in labels]

class Plan:
    def __init__(self):
        env = {}
        steps = []

    def add_step(self, step):
        self.steps.append(step)

    def get_var(self, varname):
        return self.env[varname] if varname in self.env else None

    def set_var(self, varname, var):
        self.env[varname] = var

    def execute(self):
        env = self.env
        for f, in_varnames, out_varname, *args in self.steps:
            res = f(*map(self.get_var, in_varnames), *args)
            if out_varname:
                self.set_var(out_varname, res)
        return self.env['result']

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

def plan_einsum(operands, nop_axes, ndims_combine, label_count):
    '''
    Plans the actual execution steps.

    Results
    -------
    the execution plan
    '''
    nop_shapes = [op.shape for op in operands]
    nop = len(operands)

    plan = Plan()

    # Check if there are dimensions ready for reduce, i.e. label_count == 1
    for i in range(nop):
        reduce_dims = []
        reduce_axes = []
        for j, dim in enumerate(nop_axes[i][-ndims_combine:]):
            if label_count[j] == 1:
                reduce_dims.append(dim)
                reduce_axes.append(j)
                label_count[j] = 0

        if reduce_dims:
            # Add reduce to the plan
            step = paddle.sum, [], f'op{i}', reduce_dims
            plan.add_step(step)
            # Update axes index
            for ax in reduce_axes:
                nop_axes[ax] = -1

    # Plan the summations over the operand sequence
    for i in range(nop):
        # plan a single step
        
        if i == 1:
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
        plan_summation(plan, operands, nop_axes, nop_shapes, i-1, i, label_count)

    return plan

def einsum(equation, *operands):
    r"""
    Executes the sum of product of provided operands based on the Einstein summation convention.
    Einsum can be used to complete a variety of operations, such as sum, transpose,
    batch matrix multiplication.

    Args:
        equation (`str`):
            The equation uses uncased letters to indicate the dimensions for summation. 
            These letters are called dimension labels or dimension subscripts. The dimension labels
            are comma separated to correspond to all the input operands. The equation uses `->` to indicate 
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

    # First off, turn all the letters to lower case and remove white spaces
    equation = equation.lower().replace(' ', '')

    # Part the equation to the left hand side and the right hand side of ->
    lhs, *rhs = equation.split('->')

    assert len(rhs) < 2, "Invalid equation: multiple `->` were found."
    rhs = rhs[0] if rhs else ''

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
    if rhs:
        output_labels = parse_output_labels(rhs, list(label_count.keys()), n_bcast_dims)
    else:
        output_labels = infer_output_labels(nop_labels, n_bcast_dims)

    # The rest labels need to be combined.
    combined_labels = label_count.keys()
    for l in output_labels:
        combined_labels.remove(l)
    combined_labels = ''.join(combined_labels)

    ndims_combined = len(combined_labels)
    
    # Reorder all_labels to be consistent 
    all_labels = output_labels + combined_labels

    # Label counters for combined labels
    label_count = [label_count[l] for l in combined_labels]

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
    args = [operands, global_dims_index, ndims_combined, label_count]
    plan = plan_einsum(*args)
    result = plan.execute()

    return result


