import numpy as np
import paddle
# import torch
from einsum import einsum
from nlp_einsum import einsum as nlp_einsum

import time

def timeit(code_snippets):
    for module, code in code_snippets.items():
        print(eval(code))
        ts = []
        for _ in range(3):
            t = time.time()
            eval(code)
            ts.append(time.time() - t)
        
        print(f'{module:5}   {min(ts)}')

def bench_sum():
    code_snippets = {  \
        'np': r"np.einsum('ijk->', x).shape", \
        'my': r"einsum('ijk->', px).shape", \
        'nlp': r"nlp_einsum('ijk->', px).shape"
    }
    print(r"times for 'ijk->'")
    timeit(code_snippets)

def bench_sum1():
    code_snippets = {  \
        'np': r"np.einsum('ijk->j', x).shape", \
        'my': r"einsum('ijk->j', px).shape", \
        'sum_all': r"px.sum(0, 2).shape", \
        'sum_dim': r"px.sum(2, keepdim=True).sum(0, keepdim=True).squeeze().shape", \
        'nlp': r"nlp_einsum('ijk->j', px).shape"
    }
    print(r"times for 'ijk->j'")
    timeit(code_snippets)

def bench_transpose():
    code_snippets = {  \
        'np': r"np.einsum('ijk->jki', x).shape", \
        'my': r"einsum('ijk->jki', px).shape", \
        'nlp': r"nlp_einsum('ijk->jki', px).shape", \
        # 'torch': r"torch.einsum('ijk->jki', tx).shape"
    }
    print(r"times for 'ijk->jki'")
    timeit(code_snippets)

def bench_matrix_vector():
    code_snippets = {  \
        'np': r"np.einsum('ikj,k', x, y).shape", \
        'my': r"einsum('ikj,k', px, py).shape", \
        'nlp': r"nlp_einsum('ikj,k', px, py).shape", \
        # 'torch': r"torch.einsum('ikj,k', tx, ty).shape"
    }
    print(r"times for 'ikj,k'")
    timeit(code_snippets)

def bench_bmm():
    code_snippets = {  \
        'np': r"np.einsum('ikj,kg', x, y).shape", \
        'my': r"einsum('ikj,kg', px, py).shape", \
        'nlp': r"nlp_einsum('ikj,kg', px, py).shape", \
        # 'torch': r"torch.einsum('ikj,kg', tx, ty).shape"
    }
    print(r"times for 'ikj,kg'")
    timeit(code_snippets)

if __name__ == '__main__':
    global x, px, tx, y, py, ty
    np_x_large, np_y_large = np.random.rand(1000, 100, 10), np.random.rand(1000, 100)
    np_x_ultra_large = np.random.rand(10000, 1000, 10)

    x = np_x_ultra_large
    px = paddle.to_tensor(x)
    # tx = torch.tensor(x)
    # tx = torch.tensor(x, device='cuda:0')
    bench_sum()
    bench_sum1()
    bench_transpose()

    y = np.random.rand(1000)
    py = paddle.to_tensor(y)
    # ty = torch.tensor(y)
    # ty = torch.tensor(y, device='cuda:0')
    bench_matrix_vector()

    y = np_y_large
    py = paddle.to_tensor(y)
    # ty = torch.tensor(y)
    # ty = torch.tensor(y, device='cuda:0')
    bench_bmm()
