import paddle
from einsum import einsum
# from nlp_einsum import einsum

def attention(X, dim, heads):
    assert len(X.shape) == 3
    assert X.shape[-1] == dim
    assert dim % heads == 0

    to_QKV = paddle.nn.Linear(dim, dim * 3)
    W_0 = paddle.nn.Linear(dim, dim)

    b, d, h = X.shape[0], dim // heads, heads
    scale = dim ** 0.5

    qkv = to_QKV(X).reshape([b, -1, d, 3, h])
    q, k, v = tuple(einsum('bt dkh -> k bht d', qkv))
    scaled_qk = einsum('bhi d, bhj d -> bhij', q, k) * scale

    attention = paddle.nn.functional.softmax(scaled_qk, axis=-1)
    out = einsum('bhi j, bhj d -> bihd', attention, v).reshape([b, -1, h*d])
    
    return W_0(out)

if __name__ == '__main__':
    from time import time
    X = paddle.rand([100, 256, 256])
    
    t = time()
    print(attention(X, 256, 16).shape)
    print(f'Time: {time() - t}')
