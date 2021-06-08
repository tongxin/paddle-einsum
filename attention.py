
def get_funcs(lib):
    def attention(X, dim, heads):
        assert len(X.shape) == 3
        assert X.shape[-1] == dim
        assert dim % heads == 0

        to_QKV = Linear(dim, dim * 3)
        W_0 = Linear(dim, dim)

        b, d, h = X.shape[0], dim // heads, heads
        scale = dim ** 0.5

        qkv = to_QKV(X).reshape([b, -1, d, 3, h])
        q, k, v = tuple(einsum('bt dkh -> k bht d', qkv))
        scaled_qk = einsum('bhi d, bhj d -> bhij', q, k) * scale
        attention = softmax(scaled_qk, axis=-1)
        out = einsum('bhi j, bhj d -> bihd', attention, v).reshape([b, -1, h*d])
    
        return W_0(out)

    assert lib in {'torch', 'paddle'}
    if lib == 'torch':
        import torch
        Linear = lambda x, y: torch.nn.Linear(x, y, bias=False).cuda('cuda:0')
        softmax = torch.softmax
        einsum = torch.einsum
        to_tensor = lambda x: torch.tensor(x, dtype=torch.double, device='cuda:0')
    if lib == 'paddle':
        import paddle
        from einsum import einsum
        # from nlp_einsum import einsum
        Linear = lambda x, y: paddle.nn.Linear(x, y)
        softmax = paddle.nn.functional.softmax
        to_tensor = lambda x: paddle.to_tensor(x, dtype='float32')
    return to_tensor, attention

if __name__ == '__main__':
    import numpy as np
    from time import time

    X_ = np.random.rand(100, 256, 256)

    to_tensor, attention = get_funcs('paddle')
    X = to_tensor(X_)
    X.sum()
    ts = []
    for _ in range(3):
        t = time()
        attention(X, 256, 16)
        ts.append(time() - t)
    
    print(f'Paddle time: {min(ts)}')

    # import torch
    # to_tensor, attention = get_funcs('torch')
    # X = to_tensor(X_)
    # t = time()
    # print(attention(X, 256, 16).shape)
    # print(f'Torch time: {time() - t}')
