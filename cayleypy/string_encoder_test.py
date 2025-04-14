import math

import numpy as np
import pytest
import torch

import jax.numpy as jnp

from .string_encoder import StringEncoder


def _apply_permutation(x, p):
    return [x[p[i]] for i in range(len(p))]


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30), (10, 100)])
def test_encode_decode(code_width, n):
    num_states = 5
    s = torch.randint(0, 2 ** code_width, (num_states, n))
    enc = StringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s)
    assert s_encoded.shape == (num_states, int(math.ceil(code_width * n / 63)))
    assert torch.equal(s, enc.decode(s_encoded))

import jax

@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30), (10, 100)])
def test_permutation(code_width: int, n: int):
    num_states = 5
    s = torch.randint(0, 2 ** code_width, (num_states, n))
    perm = np.random.permutation(n)
    expected = torch.tensor([_apply_permutation(row, perm) for row in np.array(s)])
    enc = StringEncoder(code_width=code_width, n=n)
    perm_func = enc.implement_permutation(perm)
    s_encoded = enc.encode(s)
    ans = enc.decode(perm_func(s_encoded))
    assert torch.equal(ans, expected)

    jax.config.update("jax_enable_x64", True)  # TODO: remove this.
    perm_func_jax = enc.implement_permutation(perm, backend='jax')
    ans2 = enc.decode(perm_func_jax(jnp.array(s_encoded, dtype=np.int64)))
    assert torch.equal(ans2, expected)

