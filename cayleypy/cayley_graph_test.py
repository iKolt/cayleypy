import torch
import scipy

from cayleypy import CayleyGraph


def _last_layer_to_str(layer):
    return set(''.join(str(int(x)) for x in state) for state in layer)


def _lrx_permutations(n):
    """Generates 3 permutations from S_N: left shift, right shift and swapping first two elements."""
    return [list(range(1, n)) + [0], [n - 1] + list(range(0, n - 1)), [1, 0] + list(range(2, n))]


def test_bfs_growth_lrx_coset():
    # Tests growth starting from string 00..0 11..1 for even N.
    test_cases = [
        (2, [1, 1], {'10'}),
        (4, [1, 2, 3], {'1100', '1010', '0101'}),
        (6, [1, 2, 3, 4, 4, 3, 2, 1], {'010101'}),
        (8, [1, 2, 3, 4, 6, 8, 7, 9, 10, 9, 8, 2, 1], {'01010101'}),
        (10, [1, 2, 3, 4, 6, 8, 10, 13, 14, 17, 23, 25, 26, 25, 23, 21, 16, 11, 4],
         {'1101001100', '0101010101', '0100110011', '0011001101'}),
        (12, [1, 2, 3, 4, 6, 8, 10, 14, 18, 20, 26, 34, 41, 55, 55, 68, 69, 68, 81, 72, 71, 62, 46, 45, 27, 14, 4],
         {'010110010110', '100011010011', '010101010101', '011001011001'}),
        (14,
         [1, 2, 3, 4, 6, 8, 10, 14, 17, 22, 29, 32, 44, 58, 70, 90, 104, 120, 143, 155, 171, 193, 201, 210, 215, 214,
          218, 203, 190, 186, 151, 126, 107, 68, 36, 11],
         {'01010101010101', '11010100110100', '01010011010011', '00110010110011', '10011001100110', '11000110101001',
          '00011010100111', '11001100101100', '01001101001101', '10100110011001', '00110100110101'}),
    ]

    for n, expected_layer_sizes, expected_last_layer in test_cases:
        graph = CayleyGraph(_lrx_permutations(n))
        start_states = torch.tensor([[0] * (n // 2) + [1] * (n // 2)])
        result = graph.bfs_growth(start_states)
        assert result.layer_sizes == expected_layer_sizes
        assert result.diameter == len(result.layer_sizes)
        assert sum(result.layer_sizes) == scipy.special.comb(n, n // 2, exact=True)
        assert _last_layer_to_str(result.last_layer) == expected_last_layer


def _top_spin_permutations(n):
    """Generates 3 permutations from S_N: left shift, right shift and reversing first 4 elements."""
    return [list(range(1, n)) + [0], [n - 1] + list(range(0, n - 1)), [3, 2, 1, 0] + list(range(4, n))]


def test_bfs_growth_top_spin_coset():
    # Tests growth starting from string 00..0 11..1 for even N.
    test_cases = [
        (4, [1, 3], {'1100', '0110', '1001'}),
        (6, [1, 2, 3, 4, 4, 3, 2, 1], {'010101'}),
        (8, [1, 2, 4, 8, 13, 12, 16, 4, 4, 4, 2], {'10101010', '01010101'}),
        (10, [1, 2, 3, 6, 11, 17, 22, 27, 30, 34, 36, 27, 17, 13, 5, 1], {'1010101010'}),
        (12, [1, 2, 3, 5, 10, 15, 28, 37, 54, 77, 106, 133, 113, 102, 99, 66, 35, 18, 14, 4, 2],
         {'101010101010', '100101010110'}),
        (14, [1, 2, 3, 5, 9, 15, 28, 44, 62, 88, 134, 202, 259, 317, 374, 431, 459, 365, 258, 181, 118, 46, 18, 11, 2],
         {'10101010101010', '01010101010101'}),
    ]

    for n, expected_layer_sizes, expected_last_layer in test_cases:
        graph = CayleyGraph(_top_spin_permutations(n))
        start_states = torch.tensor([[0] * (n // 2) + [1] * (n // 2)])
        result = graph.bfs_growth(start_states)
        assert result.layer_sizes == expected_layer_sizes
        assert result.diameter == len(result.layer_sizes)
        if n >= 6:
            assert sum(result.layer_sizes) == scipy.special.comb(n, n // 2, exact=True)
        assert _last_layer_to_str(result.last_layer) == expected_last_layer


def test_bfs_growth_max_layers():
    graph = CayleyGraph(_lrx_permutations(10))
    start_states = torch.tensor([[0] * 5 + [1] * 5])
    result1 = graph.bfs_growth(start_states, max_layers=5)
    assert result1.layer_sizes == [1, 2, 3, 4, 6]
    assert len(result1.last_layer) == 6
    result2 = graph.bfs_growth(start_states, max_layers=7)
    assert result2.layer_sizes == [1, 2, 3, 4, 6, 8, 10]
    assert len(result2.last_layer) == 10
