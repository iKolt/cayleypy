import math
from typing import Callable, Sequence

import torch

# We are using int64, but avoid using the sign bit.
CODEWORD_LENGTH = 63


class StringEncoder:
    """Helper class to encode strings that represent elements of coset.

    Original (decoded) strings are 2D tensors where tensor elements are integers representing elements being permuted.
    In encoded format, these elements are compressed to take less memory. Each element takes only `code_width` bits.
    For binary strings (`code_width=1`) and `n<=63`, this allows to represent coset element with a single int64 number.
    Elements in the original string must be in range `[0, 2**code_width)`.
    This class also provides functionality to efficiently apply permutation in encoded format using bit operations.
    """

    def __init__(self, *, code_width: int = 1, n: int = 1):
        """
        Initializes StringEncoder.

        Args:
            code_width: Number of bits to encode one element of coset.
            string_length: Length of the string. Defaults to 1.
        """
        assert 1 <= code_width <= CODEWORD_LENGTH
        self.w = code_width
        self.n = n
        self.encoded_length = int(math.ceil(self.n * self.w / CODEWORD_LENGTH))  # Encoded length.

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        """Encodes tensor of coset elements.

        Input shape `(m, self.n)`. Output shape `(m, self.encoded_length)`.
        """
        assert len(s.shape) == 2
        assert s.shape[1] == self.n
        assert torch.min(s) >= 0, "Cannot encode negative values."
        max_value = torch.max(s)
        assert max_value < 2 ** self.w, f"Width {self.w} is not sufficient to encode value {max_value}."

        encoded = torch.zeros((s.shape[0], self.encoded_length), dtype=torch.int64)
        w, cl = self.w, CODEWORD_LENGTH
        for i in range(w * self.n):
            encoded[:, i // cl] |= ((s[:, i // w] >> (i % w)) & 1) << (i % cl)
        return encoded

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decodes tensor of coset elements.

        Input shape `(m, self.encoded_length)`. Output shape `(m, self.n)`.
        """
        orig = torch.zeros((encoded.shape[0], self.n), dtype=torch.int64)
        w, cl = self.w, CODEWORD_LENGTH
        for i in range(w * self.n):
            orig[:, i // w] |= ((encoded[:, i // cl] >> (i % cl)) & 1) << (i % w)
        return orig

    def implement_permutation(self, p: Sequence[int]) -> Callable[[torch.Tensor], torch.Tensor]:
        """Converts permutation to a function on encoded tensor implementing this permutation."""
        assert len(p) == self.n
        shift_to_mask: dict[tuple[int, int, int], float] = dict()
        for i in range(self.n):
            for j in range(self.w):
                start_bit = p[i] * self.w + j
                end_bit = i * self.w + j
                start_cw_id = start_bit // CODEWORD_LENGTH
                end_cw_id = end_bit // CODEWORD_LENGTH
                shift = (end_bit % CODEWORD_LENGTH) - (start_bit % CODEWORD_LENGTH)
                key = (start_cw_id, end_cw_id, shift)
                if key not in shift_to_mask:
                    shift_to_mask[key] = 0
                shift_to_mask[key] |= (1 << (start_bit % CODEWORD_LENGTH))

        lines = ["def f_(x):", " ans=torch.zeros_like(x)"]
        for (start_cw_id, end_cw_id, shift), mask in shift_to_mask.items():
            line = f" ans[:,{end_cw_id}] |= (x[:,{start_cw_id}] & 0b{mask:b})"
            if shift > 0:
                line += f"<<{shift}"
            elif shift < 0:
                line += f">>{-shift}"
            lines.append(line)
        lines += [" return ans"]
        src = "\n".join(lines)
        l = {}
        exec(src, {"torch": torch}, l)
        return l["f_"]
