"""Small, dependency-free reproduction of R's default RNG for ``sample()``.

The article graphs in ``simulation/simulation.R`` are selected with
``set.seed()`` and ``sample()``.  R >= 3.6 uses Mersenne-Twister plus the
"Rejection" sampler by default.  This module implements only that narrow
piece, so the Python port can reconstruct the article's eight true graphs.
"""

from __future__ import annotations

import math
from typing import Sequence, TypeVar

T = TypeVar("T")
_UINT32_MASK = (1 << 32) - 1
_UNIF_SCALE = 2.3283064365386963e-10  # 1 / 2**32, as used by R


class RRandom:
    """R-compatible Mersenne-Twister stream and sampling without replacement."""

    def __init__(self, seed: int):
        value = int(seed) & _UINT32_MASK
        for _ in range(50):
            value = (69069 * value + 1) & _UINT32_MASK
        initialized: list[int] = []
        for _ in range(625):
            value = (69069 * value + 1) & _UINT32_MASK
            initialized.append(value)
        self._state = initialized[1:]
        self._position = 624

    def _uint32(self) -> int:
        if self._position >= 624:
            mag01 = (0, 0x9908B0DF)
            for index in range(227):
                y = (self._state[index] & 0x80000000) | (
                    self._state[index + 1] & 0x7FFFFFFF
                )
                self._state[index] = (
                    self._state[index + 397] ^ (y >> 1) ^ mag01[y & 1]
                ) & _UINT32_MASK
            for index in range(227, 623):
                y = (self._state[index] & 0x80000000) | (
                    self._state[index + 1] & 0x7FFFFFFF
                )
                self._state[index] = (
                    self._state[index - 227] ^ (y >> 1) ^ mag01[y & 1]
                ) & _UINT32_MASK
            y = (self._state[623] & 0x80000000) | (
                self._state[0] & 0x7FFFFFFF
            )
            self._state[623] = (
                self._state[396] ^ (y >> 1) ^ mag01[y & 1]
            ) & _UINT32_MASK
            self._position = 0

        y = self._state[self._position]
        self._position += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & _UINT32_MASK

    def uniform(self) -> float:
        return self._uint32() * _UNIF_SCALE

    def _random_bits(self, bits: int) -> int:
        value = 0
        for _ in range(0, bits + 1, 16):
            value = 65536 * value + math.floor(self.uniform() * 65536)
        return value & ((1 << bits) - 1)

    def index(self, size: int) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        bits = math.ceil(math.log2(size))
        while True:
            value = self._random_bits(bits)
            if value < size:
                return value

    def sample(self, population: Sequence[T], size: int) -> list[T]:
        """Match ``sample(population, size, replace = FALSE)``."""
        if size < 0 or size > len(population):
            raise ValueError("invalid sample size")
        pool = list(population)
        remaining = len(pool)
        answer: list[T] = []
        for _ in range(size):
            index = self.index(remaining)
            answer.append(pool[index])
            remaining -= 1
            pool[index] = pool[remaining]
        return answer
