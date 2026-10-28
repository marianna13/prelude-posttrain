"""Exact native MD5[:8] benchmark scans, compatible with MV Unicode word semantics."""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np


def gram_hash(text):
    return int.from_bytes(hashlib.md5(text.encode('utf-8','ignore')).digest()[:8], 'big')


def canonical_words(text):
    return ' '.join(text.lower().split()).encode('utf-8','ignore')


class Scanner:
    def __init__(self, index, library, *, n=None, min_distinct=7):
        index = Path(index)
        self.hashes = np.load(index/'ngrams.npy', mmap_mode='r')
        if self.hashes.dtype != np.dtype('uint64') or not self.hashes.flags.c_contiguous:
            raise ValueError('index must be native uint64')
        if not np.all(self.hashes[1:] > self.hashes[:-1]):
            raise ValueError('index must be sorted and unique')
        self.n = n or json.loads((index/'manifest.json').read_text())['n']
        self.min_distinct = min_distinct
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.function = self.library.prelude_scan
        pointer = ctypes.c_void_p
        self.function.argtypes = [ctypes.c_char_p, ctypes.c_size_t, pointer, ctypes.c_size_t,
                                  ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t,
                                  pointer, pointer, pointer, pointer, pointer]
        self.function.restype = ctypes.c_size_t

    def scan(self, text):
        canonical = canonical_words(text)
        capacity = 1024
        while True:
            hashes, starts, ends = (np.empty(capacity,dtype=np.uint64) for _ in range(3))
            eligible = np.empty(capacity,dtype=np.uint8)
            evaluated = np.zeros(1,dtype=np.uint64)
            count = self.function(canonical,len(canonical),self.hashes.ctypes.data,len(self.hashes),
                                  self.n,self.min_distinct,capacity,hashes.ctypes.data,starts.ctypes.data,
                                  ends.ctypes.data,eligible.ctypes.data,evaluated.ctypes.data)
            if count == ctypes.c_size_t(-1).value: raise RuntimeError('native scanner failed')
            if count <= capacity: break
            capacity = count
        hits = {int(hashes[i]): {'eligible':bool(eligible[i]),
                'ngram':canonical[int(starts[i]):int(ends[i])].decode('utf-8')}
                for i in range(count)}
        return hits, int(evaluated[0])
