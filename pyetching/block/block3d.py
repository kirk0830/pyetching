import unittest
import numpy as np
import matplotlib.pyplot as plt

class Block3D:
    
    def __init__(self, r, a, b=None, c=None):
        '''instantiate a block for 3D MC modeling
        etching process.
        
        Parameters
        ----------
        r : np.ndarray
            the center coordinate of the block
        a : float
            the edge length or height of the block
        b : float
            the width of the block
        c : float
            the depth of the block
        '''
        # sanity check
        assert isinstance(r, np.ndarray)
        assert isinstance(a, float)
        assert b is None or isinstance(b, float)
        assert c is None or isinstance(c, float)
        assert r.shape == (3,)
        assert a > 0
        assert b is None or b > 0
        assert c is None or c > 0

        self.r = r
        self.a = a
        self.b = a if b is None else b
        self.c = a if c is None else c

