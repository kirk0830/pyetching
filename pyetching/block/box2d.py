'''
the box that containes blocks for 2D MC modeling
'''
import unittest
import itertools as it

import numpy as np
import matplotlib.pyplot as plt

class Box2D:
    '''the base class of box'''
    def __init__(self, a, b):
        '''instantiate a box for 2D MC modeling
        etching process.
        
        Parameters
        ----------
        a : float
            the length of the box
        b : float
            the width of the box
        '''
        # sanity check
        assert isinstance(a, (int, float))
        assert isinstance(b, (int, float))
        assert a > 0
        assert b > 0

        self.a = a
        self.b = b

class TestBox2D(unittest.TestCase):
    def test_init(self):
        a = 10
        b = 20
        box = Box2D(a, b)
        self.assertEqual(box.a, a)
        self.assertEqual(box.b, b)

class BlockBox2D(Box2D):
    '''the box that containes blocks for 2D MC modeling
    etching process.
    '''
    def __init__(self, a, b, nbx, nby, pbc=True):
        '''instantiate a box containing blocks for 2D MC modeling
        etching process.
        
        Parameters
        ----------
        a : float
            the length of the box
        b : float
            the width of the box
        nbx : int
            the number of blocks in x direction
        nby : int
            the number of blocks in y direction
        pbc : bool
            Periodic boundary condition. Default is True.
        '''
        super().__init__(a, b)
        
        assert isinstance(nbx, int)
        assert isinstance(nby, int)
        assert nbx > 0
        assert nby > 0
        
        self.nbx = nbx
        self.nby = nby
        
        self.pbc = pbc
        
        dx = a / nbx
        dy = b / nby
        self.centers = list(it.product(np.arange(nbx) + dx / 2, np.arange(nby) + dy / 2))
        self.centers = np.array(self.centers)
    
    def i2xy(self, i):
        '''indexing from the flattened index to the 2D index'''
        assert isinstance(i, int)
        assert i >= 0
        assert i < self.nbx * self.nby
        
        iy = i // self.nbx
        ix = i - iy * self.nbx
        return ix, iy
    
    def xy2i(self, ix, iy):
        '''indexing from the 2D index to the flattened index'''
        assert isinstance(ix, int)
        assert isinstance(iy, int)
        assert ix >= 0
        assert ix < self.nbx
        assert iy >= 0
        assert iy < self.nby
        
        return iy * self.nbx + ix
    
    def left(self, i):
        '''get the left neighbor of block i'''
        ix, iy = self.i2xy(i)
        ix -= 1
        if ix < 0:
            if self.pbc:
                ix = self.nbx - 1
            else:
                return None
        return self.xy2i(ix, iy)
    
    def right(self, i):
        '''get the right neighbor of block i'''
        ix, iy = self.i2xy(i)
        ix += 1
        if ix >= self.nbx:
            if self.pbc:
                ix = 0
            else:
                return None
        return self.xy2i(ix, iy)
    
    def up(self, i):
        '''get the up neighbor of block i'''
        ix, iy = self.i2xy(i)
        iy -= 1
        if iy < 0:
            if self.pbc:
                iy = self.nby - 1
            else:
                return None
        return self.xy2i(ix, iy)
    
    def down(self, i):
        '''get the down neighbor of block i'''
        ix, iy = self.i2xy(i)
        iy += 1
        if iy >= self.nby:
            if self.pbc:
                iy = 0
            else:
                return None
        return self.xy2i(ix, iy)
    
    def allocate(self, f):
        '''allocate the block with the function f
        
        Parameters
        ----------
        f : callable
            the function to allocate the block
        '''
        assert callable(f)
        
        self.blocks = [f(r=center, a=self.a / self.nbx, b=self.b / self.nby) \
            for center in self.centers]

class TestBlockBox2D(unittest.TestCase):
    def test_init(self):
        a = 10
        b = 20
        nbx = 5
        nby = 10
        pbc = True
        box = BlockBox2D(a, b, nbx, nby, pbc)
        self.assertEqual(box.a, a)
        self.assertEqual(box.b, b)
        
        self.assertEqual(box.nbx, nbx)
        self.assertEqual(box.nby, nby)
        
        self.assertEqual(box.pbc, pbc)
        
        self.assertEqual(len(box.centers), nbx * nby)
        for center in box.centers:
            self.assertEqual(len(center), 2)
            self.assertTrue(center[0] >= 0)
            self.assertTrue(center[0] <= a)
            self.assertTrue(center[1] >= 0)
            self.assertTrue(center[1] <= b)
        
        for ix, iy in it.product(range(nbx), range(nby)):
            i = box.xy2i(ix, iy)
            self.assertEqual((ix, iy), box.i2xy(i))
            
    def test_i2xy(self):
        a = 10
        b = 20
        nbx = 5
        nby = 10
        pbc = True
        box = BlockBox2D(a, b, nbx, nby, pbc)
        for i in range(nbx * nby):
            ix, iy = box.i2xy(i)
            self.assertEqual(i, box.xy2i(ix, iy))
        
    def test_left(self):
        a = 10
        b = 20
        nbx = 5
        nby = 10
        pbc = True
        box = BlockBox2D(a, b, nbx, nby, pbc)
        for i in range(nbx * nby):
            ix, iy = box.i2xy(i)
            if ix == 0:
                if pbc:
                    self.assertEqual(i, box.right(box.left(i)))
                else:
                    self.assertIsNone(box.left(i))
            else:
                self.assertEqual(i, box.right(box.left(i)))
    
    def test_right(self):
        a = 10
        b = 20
        nbx = 5
        nby = 10
        pbc = True
        box = BlockBox2D(a, b, nbx, nby, pbc)
        for i in range(nbx * nby):
            ix, iy = box.i2xy(i)
            if ix == nbx - 1:
                if pbc:
                    self.assertEqual(i, box.left(box.right(i)))
                else:
                    self.assertIsNone(box.right(i))
            else:
                self.assertEqual(i, box.left(box.right(i)))
                
    def test_up(self):
        a = 10
        b = 20
        nbx = 5
        nby = 10
        pbc = True
        box = BlockBox2D(a, b, nbx, nby, pbc)
        for i in range(nbx * nby):
            ix, iy = box.i2xy(i)
            if iy == 0:
                if pbc:
                    self.assertEqual(i, box.down(box.up(i)))
                else:
                    self.assertIsNone(box.up(i))
            else:
                self.assertEqual(i, box.down(box.up(i)))
                
    def test_down(self):
        a = 10
        b = 20
        nbx = 5
        nby = 10
        pbc = True
        box = BlockBox2D(a, b, nbx, nby, pbc)
        for i in range(nbx * nby):
            ix, iy = box.i2xy(i)
            if iy == nby - 1:
                if pbc:
                    self.assertEqual(i, box.up(box.down(i)))
                else:
                    self.assertIsNone(box.down(i))
            else:
                self.assertEqual(i, box.up(box.down(i)))
                    
    def test_allocate(self):
        from pyetching.block.block2d import \
            Block2D, ElasticBlock2D, PermeableBlock2D, WeightedBlock2D,\
                PhysicalScatterBlock2D
                
        box = BlockBox2D(a=10, b=20, nbx=5, nby=10)
        def f(r, a, b):
            return Block2D(r, a, b)
        box.allocate(f)
        self.assertEqual(len(box.blocks), 50)
        for block in box.blocks:
            self.assertIsInstance(block, Block2D)
            self.assertTrue(block.r in box.centers)
            self.assertEqual(block.a, box.a / box.nbx)
            self.assertEqual(block.b, box.b / box.nby)
        
        def f(r, a, b):
            return ElasticBlock2D(r, a, b, f=1.0)
        box.allocate(f)
        self.assertEqual(len(box.blocks), 50)
        for block in box.blocks:
            self.assertIsInstance(block, ElasticBlock2D)
            self.assertTrue(block.r in box.centers)
            self.assertEqual(block.a, box.a / box.nbx)
            self.assertEqual(block.b, box.b / box.nby)
            self.assertEqual(block.f, 1.0)
            
        def f(r, a, b):
            return PermeableBlock2D(r, a, b, f=1.0)
        box.allocate(f)
        self.assertEqual(len(box.blocks), 50)
        for block in box.blocks:
            self.assertIsInstance(block, PermeableBlock2D)
            self.assertTrue(block.r in box.centers)
            self.assertEqual(block.a, box.a / box.nbx)
            self.assertEqual(block.b, box.b / box.nby)
            self.assertEqual(block.f, 1.0)
            
        def f(r, a, b):
            return WeightedBlock2D(r, a, b, w=1.0)
        box.allocate(f)
        self.assertEqual(len(box.blocks), 50)
        for block in box.blocks:
            self.assertIsInstance(block, WeightedBlock2D)
            self.assertTrue(block.r in box.centers)
            self.assertEqual(block.a, box.a / box.nbx)
            self.assertEqual(block.b, box.b / box.nby)
            self.assertEqual(block.w, 1.0)

class EtchingBox2D(BlockBox2D):
    
    def __init__(self, a, b, nbx, nby, 
                 pbc=True):
        '''
        instantiate an etching box for 2D MC modeling
        etching process.
        
        Parameters
        ----------
        a : float
            the length of the box
        b : float
            the width of the box
        nbx : int
            the number of blocks in x direction
        nby : int
            the number of blocks in y direction
        pbc : bool
            Periodic boundary condition. Default is True.
        '''
        super().__init__(a, b, nbx, nby, pbc)
    
    def set_etch_source(self, src, v, a, **kwargs):
        '''initialize the etching source
        
        Parameters
        ----------
        src : np.ndarray
            the source position
        v : tuple
            the mean and stddev of the velocity of the source
        a : tuple
            the range of angles of the source emission
        vmin : float
            optional, the minimum velocity of the source
        vmax : float
            optional, the maximum velocity of the source
        '''
        assert isinstance(src, np.ndarray)
        assert src.shape == (2,)
        
        assert isinstance(v, tuple)
        assert len(v) == 2
        assert all(isinstance(vi, (int, float)) for vi in v)
        
        assert isinstance(a, tuple)
        assert len(a) == 2
        assert all(isinstance(ai, (int, float)) for ai in a)
        
        def f():
            '''generate the velocity of the source'''
            mean, stddev = v
            
        
        
if __name__ == '__main__':
    unittest.main()