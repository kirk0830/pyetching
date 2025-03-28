'''
the box that containes blocks for 2D MC modeling
'''
import unittest
import itertools as it
import multiprocessing as mp
import os
from typing import Callable, Generator

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
            the function to instantiate the block, should be
            configured so that only three parameters are left
            to be passed, i.e. the center of the block, the
            length and width of the block. e.g.
            ```python
            # the simplest
            def f(r, a, b):
                return Block2D(r, a, b)
            # or
            def f(r, a, b):
                return SomeBlock2D(r, a, b, f=1.0)
            ```
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
        from pyetching.block.block2d import Block2D
                
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
    
    def dist(self, a, b):
        '''calculate the distance between two points'''
        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert a.shape == (2,)
        assert b.shape == (2,)
        if not self.pbc:
            return np.linalg.norm(a - b)
        
        cell = np.array([[self.a, 0], [0, self.b]])
        direct_a = np.linalg.solve(cell, a)
        direct_b = np.linalg.solve(cell, b)
        direct_dist = (direct_a - direct_b + 0.5) % 1 - 0.5
        return np.linalg.norm(np.dot(cell, direct_dist))
    
    def etch(self, etchgen, n: int, nthread: int=1, vthr: float=1e-6):
        '''perform etching for n particles (till each particle's
        velocity is zero)
        
        Parameters
        ----------
        f : Generator
            the generator function to generate etching particles,
            should yield the source position and velocity of the particle.
            should have 1 parameter, i.e. the number of particles to etch.
            e.g.
            ```python
            def etchgen(n):
                for _ in range(n):
                    yield np.array([0.0, 0.0]), np.random.normal(1.0, 0.1)
            ```
        n : int
            the number of particles to etch
        nthread : int
            the number of threads to use for etching, default is 1
        vthr : float
            the threshold of velocity, below which the particle is assumed
            to be stopped. default is 1e-6
        '''
        assert callable(etchgen)
        assert isinstance(n, int)
        assert n > 0
        assert self.blocks is not None
        assert len(self.blocks) > 0

        print('')
        for src, v in etchgen(n):
            while np.linalg.norm(v) >= vthr:
                # the following part can be highly parallelized. 
                print('before:\n', [b.w for b in self.blocks])
                print('r:\n', [b.r for b in self.blocks if b.w > 0])
                temp = [b.interact(src, v) for b in self.blocks if b.w > 0]
                print('after:\n', [b.w for b in self.blocks])
                # exclude those not interacted
                temp = [t for t in temp if np.linalg.norm(t[1] - src) >= 1e-6]
                if not temp:
                    break
                i = np.argmin([self.dist(src, np.array(t[0])) for t in temp])
                src, v = temp[i]
                src = np.array(src)
                print(f'Etching particle at {src} with velocity {v}')
                
    def display(self):
        '''display the weights of all blocks in the hotmap
        '''
        assert self.blocks is not None
        assert len(self.blocks) > 0
        w = np.array([b.w for b in self.blocks]).reshape(self.nby, self.nbx)
        fig, ax = plt.subplots()
        ax.imshow(w, cmap='hot', interpolation='nearest')
        # add colorscale
        cbar = plt.colorbar(ax.imshow(w, cmap='hot', interpolation='nearest'))
        cbar.set_label('Weight')
        ax.set_title('Weight of blocks')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xticks(np.arange(self.nbx))
        ax.set_yticks(np.arange(self.nby))
        return fig, ax

class TestEtchingBox2D(unittest.TestCase):
    def test_init(self):
        a = 10
        b = 20
        nbx = 5
        nby = 10
        pbc = True
        box = EtchingBox2D(a, b, nbx, nby, pbc)
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

    def test_etch(self):
        box = EtchingBox2D(a=10, b=20, nbx=5, nby=10)
        
        def etchgen(n):
            vmean, vstddev = 1.0, 0.1
            amin, amax = 0, np.pi
            for _ in range(n):
                v = np.random.normal(vmean, vstddev)
                theta = np.random.uniform(amin, amax)
                yield np.array([5.0, 0.0]), \
                    v * np.array([np.cos(theta), np.sin(theta)])
        
        # first test the generator
        v = np.array([p[1] for p in etchgen(1000)])
        self.assertEqual(v.shape, (1000, 2))
        vmean = np.mean([np.linalg.norm(v_) for v_ in v])
        vstddev = np.std([np.linalg.norm(v_) for v_ in v])
        self.assertAlmostEqual(vmean, 1.0, delta=0.1)
        self.assertAlmostEqual(vstddev, 0.1, delta=0.1)
        
        from pyetching.block.block2d import PhysicalScatterBlock2D
        def f(r, a, b):
            return PhysicalScatterBlock2D(r, a, b, w=10.0, felastic=0.5, fpermeable=0)
        
        box.allocate(f)
        box.etch(etchgen, 1, nthread=1, vthr=1e-6)
        
        # test the display function
        fig, ax = box.display()
        plt.savefig('test.png')
        plt.close()

if __name__ == '__main__':
    unittest.main()