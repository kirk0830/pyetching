import unittest
import numpy as np
import matplotlib.pyplot as plt

class Block2D:
    '''
    the etching process is simulated by a 2 dimensional
    Monte-Carlo. The all space is decritized into small
    blocks, each block can be either some materials, or
    the vacuum.

    This is the reason why we will implement a base
    class of the block in MC.
    '''
    def __init__(self, r, a, b=None):
        '''instantiate a block for 2D MC modeling
        etching process.
        
        Parameters
        ----------
        r : np.ndarray
            the center coordinate of the block
        a : float
            the edge length or height of the block
        b : float
            the width of the block
        '''
        # sanity check
        assert isinstance(r, np.ndarray)
        assert isinstance(a, (int, float))
        assert b is None or isinstance(b, (int, float))
        assert r.shape == (2,)
        assert a > 0
        assert b is None or b > 0

        self.r = r
        self.a = a
        self.b = a if b is None else b

    @staticmethod
    def incident(src, v, x=None, y=None):
        '''
        find the relation y = k*x + b from the given source
        point and the velocity vector
        
        y - y0 = k*(x - x0)
        y = k*x + (y0 - k*x0)
        
        Parameters
        ----------
        src : np.ndarray
            the source point
        v : np.ndarray
            the velocity vector
            
        Returns
        -------
        list of tuple
            set of (x, y) points that the incident particles
            will pass through.
        '''
        assert isinstance(src, np.ndarray)
        assert src.shape == (2,)
        assert isinstance(v, np.ndarray)
        assert v.shape == (2,)
        
        x0, y0 = src
        vx, vy = v
        k = vy / vx
        b = y0 - k * x0
        
        if x is not None:
            x = [x] if isinstance(x, float) else x
            return [(x, k*x + b) for x in x]
        elif y is not None:
            y = [y] if isinstance(y, float) else y
            return [((y - b) / k, y) if k != 0 else (x0, y) for y in y]
        else:
            raise ValueError('either x or y should be given')

    def within(self, p):
        '''check if the point is in the block
        
        Parameters
        ----------
        p : np.ndarray
            the point to be checked
        
        Returns
        -------
        bool
            True if the point is in the block, otherwise False
        '''
        p = np.array(p)
        assert p.shape == (2,)
        
        xmin, xmax = self.r[0] - self.a/2, self.r[0] + self.a/2
        ymin, ymax = self.r[1] - self.b/2, self.r[1] + self.b/2
        return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax

    def interact(self, src, v):
        '''interact with the incident particles
        
        Parameters
        ----------
        src : np.ndarray
            the source point
        v : np.ndarray
            the velocity vector
        
        Returns
        -------
        tuple of np.ndarray
            the new source point and the new velocity vector
        '''
        return src, v

class TestBlock2D(unittest.TestCase):
    
    def test_init(self):
        r = np.array([0, 0])
        a = 1
        b = 1
        block = Block2D(r, a, b)
        self.assertTrue(np.array_equal(block.r, r))
        self.assertEqual(block.a, a)
        self.assertEqual(block.b, b)
        
        a = 2
        block = Block2D(r, a)
        self.assertTrue(np.array_equal(block.r, r))
        self.assertEqual(block.a, a)
        self.assertEqual(block.b, a)
        
    def test_within(self):
        block = Block2D(np.array([0, 0]), 2, 2)
        p = np.array([0, 0])
        self.assertTrue(block.within(p))
        p = np.array([1, 1])
        self.assertTrue(block.within(p))
        p = np.array([1, 2])
        self.assertFalse(block.within(p))
        p = np.array([2, 2])
        self.assertFalse(block.within(p))

class ElasticBlock2D(Block2D):
    '''the block can interact with the incident particles
    by elastic scattering.
    '''
    def __init__(self, r, a, b=None, f=1.0):
        '''instantiate a block for 2D MC modeling
        etching process.
        
        Parameters
        ----------
        r : np.ndarray
            the center coordinate of the block
        a : float
            the edge length or height of the block
        b : float
            the width of the block
        f : float
            the ratio of kinetic energy loss in the
            elastic scattering. 1.0 means a fully
            elastic scattering.
        '''
        super().__init__(r, a, b)
        
        assert isinstance(f, (int, float))
        assert 1 > f >= 0
        self.f = f
        
    def interact(self, src, v):
        '''reflect the particle
        
        Parameters
        ----------
        src : np.ndarray
            the source point
        v : np.ndarray
            the velocity vector
        
        Returns
        -------
        tuple of np.ndarray
            the new source point and the new velocity vector
        '''
        assert isinstance(src, np.ndarray)
        assert src.shape == (2,)
        assert isinstance(v, np.ndarray)
        assert v.shape == (2,)
        
        # the incident point
        xmin, xmax = self.r[0] - self.a/2, self.r[0] + self.a/2
        ymin, ymax = self.r[1] - self.b/2, self.r[1] + self.b/2
        p =  Block2D.incident(src, v, x=[xmin, xmax])
        p += Block2D.incident(src, v, y=[ymin, ymax])
        p = list(set([x for x in p if self.within(x)]))
        # reflect the particle
        vout = v.copy()
        if len(p) == 0:
            return src, vout
        elif len(p) == 1:
            return p[0], vout * np.sqrt(1 - self.f)
        else:
            assert len(p) == 2, f'Invalid number of cross points: {len(p)}'
            # find the one near the source point
            p = p[0] if np.linalg.norm(p[0] - src) < np.linalg.norm(p[1] - src) else p[1]

            if abs((p[0] - xmax) * (p[0] - xmin)) <= 1e-12: # hit the vertical wall
                vout = np.array([-vout[0] * np.sqrt(1 - self.f), vout[1]])
            if abs((p[1] - ymax) * (p[1] - ymin)) <= 1e-12: # hit the horizontal wall
                vout = np.array([vout[0], -vout[1] * np.sqrt(1 - self.f)])
            return p, vout

class TestElasticBlock2D(unittest.TestCase):
    
    def test_init(self):
        r = np.array([0, 0])
        a = 1
        b = 1
        f = 0.5
        block = ElasticBlock2D(r, a, b, f)
        self.assertTrue(np.array_equal(block.r, r))
        self.assertEqual(block.a, a)
        self.assertEqual(block.b, b)
        self.assertEqual(block.f, f)
        
        a = 2
        block = ElasticBlock2D(r, a, f=f)
        self.assertTrue(np.array_equal(block.r, r))
        self.assertEqual(block.a, a)
        self.assertEqual(block.b, a)
        self.assertEqual(block.f, f)

    def test_elastic_collison(self):
        block = ElasticBlock2D(np.array([0, 0]), a=2, b=2, f=0) # fully elastic collision
        src = np.array([10, 10])
        v = np.array([-9, -10])
        p, vout = block.interact(src, v)
        self.assertTrue(np.linalg.norm(p - np.array([1, 0])) < 1e-6)
        self.assertTrue(np.linalg.norm(vout - np.array([9, -10])) < 1e-6)
        v = np.array([-10, -9])
        p, vout = block.interact(src, v)
        self.assertTrue(np.linalg.norm(p - np.array([0, 1])) < 1e-6)
        self.assertTrue(np.linalg.norm(vout - np.array([-10, 9])) < 1e-6)
        # hit the corner
        v = np.array([-9, -9]) 
        p, vout = block.interact(src, v)
        self.assertTrue(np.linalg.norm(p - np.array([1, 1])) < 1e-6)
        self.assertTrue(np.linalg.norm(vout - np.array([9, 9])) < 1e-6)
        # not interact
        v = np.array([1, 0]) 
        p, vout = block.interact(src, v)
        self.assertTrue(np.linalg.norm(p - src) < 1e-6)
        self.assertTrue(np.linalg.norm(vout - v) < 1e-6)

    def test_non_elastic_collison(self):
        block = ElasticBlock2D(np.array([0, 0]), a=2, b=2, f=0.5)
        src = np.array([10, 10])
        v = np.array([-9, -10])
        p, vout = block.interact(src, v)
        self.assertTrue(np.linalg.norm(p - np.array([1, 0])) < 1e-6)
        self.assertTrue(np.linalg.norm(vout - np.array([np.sqrt(1-0.5)*9, -10])) < 1e-6)

class PermeableBlock2D(Block2D):
    '''the block can interact with the incident particles
    by permeation.
    '''
    def __init__(self, r, a, b=None, f=1.0):
        '''instantiate a block for 2D MC modeling
        etching process.
        
        Parameters
        ----------
        r : np.ndarray
            the center coordinate of the block
        a : float
            the edge length or height of the block
        b : float
            the width of the block
        f : float
            the ratio of kinetic energy loss in the
            permeation. 1.0 means a fully permeation
            without any energy loss.
        '''
        super().__init__(r, a, b)
        
        assert isinstance(f, (int, float))
        assert 1 >= f >= 0
        self.f = f
        
    def interact(self, src, v):
        '''permeate the particle
        
        Parameters
        ----------
        src : np.ndarray
            the source point
        v : np.ndarray
            the velocity vector
        
        Returns
        -------
        tuple of np.ndarray
            the new source point and the new velocity vector
        '''
        assert isinstance(src, np.ndarray)
        assert src.shape == (2,)
        assert isinstance(v, np.ndarray)
        assert v.shape == (2,)
        
        # the incident point
        xmin, xmax = self.r[0] - self.a/2, self.r[0] + self.a/2
        ymin, ymax = self.r[1] - self.b/2, self.r[1] + self.b/2
        p =  Block2D.incident(src, v, x=[xmin, xmax])
        p += Block2D.incident(src, v, y=[ymin, ymax])
        p = list(set([x for x in p if self.within(x)]))
        # permeate the particle
        vout = v.copy()
        if len(p) == 0:
            return src, vout
        elif len(p) == 1:
            return p[0], vout * np.sqrt(1 - self.f)
        else:
            assert len(p) == 2, f'Invalid number of cross points: {len(p)}'
            # find the one far from the source point
            p = p[0] if np.linalg.norm(p[0] - src) > np.linalg.norm(p[1] - src) else p[1]
            return p, vout * np.sqrt(1 - self.f)

class TestPermeableBlock2D(unittest.TestCase):
    
    def test_init(self):
        r = np.array([0, 0])
        a = 1
        b = 1
        f = 0.5
        block = PermeableBlock2D(r, a, b, f)
        self.assertTrue(np.array_equal(block.r, r))
        self.assertEqual(block.a, a)
        self.assertEqual(block.b, b)
        self.assertEqual(block.f, f)
        
        a = 2
        block = PermeableBlock2D(r, a, f=f)
        self.assertTrue(np.array_equal(block.r, r))
        self.assertEqual(block.a, a)
        self.assertEqual(block.b, a)
        self.assertEqual(block.f, f)
    
    def test_penetrate(self):
        block = PermeableBlock2D(np.array([0, 0]), a=2, b=2, f=0)
        src = np.array([10, 10])
        v = np.array([-9, -10])
        p, vout = block.interact(src, v)
        self.assertTrue(np.linalg.norm(p - np.array([10 - 11*9/10, -1])) < 1e-6)
        self.assertTrue(np.linalg.norm(vout - v) < 1e-6)
        
        v = np.array([-10, -9])
        p, vout = block.interact(src, v)
        self.assertTrue(np.linalg.norm(p - np.array([-1, 10 - 11*9/10]) < 1e-6))
        self.assertTrue(np.linalg.norm(vout - v) < 1e-6)
        
        v = np.array([-9, -9])
        p, vout = block.interact(src, v)
        self.assertTrue(np.linalg.norm(p - np.array([-1, -1]) < 1e-6))
        self.assertTrue(np.linalg.norm(vout - v) < 1e-6)
        
        v = np.array([1, 0]) # not interact
        p, vout = block.interact(src, v)
        self.assertTrue(np.linalg.norm(p - src) < 1e-6)
        self.assertTrue(np.linalg.norm(vout - v) < 1e-6)

class WeightedBlock2D(Block2D):
    '''
    the block that has a weight, which is used to indicate
    its existence.
    '''
    def __init__(self, r, a, b=None, w=1.0):
        '''instantiate a block for 2D MC modeling
        etching process.
        
        Parameters
        ----------
        r : np.ndarray
            the center coordinate of the block
        a : float
            the edge length or height of the block
        b : float
            the width of the block
        w : float
            the weight of the block, which is used to
            indicate the existence of the block.
        '''
        super().__init__(r, a, b)
        
        assert isinstance(w, float)
        assert w > 0
        self.w = w
        
    def interact(self, d):
        self.w -= d

class ChemicalBlock2D(Block2D):
    '''the block can happen chemical reaction
    with the incident particles.
    '''
    def __init__(self, r, a, b=None, rxn=None):
        '''instantiate a block for 2D MC modeling
        etching process.
        
        Parameters
        ----------
        r : np.ndarray
            the center coordinate of the block
        a : float
            the edge length or height of the block
        b : float
            the width of the block
        rxn : dict
            the chemical reaction recipe for the chemical
            etching process.
        '''
        super().__init__(r, a, b)
        
        assert rxn is None or isinstance(rxn, dict)
        self.rxn = rxn

class PhysicalScatterBlock2D(WeightedBlock2D,
                             ElasticBlock2D,
                             PermeableBlock2D):
    '''the block can interact with the incident particles
    by elastic scattering and permeation.
    '''
    def __init__(self, r, a, b=None,
                 w=1.0, 
                 felastic=1.0,
                 fpermeable=1.0):
        '''instantiate a block for 2D MC modeling
        etching process.
        
        Parameters
        ----------
        r : np.ndarray
            the center coordinate of the block
        a : float
            the edge length or height of the block
        b : float
            the width of the block
        w : float
            the weight of the block, which is used to
            indicate the existence of the block.
        felastic : float
            the ratio of kinetic energy loss in the
            elastic scattering. 1.0 means a fully
            elastic scattering.
        fpermeable : float
            the ratio of kinetic energy loss in the
            permeation. 1.0 means a fully permeation
            without any energy loss.
        '''
                
        WeightedBlock2D.__init__(self, r, a, b, w)
        ElasticBlock2D.__init__(self, r, a, b, felastic)
        PermeableBlock2D.__init__(self, r, a, b, fpermeable)
        
        assert felastic + fpermeable <= 1 # energy conservation

    def interact(self, src, v):
        '''interact with the incident particles
        
        Parameters
        ----------
        src : np.ndarray
            the source point
        v : np.ndarray
            the velocity vector
        
        Returns
        -------
        tuple of np.ndarray
            the new source point and the new velocity vector
        '''
        WeightedBlock2D.interact(self, 1 - self.f) # transfer the energy loss to the weight
        return ElasticBlock2D.interact(self, src, v)

class EtchableBlock2D(ChemicalBlock2D, PhysicalScatterBlock2D):
    '''the block can interact with the incident particles
    by chemical reaction, elastic scattering and permeation.
    '''
    def __init__(self, r, a, b=None,
                 w=1.0, 
                 felastic=1.0,
                 fpermeable=1.0,
                 rxn=None):
        '''instantiate a block for 2D MC modeling
        etching process.
        
        Parameters
        ----------
        r : np.ndarray
            the center coordinate of the block
        a : float
            the edge length or height of the block
        b : float
            the width of the block
        w : float
            the weight of the block, which is used to
            indicate the existence of the block.
        felastic : float
            the ratio of kinetic energy loss in the
            elastic scattering. 1.0 means a fully
            elastic scattering.
        fpermeable : float
            the ratio of kinetic energy loss in the
            permeation. 1.0 means a fully permeation
            without any energy loss.
        rxn : dict
            the chemical reaction recipe for the chemical
            etching process.
        '''
                
        ChemicalBlock2D.__init__(self, r, a, b, rxn)
        PhysicalScatterBlock2D.__init__(self, r, a, b, w, felastic, fpermeable)

'''
then the evolution would be like this:

for i in range(n):
    r, v = incident_particles()
    while v >= 0:
        r, v = all_blocks().interact(r, v)
'''

if __name__ == '__main__':
    unittest.main()