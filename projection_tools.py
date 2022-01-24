#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import h5py
import transforms3d

from glob import glob
from progressbar import ProgressBar


class Projection(object):
    def __init__(self, x0, dx):
        self.shape = x0.shape[1:]
        self.x0 = x0 # shape = (3, ni, nj)
        self.dx = dx # shape = (3, ni, nj)
    
    def rotated(self, alpha, beta, gamma, convention):
        R = transforms3d.euler.euler2mat(alpha, beta, gamma, convention)
        x0_rot = np.einsum('ij,jkl', R, self.x0)
        dx_rot = np.einsum('ij,jkl', R, self.dx)
        return Projection(x0_rot, dx_rot)
    
    def get_surface(self, x, d, d_scatter=None):
        if d_scatter is not None:
            dd = np.random.random(self.dx.shape[1:]) * d_scatter
            return x[:,None,None] + self.x0 + (dd+d)[None,:,:] * self.dx
        return x[:,None,None] + self.x0 + d * self.dx


class OrthographicProjection(Projection):
    def __init__(self, shape, scale):
        x0 = np.zeros((3,)+shape, dtype='f8')
        x0[:2] = np.indices(shape)
        for k in range(len(shape)):
            x0[k] -= 0.5*(shape[k]-1)
            x0[k] *= scale / shape[k]
        dx = np.zeros((3,)+shape, dtype='f8')
        dx[2] = 1.
        super(OrthographicProjection, self).__init__(x0, dx)


class StereographicProjection(Projection):
    def __init__(self, shape, theta_max, randomize_angles=False):
        # Screen size required to achieve correct field of view
        R_max = 2. * np.sin(theta_max) / (1. + np.cos(theta_max))#np.tan(theta_max)
        print(R_max)
        
        # Produce grid of screen coordinates
        XY = np.indices(shape).astype('f8')
        
        if randomize_angles:
            XY += np.random.random(XY.shape) - 0.5
        
        for k in range(2):
            XY[k] -= 0.5*(shape[k]-1)
            XY[k] *= (2. * R_max) / shape[0]
        
        # Project screen to surface of sphere
        R2 = np.sum(XY**2., axis=0)
        a = 1. / (R2 + 1.)
        
        dx = np.empty((3,)+shape, dtype='f8')
        dx[0,:] = 2. * a * XY[0]
        dx[1,:] = 2. * a * XY[1]
        dx[2,:] = -a * (R2 - 1.)
        
        # Camera at the origin
        x0 = np.zeros((3,)+shape, dtype='f8')
        
        super(StereographicProjection, self).__init__(x0, dx)


class MapData(object):
    def __init__(self, nside, pix_idx, pix_val, dm0, dm1):
        self.nside = nside
        self.pix_idx = pix_idx
        self.pix_val = pix_val
        
        self.dm0, self.dm1 = dm0, dm1
        self.n_dists = self.pix_val.shape[2]
        self.n_samps = self.pix_val.shape[1]
        
        self.nside_max = np.max(nside)
        
        # data_idx: (high-res healpix idx) -> (idx in stored data)
        n_pix_hires = hp.pixelfunc.nside2npix(self.nside_max)
        self.data_idx = np.full(n_pix_hires, -1, dtype='i8')
        
        for n in np.unique(self.nside):
            idx = (self.nside == n)
            pix_idx_n = self.pix_idx[idx]
            data_idx_n = np.where(idx)[0]
            mult = (self.nside_max // n)**2
            for k in range(mult):
                self.data_idx[mult*pix_idx_n+k] = data_idx_n
        print(r'{} % filled'.format(100.*np.sum(self.data_idx != -1)/len(self.data_idx)))
    
    def get_hires_map(self, fill=np.nan):
        idx = (self.data_idx == -1)
        d = self.pix_val[self.data_idx]
        d[idx] = np.nan
        return d
    
    def get_pix_val(self, v, outer_fill=0.):
        # (x,y,z) -> (high-res healpix index)
        pix_idx_hires = hp.pixelfunc.vec2pix(
            self.nside_max,
            v[0], v[1], v[2],
            nest=True
        )
        
        # (high-res healpix index) -> (idx in stored data)
        pix_idx = self.data_idx[pix_idx_hires]
        
        # (x,y,z) -> r^2 -> (distance bin)
        r2 = np.sum(v**2, axis=0)
        dm = 2.5 * np.log10(r2) - 5. 
        #print('n_dists =', self.n_dists)
        #print('dm0,dm1 =', self.dm0, self.dm1)
        #print('ddm =', (self.dm1-self.dm0)/(self.n_dists-1))
        dist_idx = (dm - self.dm0) * ((self.n_dists-1) / (self.dm1-self.dm0))
        dist_idx = np.ceil(dist_idx).astype('i4')
        idx = (dist_idx < 0)
        if np.any(idx):
            dist_idx[idx] = 0
        #print('dist_idx({}) = {}'.format(np.sqrt(r2), dist_idx))
        
        # Determine out-of-bounds indices
        #inner_idx = (dist_idx < 0)
        outer_idx = (dist_idx >= self.n_dists) | (pix_idx == -1)
        bad_idx = outer_idx #(inner_idx | outer_idx)
        if np.any(bad_idx):
            dist_idx[bad_idx] = -1
        
        # Fetch data
        d = self.pix_val[pix_idx, :, dist_idx]
        
        # Fill in out-of-bounds indices
        #if np.any(inner_idx):
        #    d[inner_idx] = inner_fill
        if np.any(outer_idx):
            d[outer_idx] = outer_fill
        
        print('n_good = {}'.format(np.sum(~bad_idx)))
        
        # Return data
        return d


class MapDataHires(object):
    def __init__(self, pix_val, dm0, dm1):
        self.pix_val = pix_val
        self.nside = hp.pixelfunc.npix2nside(pix_val.shape[0])
        
        self.dm0, self.dm1 = dm0, dm1
        self.n_dists = self.pix_val.shape[1]
    
    def get_pix_val(self, v, outer_fill=0.):
        # (x,y,z) -> (healpix index)
        pix_idx = hp.pixelfunc.vec2pix(
            self.nside,
            v[0], v[1], v[2],
            nest=True
        )
        
        # (x,y,z) -> r^2 -> (distance bin)
        r2 = np.sum(v**2, axis=0)
        dm = 2.5 * np.log10(r2) - 5. 
        dist_idx = (dm - self.dm0) * ((self.n_dists-1) / (self.dm1-self.dm0))
        dist_idx = np.ceil(dist_idx).astype('i4')
        idx = (dist_idx < 0)
        if np.any(idx):
            dist_idx[idx] = 0
        
        # Determine out-of-bounds indices
        outer_idx = (dist_idx >= self.n_dists)
        bad_idx = outer_idx #(inner_idx | outer_idx)
        if np.any(bad_idx):
            dist_idx[bad_idx] = -1
        
        # Fetch data
        d = self.pix_val[pix_idx, dist_idx]
        
        # Fill in out-of-bounds indices
        if np.any(outer_idx):
            d[outer_idx] = outer_fill
        
        #print('n_good = {}'.format(np.sum(~bad_idx)))
        
        # Return data
        return d

