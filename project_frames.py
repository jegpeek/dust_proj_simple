#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import healpy as hp

import os
from glob import glob
from progressbar import ProgressBar
from argparse import ArgumentParser
import json

from PIL import Image

import multiprocessing
import multiprocessing.queues
import queue

from projection_tools import StereographicProjection, OrthographicProjection


class MapDataSingleRes(object):
    """
    Contains a single-resolution HEALPix map with multiple distance slices.
    """
    
    def __init__(self, pix_val, dm0, dm1):
        self.dm0, self.dm1 = dm0, dm1
        self.pix_val = pix_val
        n_pix, self.n_dists = self.pix_val.shape
        self.nside = hp.pixelfunc.npix2nside(n_pix)
    
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


def project_map(density_map, proj,
                x0, step_size, max_dist,
                fuzzy=None):
    img = np.zeros(proj.shape, dtype='f8')
    
    for d in np.arange(0., max_dist+0.1*step_size, step_size):
        v = proj.get_surface(x0, d, d_scatter=step_size)
        
        if fuzzy is not None:
            v += fuzzy * step_size * np.random.normal(size=v.shape)
        
        img += density_map.get_pix_val(v, outer_fill=0.)
        
    img *= step_size / 1000.
    
    return img


def save_image(img, fname, vmax, clip_mode):
    # Remove NaNs and Infs
    idx = ~np.isfinite(img)
    img[idx] = 0.
    
    if clip_mode == 'tanh':
        # Soft clip at vmax
        img = np.tanh(1.5 * img / vmax)
    elif clip_mode == 'clip':
        # Hard clip at vmax
        img = img / vmax
    else:
        raise ValueError('clip_mode must be "tanh" or "clip"')
    
    # Convert to unsigned integer
    depth = 16
    max_value = 2**depth-1
    img = np.clip(max_value*img, 0., max_value).astype('uint16')
    
    # Save image
    im = Image.fromarray(img.T[::-1,::-1], 'I;16')
    im.save(fname)


def make_frames_parallel(n_procs, dustmap_fname, frame_fname, spec):
    # Set up queue for workers to pull frame numbers from
    frame_q = multiprocessing.JoinableQueue()
    
    n_frames = len(spec['frame_props'])
    
    for k in range(n_frames):
        frame_q.put(k)
    
    # Set up lock to allow first image to be written without interference btw/ processes
    #lock = multiprocessing.Lock()
    
    # Spawn worker processes to plot images
    procs = []
    for i in range(n_procs):
        p = multiprocessing.Process(
            target=make_frames,
            args=(dustmap_fname, frame_fname, spec, frame_q)
        )
        procs.append(p)
    
    for p in procs:
        p.start()
    
    frame_q.join()
    
    print('All workers done.')
    

def make_frames(dustmap_fname, frame_fname, spec, frame_q):
    frame_props = spec['frame_props']
    n_frames = len(frame_props)
    
    # Set up the projection
    p = spec['camera_props']
    if p['projection'] == 'stereographic':
        proj = StereographicProjection(
            (p['x_pix'], p['y_pix']),
            0.5*np.radians(p['fov']),
            randomize_angles=p['randomize_angles']
        )
    elif p['projection'] == 'orthographic':
        proj = OrthographicProjection(
            (p['x_pix'], p['y_pix']),
            0.5*np.radians(p['fov']),
            randomize_angles=p['randomize_angles']
        )
    else:
        raise ValueError('Projection must be "stereographic" or "gnomonic"')
    
    # Rotate camera from +z-direction into standard
    # orientiation, facing Galactic center.
    proj = proj.rotated(0.5*np.pi, 0.5*np.pi, 0., 'szyz')
    
    # Load density map
    print('Loading map ...')
    density_map = load_map(dustmap_fname)
    
    # Ray-cast each frame along path
    print('Ray-casting frames ...')
    bar = ProgressBar(max_value=n_frames)
    bar.update(0)

    while True:
        try:
            k = frame_q.get(True, 1.0)
        except queue.Empty:
            print('Worker finished.')
            return
        
        # Get camera location and orientation
        x = np.array(spec['frame_props'][k]['xyz'])
        a,b,g = spec['frame_props'][k]['angles']
        
        proj_k = proj.rotated(a, b, g, p['euler_convention'])
        
        img = project_map(
            density_map,
            proj_k,
            x,
            p['step_size'],
            p['max_dist'],
            fuzzy=p['fuzzy']
        )
        
        # Get the maximum pixel value (at which the image will saturate)
        if p['vmax'] == 'auto':
            vmax = np.max(img)
            print(f'frame {k}: vmax = {vmax}')
        else:
            vmax = p['vmax']
        
        save_image(img, frame_fname.format(k), vmax, p['clip_mode'])
        
        bar.update(k+1)
        
        if isinstance(frame_q, multiprocessing.queues.JoinableQueue):
            frame_q.task_done()
    
    print('Worker finished.')


def load_map(dustmap_fname):
    # Density map
    pix_val = np.load(dustmap_fname)
    idx = ~np.isfinite(pix_val)
    if np.any(idx):
        pix_val[idx] = 0.
    density_map = MapDataSingleRes(pix_val, 4., 11.5)

    return density_map


def main():
    parser = ArgumentParser(
        description='Project images of 3D dust distribution down to 2D.',
        add_help=True
    )
    parser.add_argument(
        'spec',
        type=str,
        help='JSON specifying projection properties for each image.'
    )
    parser.add_argument(
        'dustmap',
        type=str,
        help='Filename of dust map.'
    )
    parser.add_argument(
        'output',
        type=str,
        help='Output filename pattern, indicating where the frame index '
             'should go (e.g., "image_{:05d}.png").'
    )
    parser.add_argument(
        '--scale-resolution',
        type=float,
        help='Scale the resolution of the images by this factor (default: no scaling).'
    )
    parser.add_argument(
        '--n-processes',
        type=int,
        default=1,
        help='# of parallel processes to use to generate images.'
    )
    args = parser.parse_args()
    
    with open(args.spec, 'r') as f:
        spec = json.load(f)
    
    print(f'Loaded specifications for {len(spec["frame_props"])} images.')
    
    make_frames_parallel(args.n_processes, args.dustmap, args.output, spec)
    
    return 0


if __name__ == '__main__':
    main()
