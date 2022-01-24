simple_dust_proj
================

Projects Bayestar-like 3D dust maps to 2D.

To project a set of frames, run

    python3 project_frames.py frame_props.json /path/to/dust/map.npy "/output/filename/pattern_{:05d}.png"

The output filename pattern uses the Python `f-string` format, and will be fed a sequence of frame numbers. The
above format string will generate frames ending with `_00000.png`, `_00001.png`, and so on.

The dust map is numpy array representing a dust extinction density, with shape `(pixel, distance)`. The pixels
are to follow `HEALPix` nested ordering, and there should be 120 distance slices, spaced evenly in distance modulus,
from 4 to 11.5. Right now, the distances to the slices are hard-coded, though this could easily be relaxed.

The file `frame_props.json` contains two types of information: the camera projection, and the location and
orientation of the camera for each frame. An example file can be found at `frame_props_example.json`. A commented
version of this file can be found at `frame_props_example_with_comments.json`. The commented version will not
work, because JSON has no comment functionality.
