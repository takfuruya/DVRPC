Vision Based Vehicle Tracking
=============================

### Required programs & libraries

                   Tested Version
    - Linux        Ubuntu 13.10
    - Make         3.81
    - OpenCV       2.4.5

You also need a video and specify its location in the Makefile.

### Project directory structure

    - bin            Directory containing object and executable files.
    - intermediate   Directory containing background model and trajectories which are used between programs.
    - Makefile       File for compiling C++ code.
    - MATLAB         Directory containing MATLAB code (used for prototyping).
    - src            Directory containing C++ code.

   
### List of commands to build & run


`make extract_bg`

Extracts background model from video and stores it in `intermediate\bg.txt`.

`make extract_traj`

Extracts trajectories from video and stores it in `intermediate\traj.txt`. It uses `intermediate\bg.txt`.

`make group_traj`

Groups trajectories from video. It uses `intermediate\traj.txt`.

`make play_video`

Plays video between specified range.

For further options and other programs, see `Makefile`.
