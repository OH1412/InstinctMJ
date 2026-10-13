# Unitree H2 MuJoCo asset

This directory contains the 31-joint H2 MJCF used by the InstinctMJ HMV port:
29 body joints plus `head_pitch_joint` and `head_yaw_joint`. The fixed camera
mount is represented by the offset from `head_yaw_link` in the H2 parkour
configuration.

The MJCF and meshes were adapted from the Apache-2.0-licensed
[`wbc-mjlab-extension-h2`](https://github.com/wbc-mjlab/wbc-mjlab-extension-h2)
asset. As noted by that project, its H2 actuator parameters and geometry are a
reference model and should be validated against hardware/vendor data before
deployment.
