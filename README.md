* Code for: Mechanics of Fluid-Filled Interstitial Gaps. I. Modeling Gaps in a Compact Tissue.

The environment file these scripts were last run with was 'gaps1_env.yml'.

I developed and ran the scripts in Anaconda using Python 3.9.7. To run everything I required approximately 4GB of disk space. 

I last validated these scripts and their output on Sept. 14, 2022.


*** Figure 5: theta_i vs. l/l_s plots:
The associated file to produce the plot in Figure 5B is: gaps1_fig5_eq4.py

*** Figure 6: simulations of asymmetric gaps:
There are 3 scripts used to produce the figures in the paper. The simulations were originally written in Python 2, but have been maintained by me to run in Python 3. In the process of maintenance, I sped up the simulation such that it doesn't need to make use of multiprocessing. I also changed the process of finding circle-circle intersection points such that it is now done analytically instead of numerically. As a result, the accuracy and speed of the simulation has increases dramatically, and you will notice a tighter distribution of data. The key points, however, remain -- i.e. that when curvatures can vary whilst maintaining a constant average curvature, the gap sizes also begin to vary, and no longer fit to a line as neatly as they do when contact angles vary. 
The scripts used to produce this figure are run in this order: 'sct0_making_circles_py3.py', 'sct1_get_intersections_py3.py', 'sct2_gap_data_reader_py3.py'.
