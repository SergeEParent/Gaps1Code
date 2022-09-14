# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:22:04 2016
Maintained on Wed Sept 14 2022


@author: Serge E. Parent
"""


from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
labelfont = {'fontname':'Times New Roman'}





print('Preparing input and output directories...')
## get the location of this script:
sct_dir = Path.cwd() # could also use sct_dir = Path('.').resolve()
## save the plot in the same folder that this script resides in:
out_dir = Path.joinpath(sct_dir, 'gaps1_fig6')

## if out_dir exists already, do nothing, otherwise, make it:
if Path.exists(out_dir):
    pass
else:
    Path.mkdir(out_dir)



## Get the directory where the three circles parameter data is stored:
threeCircle_dir = Path.joinpath(sct_dir, 'three_circles')

## Load the circles that were output by 'making_circles.py':
circle0_yxr = np.load(Path.joinpath(threeCircle_dir, 'circle0_yxr.npy'))
circle1_yxr = np.load(Path.joinpath(threeCircle_dir, 'circle1_yxr.npy'))
circle2_yxr = np.load(Path.joinpath(threeCircle_dir, 'circle2_yxr.npy'))

## Get the directory where the gaps simulation data is stored:
gapData_dir = Path.joinpath(sct_dir, 'gap_data')

## Load the data from the simulations of tricellular gaps:
gap_data = np.load(Path.joinpath(gapData_dir, 'gap_data.npy'))

# gap_data is an array with 31 columns, where the columns are, in order:
# (r0, r1, r2, 
# c01_inters_x, c01_inters_y, 
# c02_inters_x, c02_inters_y, 
# c12_inters_x, c12_inters_y, 
# side0, side1, side2, tri_area, tri_perim, 
# *ori0, *ori1, *ori2, 
# angle01, angle02, angle12, 
# arc0, arc1, arc2, curv_area, curv_perim, 
# curv_angle01, curv_angle02, curv_angle12)

headers = ['r0', 'r1', 'r2', 
           'c01_inters_x', 'c01_inters_y', 
           'c02_inters_x', 'c02_inters_y', 
           'c12_inters_x', 'c12_inters_y', 
           'side0', 'side1', 'side2', 'tri_area', 'tri_perim', 
           'ori0_x', 'ori0_y', 'ori1_x', 'ori1_y', 'ori2_x', 'ori2_y', 
           'angle01', 'angle02', 'angle12', 
           'arc0', 'arc1', 'arc2', 'curv_area', 'curv_perim', 
           'curv_angle01', 'curv_angle02', 'curv_angle12']

## turn the data into a DataFrame because it's easier to work with columns with
## header names:
gaps_df = pd.DataFrame(data=gap_data, columns=headers)



## one of the configurations looks strange (falls outside expectations), 
## so I will check it; it has a datapoint where the angle sum is < 166degrees
## and the average sidelength is < 0.04:

# query = np.logical_and(gaps_ang_sums < 166, gaps_avg_sides < 0.04)
# indices = np.where(query)
# print(indices)
## returns index 6480 and index 52560
# gaps_df.iloc[6480]
# gaps_df.iloc[52560]
## they are extremely small, and as far as I can tell this is the actual value,
## although maybe prone to limits of computational precision?

# fig, ax = plt.subplots(figsize=(3.25, 3.25))
# ax.scatter(gaps_avg_sides[query], gaps_ang_sums[query], c='k', marker='.', s=1)


## in any case, those two points will be masked since it's unclear why they 
## deviate so much from all the others, but most likely it is due to a very 
## small gap size. 
mask_indices = [6480, 52560]
mask = np.add.reduce([gaps_df.index == mask_idx for mask_idx in mask_indices], dtype=bool)
mask = np.expand_dims(mask, axis=1)
mask_arr = np.ones(gaps_df.shape, dtype=bool)
mask_arr *= mask
gaps_df.mask(mask_arr, axis=0, inplace=True)







print('Plotting figures...')
##
## Figure 6B:
##    
    
## plot all of the curved angle data vs all the sidelength data:
## get the sums of the curved angles:
gaps_ang_sums = gaps_df[['curv_angle01', 'curv_angle02', 'curv_angle12']].sum(axis=1)

## get the average sidelengths:
gaps_avg_sides = gaps_df[['side0', 'side1', 'side2']].mean(axis=1) / 1
## keep in mind that the sidelength ell_s is assumed to be 1 in this scenario,
## which is where the divide by 1 at the end of the above line comes from.

## return the lower limit of circle combinations radii:
## (circle0 radius = 1, circle1 radius = 0.8, circle2 radius = 0.8):
gaps_r_10_08_08 = gaps_df.query('r0==1 and r1==0.8 and r2==0.8')
## get the sums of the curved angles:
gaps_r_10_08_08_ang_sums = gaps_r_10_08_08[['curv_angle01', 'curv_angle02', 'curv_angle12']].sum(axis=1)
## get the average sidelengths:
gaps_r_10_08_08_avg_sides = gaps_r_10_08_08[['side0', 'side1', 'side2']].mean(axis=1) / 1

## return the upper limit of circle combinations radii:
## (circle0 radius = 1, circle1 radius = 1.2, circle2 radius = 1.2):
gaps_r_10_12_12 = gaps_df.query('r0==1 and r1==1.2 and r2==1.2')
## get the sums of the curved angles:
gaps_r_10_12_12_ang_sums = gaps_r_10_12_12[['curv_angle01', 'curv_angle02', 'curv_angle12']].sum(axis=1)
## get the average sidelengths:
gaps_r_10_12_12_avg_sides = gaps_r_10_12_12[['side0', 'side1', 'side2']].mean(axis=1) / 1






### Equation 4, but for if all radii are 0.8 or all radii are 1.2:
n_steps = 1000
min_yax_angle = 0
max_yax_angle = 45 
## keep in mind that the equation uses half angles, where as the simulation
## produced measured angles. a measured angle is twice the half-angle (if you
## assume that the angle is symmetric).

theta = np.linspace(min_yax_angle, np.deg2rad(max_yax_angle), n_steps)

r_10_08_08_avg = np.mean([1.0, 0.8, 0.8])
r_10_12_12_avg = np.mean([1.0, 1.2, 1.2])

dif_r_10_08_08 = np.linspace(r_10_08_08_avg, r_10_08_08_avg, n_steps)
dif_r_10_12_12 = np.linspace(r_10_12_12_avg, r_10_12_12_avg, n_steps)

dif_Ri = np.vstack([dif_r_10_08_08, dif_r_10_12_12])

dif_li = ( np.cos(theta) - np.sqrt(3.0) * np.sin(theta) ) * dif_Ri

for index, value in enumerate(theta):
    if theta[index] <= np.deg2rad(30):
        dif_li[:,index] *= 1
    elif theta[index] > np.deg2rad(30):
        dif_li[:,index] *= -1








fig6b, ax6b = plt.subplots(figsize=(3.25,3.25))
ax6b.tick_params(labelsize=10, direction='in', top=True, right=True)
ax6b.scatter(gaps_avg_sides, gaps_ang_sums, c='k', marker='.', s=1, zorder=1)
# ax6b.scatter(gaps_r_10_08_08_avg_sides, gaps_r_10_08_08_ang_sums, c='lightgrey', marker='.', s=1)
# ax6b.scatter(gaps_r_10_12_12_avg_sides, gaps_r_10_12_12_ang_sums, c='lightgrey', marker='.', s=1)
## plot the equation solution for symmetric gaps with the same mean radius of 
## curvature as the r0=1, r1=0.8, r2=0.8 solution and r0=1, r1=1.2, r2=1.2 
## solution. Also, since these are the half angles we need to multiply them by 
## 2 to make them comparable to the measured angles calculated from the 
## simulations. In addition, we need to then multiply them by 3 since the plot
## is plotting the sum of the angles:
ax6b.plot(dif_li[0], np.rad2deg(theta)*2*3, linestyle='-', lw=1, c='lightgrey', zorder=2)
ax6b.plot(dif_li[1], np.rad2deg(theta)*2*3, linestyle='-', lw=1, c='lightgrey', zorder=2)
ax6b.axhline(180, c='lightgrey', ls='--', zorder=0)
ax6b.set_xlim(0, 1.5)
ax6b.set_ylim(0, 300)
ax6b.set_xticks(np.arange(0, 1.5, 0.2))
ax6b.set_yticks(np.arange(0,301, 50))
ax6b.set_xlabel(r'$\bar\ell$/$\ell_s$')
ax6b.set_ylabel(r'$\Sigma$($\theta_i$) (degrees)')
fig6b.savefig(Path.joinpath(out_dir, 'figure6b.tif'))









##
## Figure 6C:
##

## return where all circles have a radius of 1:
gaps_r_10_10_10 = gaps_df.query('r0==1 and r1==1 and r2==1')

## get the sums of the curved angles:
gaps_r_10_10_10_ang_sums = gaps_r_10_10_10[['curv_angle01', 'curv_angle02', 'curv_angle12']].sum(axis=1)

## get the average sidelengths:
gaps_r_10_10_10_avg_sides = gaps_r_10_10_10[['side0', 'side1', 'side2']].mean(axis=1) / 1
## keep in mind that the sidelength ell_s is assumed to be 1 in this scenario,
## which is where the divide by 1 at the end of the above line comes from.



fig6c, ax6c = plt.subplots(figsize=(3.25,3.25))
ax6c.tick_params(labelsize=10, direction='in', top=True, right=True)
ax6c.scatter(gaps_avg_sides, gaps_ang_sums, c='lightgrey', marker='.', s=1)
ax6c.scatter(gaps_r_10_10_10_avg_sides, gaps_r_10_10_10_ang_sums, c='k', marker='.', s=1)
ax6c.set_xlim(0, 1.5)
ax6c.set_ylim(0, 300)
ax6c.set_xticks(np.arange(0, 1.5, 0.2))
ax6c.set_yticks(np.arange(0,301, 50))
ax6c.set_xlabel(r'$\bar\ell$/$\ell_s$')
ax6c.set_ylabel(r'$\Sigma$($\theta_i$) (degrees)')
fig6c.savefig(Path.joinpath(out_dir, 'figure6c.tif'))



##
## Figure 6D:
##

fig6d, ax6d = plt.subplots(figsize=(3.25,3.25))
ax6d.tick_params(labelsize=10, direction='in', top=True, right=True)
ax6d.scatter(gaps_avg_sides, gaps_ang_sums, c='lightgrey', marker='.', s=1)
ax6d.scatter(gaps_r_10_08_08_avg_sides, gaps_r_10_08_08_ang_sums, c='k', marker='.', s=1)
ax6d.set_xlim(0, 1.5)
ax6d.set_ylim(0, 300)
ax6d.set_xticks(np.arange(0, 1.5, 0.2))
ax6d.set_yticks(np.arange(0,301, 50))
ax6d.set_xlabel(r'$\bar\ell$/$\ell_s$')
ax6d.set_ylabel(r'$\Sigma$($\theta_i$) (degrees)')
fig6d.savefig(Path.joinpath(out_dir, 'figure6d.tif'))



##
## Figure 6E:
##

fig6e, ax6e = plt.subplots(figsize=(3.25,3.25))
ax6e.tick_params(labelsize=10, direction='in', top=True, right=True)
ax6e.scatter(gaps_avg_sides, gaps_ang_sums, c='lightgrey', marker='.', s=1)
ax6e.scatter(gaps_r_10_12_12_avg_sides, gaps_r_10_12_12_ang_sums, c='k', marker='.', s=1)
ax6e.set_xlim(0, 1.5)
ax6e.set_ylim(0, 300)
ax6e.set_xticks(np.arange(0, 1.5, 0.2))
ax6e.set_yticks(np.arange(0,301, 50))
ax6e.set_xlabel(r'$\bar\ell$/$\ell_s$')
ax6e.set_ylabel(r'$\Sigma$($\theta_i$) (degrees)')
fig6e.savefig(Path.joinpath(out_dir, 'figure6e.tif'))




##
## Figure 6F:
##

## find where the average curvature of hte gap sides is 1 (i.e. the average of
## the circle radii is 1):
gaps_r_avg_10 = gaps_df[['r0', 'r1', 'r2']].mean(axis=1) == 1

## return angle sums from those:
gaps_r_avg_10_ang_sums = gaps_df[gaps_r_avg_10][['curv_angle01', 'curv_angle02', 'curv_angle12']].sum(axis=1)

## also return average sidelengths from those:
gaps_r_avg_10_avg_sides = gaps_df[gaps_r_avg_10][['side0', 'side1', 'side2']].mean(axis=1) / 1


fig6f, ax6f = plt.subplots(figsize=(3.25,3.25))
ax6f.tick_params(labelsize=10, direction='in', top=True, right=True)
ax6f.scatter(gaps_avg_sides, gaps_ang_sums, c='lightgrey', marker='.', s=1)
ax6f.scatter(gaps_r_avg_10_avg_sides, gaps_r_avg_10_ang_sums, c='k', marker='.', s=1)
ax6f.set_xlim(0, 1.5)
ax6f.set_ylim(0, 300)
ax6f.set_xticks(np.arange(0, 1.5, 0.2))
ax6f.set_yticks(np.arange(0,301, 50))
ax6f.set_xlabel(r'$\bar\ell$/$\ell_s$')
ax6f.set_ylabel(r'$\Sigma$($\theta_i$) (degrees)')
fig6f.savefig(Path.joinpath(out_dir, 'figure6f.tif'))



print('Done.')



