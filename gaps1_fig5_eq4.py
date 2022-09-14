# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 16:36:53 2016
- originally created using Python 2

Last maintained on Fri Sept 02 15:37 2022
- code was updated to work in Python 3
- original lacked environment file (the '.yml' file), therefore one was made
- some comments were added to improve clarity
- unused code was removed to improve clarity
- relative directory locations were added so that plot would save correctly on
    another persons device

@author: Serge
"""

from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
labelfont = {'fontname':'Times New Roman'}

## get the location of this script "gaps1_fig5_eq4.py":
sct_dir = Path.cwd() # could also use sct_dir = Path('.').resolve()
## save the plot in the same folder that this script resides in:
out_dir = sct_dir



### Equation 4:

min_yax_angle = 0
max_yax_angle = 45 
angles = np.linspace(min_yax_angle, max_yax_angle, 50)
theta = np.linspace(min_yax_angle, np.deg2rad(max_yax_angle), 50)
Rs = np.linspace(1, 1, 50)
betai = np.linspace(0.9, 0.9, 50)
beta = np.linspace(1,1,50)


angle_term_only = np.cos(theta) - np.sqrt(3.0) * np.sin(theta)
ls = ( np.cos(theta) - np.sqrt(3.0) * np.sin(theta) ) * Rs

for index, value in enumerate(theta):
    if theta[index] <= np.deg2rad(30):
        ls[index] *= 1
    elif theta[index] > np.deg2rad(30):
        ls[index] *= -1


### pi is meant to be pressure i, not pi (3.1415...)
pc = 2.0 * beta / Rs #=2 if beta=1 and Rs=1
pi = np.linspace(1,1,50)
Ri = betai / (pc - pi)
li = ( np.cos(theta) - np.sqrt(3.0) * np.sin(theta) ) * Ri

pi_0 = 0
Ri_pi_0 = betai / (pc - pi_0)
li_pi_0 = ( np.cos(theta) - np.sqrt(3.0) * np.sin(theta) ) * Ri_pi_0

for index, value in enumerate(theta):
    if theta[index] <= np.deg2rad(30):
        li_pi_0[index] *= 1
    elif theta[index] > np.deg2rad(30):
        li_pi_0[index] *= -1


dif_pi = np.asarray([1.0, 1.2, 1.4, 1.6, 1.8])
dif_pi = np.resize(dif_pi, (50, len(dif_pi)))
dif_pi = dif_pi.T


dif_Ri_08 = np.linspace(0.8, 0.8, 50)
dif_Ri_11 = np.linspace(1.1, 1.1, 50)
dif_Ri_12 = np.linspace(1.2, 1.2, 50)
dif_Ri_13 = np.linspace(1.3, 1.3, 50)
dif_Ri_14 = np.linspace(1.4, 1.4, 50)
dif_Ri_15 = np.linspace(1.5, 1.5, 50)
dif_Ri_16 = np.linspace(1.6, 1.6, 50)
dif_Ri = np.vstack([dif_Ri_08, dif_Ri_11,dif_Ri_12,dif_Ri_13,dif_Ri_14,dif_Ri_15,dif_Ri_16])


dif_li = ( np.cos(theta) - np.sqrt(3.0) * np.sin(theta) ) * dif_Ri

for index, value in enumerate(theta):
    if theta[index] <= np.deg2rad(30):
        dif_li[:,index] *= 1
    elif theta[index] > np.deg2rad(30):
        dif_li[:,index] *= -1







fig = plt.figure(figsize=(4,3.25))
ax = fig.add_subplot(111, adjustable='box')
plt.axis((0,1.5,min_yax_angle,max_yax_angle))
ax.tick_params(labelsize=10, direction='in', top=True)
ax.set_xticks(np.linspace(0, 1.4, 8))
ax.set_xlabel(r'$\ell / \ell_s$', fontsize=12, **labelfont)
ax.set_ylabel(r'$\theta_i$ (degrees)', fontsize=12, **labelfont)

ax2 = ax.twinx()
ax2.tick_params(labelsize=10, direction='in', top=True)
ax2_ytickmin = 1.0-np.cos(np.deg2rad(0))
ax2_ytickmax = 1.0-np.cos(np.deg2rad(30))
ax2_yticks = np.linspace(ax2_ytickmin, ax2_ytickmax, 7)
ax2.set_ylim([ax2_ytickmin, ax2_ytickmax])
for i,x in enumerate(ax2_yticks):
    ax2_yticks[i] = round(x,3)
ax2.set_yticks(range(7))
ax2.set_yticklabels(ax2_yticks)
ax2.set_ylabel(r'$\alpha_i$', fontsize=12, rotation=270, labelpad = 15, **labelfont)

ax.plot(ls, np.rad2deg(theta), linestyle='--', lw=1, c = 'k')
ax.plot(li_pi_0, np.rad2deg(theta), linestyle=':', lw=1, c='k')
ax.plot(dif_li[0], np.rad2deg(theta), linestyle='-', lw=1, c='k')
ax.plot(dif_li[1], np.rad2deg(theta), linestyle='-', lw=1, c='k')
ax.plot(dif_li[2], np.rad2deg(theta), linestyle='-', lw=1, c='k')
ax.plot(dif_li[3], np.rad2deg(theta), linestyle='-', lw=1, c='k')
ax.plot(dif_li[4], np.rad2deg(theta), linestyle='-', lw=1, c='k')
ax.plot(dif_li[5], np.rad2deg(theta), linestyle='-', lw=1, c='k')
ax.hlines(30, 0, 1.5, linestyle='-.', color='k')
ax.vlines(1.0, min_yax_angle, max_yax_angle, linestyle='-.', color='k')


plt.tight_layout()
fig.savefig(Path.joinpath(out_dir, 'symm_gap_sizeVSangle.tif'), format='tif', dpi=1000)






