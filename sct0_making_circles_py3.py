# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:26:50 2016

Maintained on Sep 6, 2022 

@author: Serge
"""


from pathlib import Path
import numpy as np
import multiprocessing as multiproc

    


def make_circle_configs(circle_query_points, args):
    """Makes arrays of circles contain their x coordinates, y coordinates, radii, 
    separation distance between circle0 and circle1 (t1), separation distance between 
    circle0 and circle2 (t2), and the angle between those two lines defined by the 
    separation distances. The circles are oriented as follows: circle0 has an 
    origin of (0,0), circle2 has an origin that lies along the x-axis, and 
    circle1 has an origin that lies along a path produced by a circle with a 
    radius of t1 and an origin at (0,0). """

    phi = circle_query_points
    r1, r2, t1, t2, t3min, t3max, theta = args.T


    ## For circle0:
    ## get x and y coordinates for circle 0 at each angle 'phi':
    x0_arr = r0 * np.cos(phi) + 0.0
    y0_arr = r0 * np.sin(phi) + 0.0

    ## reshape these arrays of x and y coords so that they are 2D arrays with
    ## angles 'phi' located along the columns:
    x0_arr = x0_arr.reshape((1, len(phi)))
    y0_arr = y0_arr.reshape((1, len(phi)))
    
    ## make array of same shape, consisting of the radius of circle0:
    r0_arr = np.array(r0, ndmin=2)
    
    ## make array of same shape, consisting of the x and y coordinates of the
    ## origin of circle0:
    orix0_arr = np.ones((1, len(phi))) * 0.0
    oriy0_arr = np.ones((1, len(phi))) * 0.0


    ## Repeat for circle1:
    x1_arr = r1 * np.cos(phi) + t1 * np.cos(theta)
    y1_arr = r1 * np.sin(phi) + t1 * np.sin(theta)

    x1_arr = x1_arr.reshape((1, len(phi)))
    y1_arr = y1_arr.reshape((1, len(phi)))
    
    r1_arr = np.array(r1, ndmin=2)
    
    orix1_arr = np.ones((1, len(phi))) * (t1 * np.cos(theta))
    oriy1_arr = np.ones((1, len(phi))) * (t1 * np.sin(theta))


    ## Repeat for circle2:
    x2_arr = r2 * np.cos(phi) + t2
    y2_arr = r2 * np.sin(phi) + 0

    x2_arr = x2_arr.reshape((1, len(phi)))
    y2_arr = y2_arr.reshape((1, len(phi)))
    
    r2_arr = np.array(r2, ndmin=2)
    
    orix2_arr = np.ones((1, len(phi))) * t2
    oriy2_arr = np.ones((1, len(phi))) * 0.0
    

    ## put the separation distances and angles together in an array, we will 
    ## row-stack them with the other parameters:    
    t1_arr = np.array(t1, ndmin=2)
    t2_arr = np.array(t2, ndmin=2)
    theta_arr = np.array(theta, ndmin=2)

    t_theta_arr = np.r_[t1_arr, t2_arr, theta_arr]

    ## c0_yxr_temp = [circle0_y_pts, circle0_x_pts, circle0_r, t1, t2, theta, circle0_origin_x, circle0_origin_y]    
    c0_yxr = np.r_[y0_arr, x0_arr, np.ones((1,len(phi)))*r0_arr, np.ones((3,len(phi)))*t_theta_arr, orix0_arr, oriy0_arr]
    c1_yxr = np.r_[y1_arr, x1_arr, np.ones((1,len(phi)))*r1_arr, np.ones((3,len(phi)))*t_theta_arr, orix1_arr, oriy1_arr]
    c2_yxr = np.r_[y2_arr, x2_arr, np.ones((1,len(phi)))*r2_arr, np.ones((3,len(phi)))*t_theta_arr, orix2_arr, oriy2_arr]


    return (c0_yxr, c1_yxr, c2_yxr)






print('Preparing input and output directories...')
## get the location of this script "gaps1_fig5_eq4.py":
sct_dir = Path.cwd() # could also use sct_dir = Path('.').resolve()
## save the plot in the same folder that this script resides in:
out_dir = Path.joinpath(sct_dir, 'three_circles')

## if out_dir exists already, do nothing, otherwise, make it:
if Path.exists(out_dir):
    pass
else:
    Path.mkdir(out_dir)
    
    

print('Defining circle parameters...')
# size_of_r_linspace = How many radii you want to use for r1, r2.
# **Note: r0 is kept constant at 1.0, ie. r1 and r2 are normalized to r0
size_of_r_linspace = 9
# size_of_t_linspace = How many distances you want to use between the circles 0 and 1 (t1) and circles 0 and 2 (t2)
size_of_t_linspace = 9
# size_of_theta_linspace = How many distances you want to use between the circles 1 and 2 (t3; also expressed as an angle theta)
size_of_theta_linspace = 9
## circle_resolution = how many (x,y) points to construct the circle
circle_resolution = 360

# Total possibilities = number of r1 radii * number of r2 radii * number of t1 distances * number off t2 distances * number of different thetas (ie. t3 distances)
total_possibilities = size_of_r_linspace * size_of_r_linspace * size_of_t_linspace * size_of_t_linspace * size_of_theta_linspace


print('...the circle radii...')
## r0, r1, r2 = radius of circle1, circle2, circle3, etc.
r0 = 1.0#1.2
r1_arr = np.linspace(0.8, 1.2, size_of_r_linspace)
r2_arr = np.linspace(0.8, 1.2, size_of_r_linspace)

## curvatures of respective circles
curvature0 = 1.0 / r0
curvature1 = 1.0 / r1_arr
curvature2 = 1.0 / r2_arr



### Determine the minimum distance that the origins of each circle can move to:
    
## Initialize some empty arrays that we will fill up:
print('...the circle-to-circle distances for t1...')
   
## t1min = smallest line segment between origin0 and origin1
## t1max = largest line segment between origin0 and origin1
t1min = np.zeros(r1_arr.shape)
t1max = np.zeros(r1_arr.shape)

## Now fill up the empty arrays with smallest and largest separation distances:
for i,r1 in enumerate(r1_arr):
    ## We are going to say the smallest origin-origin separation distance to 
    ## bother simulating will be equal to half the smaller radius plus the 
    ## larger radius.
    ## if the radius of circle1 is smaller than that of circle0...
    if r0 >= r1:
        ## then this is the smallest line segment between them:
        t1min[i] = (r0 + 0.5 * r1)
        
    ## if the radius of circle1 is larger than that of circle0...
    elif r0 < r1:
        ## then this is the smallest line segment between them:
        t1min[i] = (0.5 * r0 + r1)
    
    ## the largest separation between circle0 and circle1 such that they still
    ## touch is:
    t1max[i] = r0 + r1
    
## pair the min and max distances between circle origins by stacking the 
## arrays vertically:
t1minmax = np.vstack((t1min, t1max)).T

## now it's time to make an array of equally distributed points between the 
## smallest and the largest separation distances:
t1_arr = np.zeros((len(t1min), size_of_t_linspace))
for (i,j),v in np.ndenumerate(t1minmax):
    ## each row in the array will consist of 'n' equally spaced points, where
    ## n = size_of_t_linspace
    t1_arr[i] = np.linspace(t1minmax[i,0], t1minmax[i,1], size_of_t_linspace)
    
## Append the associated radii of circle1 to the first column so that the 
## radii and the origin-origin separation distances info is kept together:
t1_arr = np.c_[r1_arr,t1_arr]




## Initialize some empty arrays that we will fill up:
print('...the circle-to-circle distances for t2...')

## t2min = smallest line segment between origin0 and origin2
## t2max = largest line segment between origin0 and origin2
t2min = np.zeros(r2_arr.shape)
t2max = np.zeros(r2_arr.shape)

## Now fill up the empty arrays with smallest and largest separation distances:
for i,r2 in enumerate(r2_arr):
    ## We are going to say the smallest origin-origin separation distance to 
    ## bother simulating will be equal to half the smaller radius plus the 
    ## larger radius.
    ## if the radius of circle2 is smaller than that of circle0...
    if r0 >= r2:
        ## then this is the smallest line segment between them:
        t2min[i] = (r0 + 0.5 * r2)
        
    ## if the radius of circle2 is larger than that of circle0...    
    elif r0 < r2:
        ## then this is the smallest line segment between them:
        t2min[i] = (0.5 * r0 + r2)
        
    ## the largest separation between circle0 and circle2 such that they still
    ## touch is:
    t2max[i] = r0 + r2

## pair the min and max distances between circle origins by stacking the 
## arrays vertically:
t2minmax = np.vstack((t2min, t2max)).T

## now it's time to make an array of equally distributed points between the 
## smallest and the largest separation distances:
t2_arr = np.zeros((len(t2min), size_of_t_linspace))
for (i,j),v in np.ndenumerate(t2minmax):
    ## each row in the array will consist of 'n' equally spaced points, where
    ## n = size_of_t_linspace
    t2_arr[i] = np.linspace(t2minmax[i,0], t2minmax[i,1], size_of_t_linspace)
    
## Append the associated radii of circle1 to the first column so that the 
## radii and the origin-origin separation distances info is kept together:
t2_arr = np.c_[r2_arr,t2_arr]



## Initialize some empty arrays that we will fill up:
print('...the circle-to-circle distances for t3...')

## t3min = smallest line segment between origin0 and origin2
## t3max = largest line segment between origin0 and origin2

## The variable 't3minmax' is an array with the following information:
## t3minmax = (r1, r2, t1, t2, t3min, t3max)
t3minmax = np.zeros((1,6))
for (i,j),t1 in np.ndenumerate(t1_arr[:,1:]):
    for (p,q),t2 in np.ndenumerate(t2_arr[:,1:]):
        if t1 < t2:
            # print t1[i,0], t2[p,0], 0.5*t1[i,0]+t2[p,0], t1[i,0]+t2[p,0]
            temp = np.asarray([t1_arr[i,0], t2_arr[p,0], t1, t2, 0.5*t1_arr[i,0]+t2_arr[p,0], t1_arr[i,0]+t2_arr[p,0]])
            temp = np.reshape(temp, (1,6))
            t3minmax = np.r_[t3minmax,temp]
        elif t1 >= t2:
            # print t1[i,0], t2[p,0], t1[i,0]+0.5*t2[p,0], t1[i,0]+t2[p,0]
            temp = np.asarray([t1_arr[i,0], t2_arr[p,0], t1, t2, t1_arr[i,0]+0.5*t2_arr[p,0], t1_arr[i,0]+t2_arr[p,0]])
            temp = np.reshape(temp, (1,6))
            
            ## append the initial t3minmax array with the new information by 
            ## rowstacking the new info onto the existing t3minmax array:
            t3minmax = np.r_[t3minmax,temp]

## The first row is the initial array (consisting of only zeros) that we used
## to start stacking the other, information-carrying t3minmax rows, so we will
## now delete it since it is no longer needed:
t3minmax = np.delete(t3minmax, 0 ,0)
  



## the array thetaminmax contains the following information:
## thetaminmax = (r1, r2, t1, t2, t3min, t3max, thetamin, thetamax)
thetaminmax = np.copy(t3minmax)

## now calculate the angle associated with t3min and t3max. 
## To do so we use the law of cosines to calculate the max and min angle:
## a**2.0 = b**2.0 + c**2.0 - 2.0 * b * c *np.cos(A)
thetamax = np.arccos( np.round((t3minmax[:,2]**2.0 + t3minmax[:,3]**2.0 - t3minmax[:,5]**2.0), decimals=10) / np.round((2.0 * t3minmax[:,2] * t3minmax[:,3]), decimals=10 ) )
thetamin = np.arccos( np.round((t3minmax[:,2]**2.0 + t3minmax[:,3]**2.0 - t3minmax[:,4]**2.0), decimals=10) / np.round((2.0 * t3minmax[:,2] * t3minmax[:,3]), decimals=10 ) )

## pairing the minimum and maximum angle information together with associated 
## circle radii, and t1, t2, and t3minmax distances by column-stacking them:
thetaminmax = np.c_[thetaminmax, thetamin, thetamax]


## now we will create an populate an array that contains equal-spaced points
## between the smallest and largest angle between circle1 and circle2:
theta_arr = np.zeros((len(thetamin),size_of_theta_linspace))
for (i,j),v in np.ndenumerate(thetaminmax[:,6:8]):
    theta_arr[i] = np.linspace(thetaminmax[i,6], thetaminmax[i,7], size_of_theta_linspace)

## now we will make an array containing only information important to creating
## the circles by stacking:
## master_variable_array = ['r1', 'r2', 't1', 't2', 't3min', 't3max', 'theta']
# master_variable_array = np.zeros((1,7))
# for (i,j),v in np.ndenumerate(theta_arr):
#     mstr_var_arr_temp = np.r_[t3minmax[i,:],v]
#     mstr_var_arr_temp = mstr_var_arr_temp.reshape((1,7))
#     master_variable_array = np.r_[master_variable_array, mstr_var_arr_temp]

# ## delete the base array (contains only zeros):
# master_variable_array = np.delete(master_variable_array, 0, 0)


print('Creating the master variable list...')
master_var_list = list()
for (i,j),v in np.ndenumerate(theta_arr):
    mstr_var_arr_temp = np.r_[t3minmax[i,:],v]
    mstr_var_arr_temp = mstr_var_arr_temp.reshape((1,7))

    master_var_list.append(mstr_var_arr_temp)




### Now you have multiple values for radius1 & radius 2 (r1 & r2), t1 & t2, and
### theta. It's time to use this information to draw the circles and find their
### points of intersection.
## evenly distribute angles from 0radians to 2*pi radians (ie. a whole circle):
print('Creating circle subdivisions...')
phi = np.linspace(0.0, 2.0 * np.pi, circle_resolution, endpoint = False)




## A point on a circle can be uniquely defined by a y coordinate, x coordinate, 
## and the origin of the circle.
## Set initial circle lists:
circle0 = list()
circle1 = list()
circle2 = list()

## Set the initial counter:
count = 0
for params in master_var_list:
    ## params are the parameters of the three circles.
    
    ## This if statement is only used for development, during actual execution
    ## it should be commented out:
    # if count > 100:
    #     break


    print('making circle config ', count, '/' , total_possibilities-1)

    ## params = parameters of the three circles
    
    results = make_circle_configs(phi, params)

    circle0_tmp, circle1_tmp, circle2_tmp = results
    
    circle0.append(circle0_tmp)
    circle1.append(circle1_tmp)
    circle2.append(circle2_tmp)

    ## advance counter:
    count += 1


## convert circle lists into depth-stacked arrays:
circle0_yxr = np.dstack(circle0)
circle1_yxr = np.dstack(circle1)
circle2_yxr = np.dstack(circle2)

## save the resulting numpy arrays:
np.save(Path.joinpath(out_dir, 'circle0_yxr'), circle0_yxr)
np.save(Path.joinpath(out_dir, 'circle1_yxr'), circle1_yxr)
np.save(Path.joinpath(out_dir, 'circle2_yxr'), circle2_yxr)


input("Done. Press 'Enter' to exit. \n>")





    
    
    
    
    
    





    


    



