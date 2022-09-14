# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 17:06:53 2016
Maintained on Thurs Sept 08, 2022


This script essentially works like this:
1. find all 6 intersection points between the three partially 
    overlapping circle (2 points per pair of circles)
2. identify which of the 8 unique triangles that are formed from 
    connecting the points from each of the three intersections makes a
    triangle that represents the gap (hint: it's the one with the
    smallest area)
3. take those intersection points and find the angles between straight
    lines that connect each of them, and also angles between the curved
    surfaces that connect each of them, as well as the areas and 
    perimeters
4. put all of this information (as well as the radii and origin points)
    that define the circle configuration) into an array and save it.

    
The maintenance that was performed changed the model from producing 
circle-circle intersections points numerically, to solving for those points 
analytically. As a result, the accuracy of the gaps has dramatically improved 
and only 2/59049 gaps configurations appear to be outliers now. They appear 
correct when plotted, but the curved angles and sidelengths seem to be 
unexpectedly small, and this may be due to floating point precision I think...

@author: Serge
"""



from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
# import multiprocessing as multiproc
# import gc


def vector(pt1, pt2):
    ## The first point (pt1) is treated as the origin (or tail) of the vector.
    x1, y1 = pt1
    x2, y2 = pt2
    return np.array((x2-x1, y2-y1))
    

def angle(vector1, vector2):
    ## Finding the angle between two vectors using the dot product. Need to be
    ## sure that the vectors are oriented correctly:
    ## floating point errors can result in trying to take the arccos of a 
    ## number ever-so-slightly larger than 1 (when it is actually 1), in the 
    ## case of circles touching at points, therefore catch that scenario:
    dot_v1v2 = np.dot(vector1, vector2)
    magn_v1v2 = ( np.linalg.norm(vector1) * np.linalg.norm(vector2) )
    if (dot_v1v2 / magn_v1v2) > 1:
        ang = 0
    else: 
        ang = np.arccos( dot_v1v2 / magn_v1v2 ) 
    return ang


def circle_deriv(pt, origin_pt):
    ## This function will return a vector that is perpendicular to the vector 
    ## that goes from origin_pt to pt. If the vector is from the origin of a 
    ## circle to a point on the circle (as it is in this script), then the 
    ## result will be a vector that is tangent to the circle. 
    x, y = pt
    x0, y0 = origin_pt
    return (x0 - x) / (y - y0)


def slope_to_unit_vecs(point, slope):
    """Takes an (x,y) point and a (scalar) slope and creates two unit 
    vectors pointing away from the point (one vector in each direction."""
    ## N.B. the magnitude could be figured out for just 1 vector and then
    ## two unit vectors could be constructed afterwards. 
    
    ## Note that the slope could be infinity (i.e. vertical), and we will need
    ## to catch that possibility, so:
    if np.isfinite(slope):
        x0, y0 = point
        x1 = x0 + 1 # the number 1 is arbitrary, but the sign is not.
        x2 = x0 - 1 # the number 1 is arbitrary, but the sign is not.
        
        ## equation of a line:
        y1 = slope * (x1 - x0) + y0
        y2 = slope * (x2 - x0) + y0
        
        ## construct vectors of arbitrary magnitude:
        vec1 = np.array((x1 - x0, y1 - y0))
        vec2 = np.array((x2 - x0, y2 - y0))
        
        ## get the magnitude of the vectors
        vec1_magn = np.linalg.norm(vec1)
        vec2_magn = np.linalg.norm(vec2)
        
        ## create the unit vectors:
        unit_vec1 = vec1 / vec1_magn
        unit_vec2 = vec2 / vec2_magn
    
    else:
        ## if the slope is infinite, then the unit vector will be either
        ## one unit up or one unit down:
        unit_vec1 = np.array([0, 1])
        unit_vec2 = np.array([0, -1])
    
    return unit_vec1, unit_vec2


def get_circle_intersection_pts(circle1_origin, circle1_radius, circle2_origin, circle2_radius):
    
    ## figure out what the origin-to-origin distance between the circles is:
    orix1, oriy1 = circle1_origin
    orix2, oriy2 = circle2_origin
    
    ## 't' is the origin-to-origin distance:
    t = np.sqrt( (orix1 - orix2)**2 + (oriy1 - oriy2)**2 )
    ## note that this is the square root, so 't' could be either positive or 
    ## negative. This is important since the relative origin can be further
    ## along the x-axis then the other circle (meaning that the x-intersect
    ## should be negative). 
    # t = np.around(t, decimals=np.finfo(np.double).precision)
    # t = np.around(t, decimals=8)
    ## unfortunately floating point errors cause problems when the circles are 
    ## touching at a point, therefore, while not ideal, I'm rounding the answer
    ## a little. This should hopefully resolve the issue. 
    ## It does not resolve the issue, instead I will try to catch it at np.sqrt
    
    ## figure out which circle has the smaller radius:
    if circle1_radius <= circle2_radius:
        r = circle1_radius
        R = circle2_radius
        ## note that in this case the origin for circle2 is (0,0) when 
        ## considering rel_x 
        rel_orix, rel_oriy = circle2_origin
        other_orix, other_oriy = circle1_origin
    else:
        r = circle2_radius
        R = circle1_radius
        ## note that in this case the origin for circle1 is (0,0) when 
        ## considering rel_x (it is always the larger radius).
        rel_orix, rel_oriy = circle1_origin
        other_orix, other_oriy = circle2_origin
        
    ## 't' is the separation distance between each circle origin, 'r' is the 
    ## radius of the smaller circle, and 'R' is the radius of the larger circle
    ## (this equation comes from: 
    ## https://mathworld.wolfram.com/Circle-CircleIntersection.html)
    ## to assign the proper sign to 't' (either positive or negative), we just
    ## determine if the relative origin is to the right or left of the other 
    ## circle:
    ## if rel_orix - other_orix is positive then rel_orix is to the right, and
    ## 't' is negative, otherwise 't' is positive:
    t_sign = -1 * np.sign(rel_orix - other_orix)
    t *= t_sign

    ## now we can find the relative x intercept
    rel_x = (t**2 - r**2 + R**2) / (2 * t)
    
    
    ## the square of the y intersection is represented by
    ## rel_y_inters**2 = ( 4 * t**2 * R**2 - (t**2 - r**2 + R**2)**2 ) / (4 * t**2)
    ## and taking the square root of this can be problematic if the 
    ## intersection is very nearly zero, therefore, if rel_y_inters**2 is 
    ## so small that it becomes negative, we will consider it zero, \
    ## otherwise we will find it properly:
    rel_y_inters_sq = ( 4 * t**2 * R**2 - (t**2 - r**2 + R**2)**2 ) / (4 * t**2)
    if rel_y_inters_sq < 0:
        rel_y_inters1 = 0
        rel_y_inters2 = 0
    else:
        ## the y intersection is represented by
        rel_y_inters1 = np.sqrt( ( 4 * t**2 * R**2 - (t**2 - r**2 + R**2)**2 ) / (4 * t**2) )
        rel_y_inters2 = -1 * np.sqrt( ( 4 * t**2 * R**2 - (t**2 - r**2 + R**2)**2 ) / (4 * t**2) )    
    
    ## the above equations are for finding the relative x,y points of the 
    ## intersections assume that the circles origins lie along the x-axis, 
    ## therefore we need to (mathematically) rotate and translate the x,y 
    ## points such that they are consistent with the origins of circle0-2.
    
    ## first we need to get the angle between the x-axis and the line 't':
    theta = np.arctan( (oriy1 - oriy2) / (orix1 - orix2) )
    
    ## then we need to calculate the rotated points:
    ## we shift the pair of circles such that rel_x is the origins x-coordinate
    ## i.e. rel_x = 0 now, and leave the rel_y coordinates the same. 
    ## Then we use the rotation of theta to calculate how relative x and y 
    ## coordinates have shifted. 
    ## Also keep in mind that the radius of rotation is equal to the 
    ## rel_y_inters (since it is directly vertically above the origin), and 
    ## when the angle theta = 0, then the x intercept is not shifted and is
    ## equal to 0:
    # alpha = np.pi/2 + theta #; i.e. alpha is perpendicular to theta
    # rot_rel_x1 = rel_y_inters1 * np.cos(alpha)
    # rot_rel_x2 = rel_y_inters2 * np.cos(alpha)
    # rot_rel_y1 = rel_y_inters1 * np.sin(alpha)
    # rot_rel_y2 = rel_y_inters2 * np.sin(alpha)
    ## using some trigonometry we rewrite in terms of theta
        ## some more breadcrumbs for re-deriving this: 
        ## theta = -1 * phi
        ## sin(pi/2 - phi) = cos(phi), cos(pi/2 - phi) = sin(phi)
        ## sin(-theta) = -1 * sin(theta), cos(-theta) = cos(theta)
    rot_rel_x1 = rel_y_inters1 * -1 * np.sin(theta)
    rot_rel_x2 = rel_y_inters2 * -1 * np.sin(theta)
    rot_rel_y1 = rel_y_inters1 * np.cos(theta)
    rot_rel_y2 = rel_y_inters2 * np.cos(theta)

    ## now that we have the rotated intersection points, we need to shift them
    ## back towards their locations with respect to ori0, ori1, ori2:
    ## to do so, we use the origin of the circle used for determining where 
    ## rel_x was to shift the rel_x intersection point over, and then shift it
    ## further according to the rotation of the circle configuration:
    t_x = rel_orix + rel_x * np.cos(theta)
    t_y = rel_oriy + rel_x * np.sin(theta)
    
        
    ## and now shift the rotated relative x and y coordinates such that their
    ## rotation origin of (0,0) is now (t_x, t_y):
    x1_inters = t_x + rot_rel_x1
    y1_inters = t_y + rot_rel_y1
    x2_inters = t_x + rot_rel_x2
    y2_inters = t_y + rot_rel_y2
    
        
    return (x1_inters, y1_inters), (x2_inters, y2_inters)


def get_circle_line_intersection_pts(circle_origin_xy, circle_radius, pt_on_line, line_vector):
    """This function will return the points where a line intersects with a 
    circle as two (x,y) points. If no intersections are found, then np.nan is 
    returned."""
    
    ori_x, ori_y = circle_origin_xy
    r = circle_radius
    pt_x, pt_y = pt_on_line
    # line_vector_norm = line_vector / np.linalg.norm(line_vector)
    # vec_x, vec_y = line_vector_norm
    vec_x, vec_y = line_vector
 
    
    ## first write the parametric equation of the line:
    ## <x,y> = <pt_x, pt_y> + t * <vec_x, vec_y>
    ## > x = pt_x + t * vec_x
    ## > y = pt_y + t * vec_y
    
    ## equation of a circle:
    ## r**2 = (x - ori_x)**2 + (y - ori_y)**2
    ## > (x - ori_x)**2 + (y - ori_y)**2 - r**2 = 0
    ## expand the equation of a circle:
    ## 0 = x**2 - 2 * x * ori_x + ori_x**2 + y**2 - 2 * y * ori_y + ori_y**2 - r**2
        
    ## plug x and y from the equation of the line into the expannded equation 
    ## of the circle:
    ## 0 = [(pt_x + t * vec_x)**2 - 2 * ori_x * (pt_x + t * vec_x) + ori_x**2] 
    ##     + [(pt_y + t * vec_y)**2 - 2 * ori_y * (pt_y + t * vec_y) + ori_y**2]
    ##     - r**2
    
    ## expand further:
    ## 0 = [pt_x**2 + 2 * pt_x * t * vec_x + t**2 * vec_x**2 - 2 * ori_x * (pt_x + t * vec_x) + ori_x**2]
    ##     + [pt_y**2 + 2 * pt_y * t * vec_y + t**2 * vec_y**2 - 2 * ori_y * (pt_y + t * vec_y) + ori_y**2]
    ##     - r**2
    
    ## expand more still:
    ## 0 = [pt_x**2 + 2 * pt_x * t * vec_x + t**2 * vec_x**2 - 2 * ori_x * pt_x - 2 * ori_x * t * vec_x + ori_x**2]
    ##     + [pt_y**2 + 2 * pt_y * t * vec_y + t**2 * vec_y**2 - 2 * ori_y * pt_y - 2 * ori_y * t * vec_y + ori_y**2]
    ##     - r**2
    
    ## group the 't's and constants together:
    ## 0 = t**2 * (vec_x**2 + vec_y**2) 
    ##     + 2 * t * (pt_x * vec_x + pt_y * vec_y - vec_x * ori_x - vec_y * ori_y) 
    ##     + pt_x**2 + pt_y**2 - 2 * pt_x * ori_x - 2 * pt_y * ori_y  + ori_x**2 + ori_y**2 - r**2

    ## to make things clearer, let:
    ## A = (vec_x**2 + vec_y**2)
    ## B = 2 * (pt_x * vec_x + pt_y * vec_y - vec_x * ori_x - vec_y * ori_y)
    ## C = pt_x**2 + pt_y**2 - 2 * pt_x * ori_x - 2 * pt_y * ori_y  + ori_x**2 + ori_y**2 - r**2
    A = (vec_x**2 + vec_y**2)
    B = 2 * (pt_x * vec_x + pt_y * vec_y - vec_x * ori_x - vec_y * ori_y)
    C = pt_x**2 + pt_y**2 - 2 * pt_x * ori_x - 2 * pt_y * ori_y  + ori_x**2 + ori_y**2 - r**2
    
    ## giving us:
    ## 0 = A * t**2 + 2 * B * t + C
    
    ## now let's solve the quadratic equation:
    ## quadratic formula:
    ## t = -B +/- sqrt(B**2 - 4*A*C) / (2*A)
    ## Note that in principle there could be no intersection (although that
    ## shouldn't happen in this simulation), so we will catch any errors that 
    ## would arise from that situation with a try-except statement:
    with np.errstate(invalid='raise'):
        try:
            t1 = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
        except FloatingPointError:
            t1 = np.nan
        try:
            t2 = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
        except FloatingPointError:
            t2 = np.nan
            
    ## now we can try plugging the t parameters back in to find the line-circle
    ## intersection points:
    x1 = pt_x + t1 * vec_x
    y1 = pt_y + t1 * vec_y
    
    x2 = pt_x + t2 * vec_x
    y2 = pt_y + t2 * vec_y
    
    # print((x1,y1), (x2,y2))
    return (x1, y1), (x2, y2)


def get_closest_point(pt_on_line, vector_of_line, pt_outside_line):
    """Will return the point on a line that is closest to another point. 
    Takes a point on a line and the vector that indicates the direction of 
    the line, and lastly also requires the point outside the line. 
    Only works for 2D points and vectors. Points and vectors must be given as
    (x,y) tuples or arrays of size 2."""
    
    ln_pt_x, ln_pt_y = pt_on_line
    vec_x, vec_y = vector_of_line
    pt_x, pt_y = pt_outside_line
    
    ## closest_pt = <x,y>
    ## here is the parametric equation for the line:
    ## <x,y> = <ln_pt_x, ln_pt_y> + t * <vec_x, vec_y>
   
    ## the fot product of the line that passes through the closest point <x,y> 
    ## to the query point <pt_x, pt_y> and the line parallel to the 
    ## vector_of_line is 0 (since they are perpendicular, as the closest point
    ## forms a line that is perpendicular to vector_of_line; see
    ## https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html):
    ## (<x,y> - <pt_x, pt_y>) . <vec_x, vec_y> = 0
    
    ## find the parameter 't':
    ## first writing out an equation for the left side of the dot product: 
        ## let the left side = a for brevity:
    ## a . <vec_x, vec_y> = 0
    ## a = <ln_pt_x, ln_pt_y> + t * <vec_x, vec_y> - <pt_x, pt_y>
    ## a = <ln_pt_x - pt_x + t * vec_x, ln_pt_y - pt_y + t * vec_y>

    ## now, going back to the dot product:
    ## a . <vec_x, vec_y> = 0
    ## FYI: a = <a_x, a_y>
    
    ## the dot product between two orthogonal lines is 0, and the dot product is
    ## defined as vec1.vec2 = sum(vec1_i * vec2_i) from i=0 to i=n, where n is the
    ## number of dimensions that the vector has ( e.g. a 2D vector has x and y, so
    ## vec1.vec2 = (vec1_x * vec2_x) + (vec1_y * vec2_y) )
    
    ## (a_x * vec_x) + (a_y * vec_y) = 0
    ## (a_x * vec_x) = 0 - (a_y * vec_y)
    
    ## now, expanding back out a_x and a_y and solving for 't':
    
    ## (ln_pt_x - pt_x + t * vec_x) * vec_x = 0 - (ln_pt_y - pt_y + t * vec_y * vec_y)
    ## for brevity let c01_inters_x - tri_center_x = C1
    ## and let c01_inters_y - tri_center_y = C2
    
    ## (ln_pt_x - pt_x) * vec_x + t * vec_x**2 = 0 - ((ln_pt_y - pt_y) * vec_y + t * vec_y**2)
    ## (ln_pt_x - pt_x) * vec_x + t * vec_x**2 = -1 * (ln_pt_y - pt_y) * vec_y - t * vec_y**2
    ## t * vec_x**2 = -1 * (ln_pt_y - pt_y) * vec_y - t * vec_y**2 - (ln_pt_x - pt_x) * vec_x
    ## t * vec_x**2 + t * vec_y**2 = -1 * (ln_pt_y - pt_y) * vec_y - (ln_pt_x - pt_x) * vec_x
    ## t * (vec_x**2 + vec_y**2) = -1 * (ln_pt_y - pt_y) * vec_y - (ln_pt_x - pt_x) * vec_x
    ## t = (-1 * (ln_pt_y - pt_y) * vec_y - (ln_pt_x - pt_x) * vec_x) / (vec_x**2 + vec_y**2)
    
    
    t = (-1 * (ln_pt_y - pt_y) * vec_y - (ln_pt_x - pt_x) * vec_x) / (vec_x**2 + vec_y**2)
    
    ## plugging t back in to solve for <x,y>:
    
    closest_pt_xy = pt_on_line + t * vector_of_line
    
    return closest_pt_xy






print('Preparing input and output directories...')
## get the location of this script "gaps1_fig5_eq4.py":
sct_dir = Path.cwd() # could also use sct_dir = Path('.').resolve()
## save the plot in the same folder that this script resides in:
out_dir = Path.joinpath(sct_dir, 'gap_data')

## if out_dir exists already, do nothing, otherwise, make it:
if Path.exists(out_dir):
    pass
else:
    Path.mkdir(out_dir)



## Get the directory where the three circles parameter data is stored:
filepath = Path.joinpath(sct_dir, 'three_circles')


## Load the circles that were output by 'making_circles.py':
circle0_yxr = np.load(Path.joinpath(filepath, 'circle0_yxr.npy'))
circle1_yxr = np.load(Path.joinpath(filepath, 'circle1_yxr.npy'))
circle2_yxr = np.load(Path.joinpath(filepath, 'circle2_yxr.npy'))



## Retrieve the total number of circle combinations. This will be used to count
## the progress of the script. 
total_possibilities = circle0_yxr.shape[-1]







## make a list to store all the information we find about the gaps:
gap = list()

## start iterating through all the gap configurations:
for k in range(total_possibilities):

    # test variables
    # k = 6480
    # k = 52560
    #
    

    
    ## assign essential parameters of circles:
    ori0 = (circle0_yxr[6,0,k], circle0_yxr[7,0,k])
    ori1 = (circle1_yxr[6,0,k], circle1_yxr[7,0,k])
    ori2 = (circle2_yxr[6,0,k], circle2_yxr[7,0,k])
    
    r0 = circle0_yxr[2,0,k]
    r1 = circle1_yxr[2,0,k]
    r2 = circle2_yxr[2,0,k]
    
    
    
    ## start by getting the circle-circle intersections:
    c01_inters1, c01_inters2 = get_circle_intersection_pts(ori0, r0, ori1, r1)
    c02_inters1, c02_inters2 = get_circle_intersection_pts(ori0, r0, ori2, r2)
    c12_inters1, c12_inters2 = get_circle_intersection_pts(ori1, r1, ori2, r2)
   
        
    
    ## describe the 8 possible triangles from the 6 intersection points above:
    possible_8_triangles = list()
    
    
    
    ## each of the three circle-circle intersections has 2 intersection points,
    ## so if we make each pair of intersection points a side of a cube, similar to 
    ## how a Punnett square is drawn, we will get 8 possible triangles
    ## (a 2x2x2 cube), and one of these will describe the tricellular gap:
    print('finding gap for circle config ', k, '/',  total_possibilities-1)
    
    ## first we zip the (x,y) intersection points of intersections 1 and 2 so that 
    ## it returns the x coordinates and y coordinates in separate lists, and then
    ## turns those two lists into their own arrays:
    c01_inters_pts_x, c01_inters_pts_y = np.array(list(zip(c01_inters1, c01_inters2)))
    c02_inters_pts_x, c02_inters_pts_y = np.array(list(zip(c02_inters1, c02_inters2)))
    c12_inters_pts_x, c12_inters_pts_y = np.array(list(zip(c12_inters1, c12_inters2)))        
    
    
    c01_inters_x_arr, c02_inters_x_arr, c12_inters_x_arr = np.meshgrid(c01_inters_pts_x, c02_inters_pts_x, c12_inters_pts_x)
    c01_inters_y_arr, c02_inters_y_arr, c12_inters_y_arr = np.meshgrid(c01_inters_pts_y, c02_inters_pts_y, c12_inters_pts_y)
    
        
    ## we're using the equation for sidelength in an element-wise fashion on each 
    ## array now. Each element in our size-8 array corresponds to one sidelength of
    ## one of the 8 triangles:
    
    side0_arr = np.sqrt((c01_inters_x_arr - c02_inters_x_arr)**2 + (c01_inters_y_arr - c02_inters_y_arr)**2)
    side1_arr = np.sqrt((c01_inters_x_arr - c12_inters_x_arr)**2 + (c01_inters_y_arr - c12_inters_y_arr)**2)
    side2_arr = np.sqrt((c02_inters_x_arr - c12_inters_x_arr)**2 + (c02_inters_y_arr - c12_inters_y_arr)**2)
    
    s_arr = 0.5 * (side0_arr + side1_arr + side2_arr)
    tri_perim_arr = 2.0 * s_arr
    tri_area_arr = np.sqrt(s_arr * (s_arr - side0_arr) * (s_arr - side1_arr) * (s_arr - side2_arr))
    
    ## each circle configuration has a triangle that describes a gap, we determine 
    ## which of the 8 triangles above is the one that describes the gap based on 
    ## the fact that the gap will be described by the circle configuration with the
    ## smallest area and perimeter:    
    gap_index = np.where(tri_perim_arr == np.min(tri_perim_arr))
    
    ## if the circles touch at a point, then c##_inters1 will be identical to
    ## c##_inters2, and this will result in gap_index returning two minima, 
    ## which causes issues, therefore we will only keep the first index:
    
    ## testing if there is more than one minima index returned:
    if np.shape(gap_index)[-1] > 1:
        ## picking out just the first point, ensuring that the output is the
        ## same as what is returned by np.where:
        gap_index = tuple([np.array([arr[0]]) for arr in gap_index])
        
        
    
    
    ## now that we have the index of the triangle that corresponds to the gap, we 
    ## will get other relevant information about the gap:
    c01_inters_x = float(c01_inters_x_arr[gap_index])
    c01_inters_y = float(c01_inters_y_arr[gap_index])
    c02_inters_x = float(c02_inters_x_arr[gap_index])
    c02_inters_y = float(c02_inters_y_arr[gap_index])
    c12_inters_x = float(c12_inters_x_arr[gap_index])
    c12_inters_y = float(c12_inters_y_arr[gap_index])
    ## these intersection points are presented as (x,y):
    c01_inters = (c01_inters_x, c01_inters_y)
    c02_inters = (c02_inters_x, c02_inters_y)
    c12_inters = (c12_inters_x, c12_inters_y)
    
    side0 = float(side0_arr[gap_index])
    side1 = float(side1_arr[gap_index])
    side2 = float(side2_arr[gap_index])
    tri_area = float(tri_area_arr[gap_index])
    tri_perim = float(tri_perim_arr[gap_index])
    
    c0_ori_x = float(ori0[0])
    c0_ori_y = float(ori0[1])
    c1_ori_x = float(ori1[0])
    c1_ori_y = float(ori1[1])
    c2_ori_x = float(ori2[0])
    c2_ori_y = float(ori2[1])
    
    
    ## we have enough information to uniquely describe the gap size and shape, so
    ## we will now calculate the 3 angles of the triangle, as well as the angle
    ## based on the curvatures of each side:
    
    # print('finding gap arclengths and angles for circle config ', k, '/',  total_possibilities-1)
    print('finding gap info for circle config ', k, '/',  total_possibilities-1)
    ## start by finding the 3 arclengths:
    ## the arclength is equal to the radius times the angle (i.e. r * theta), 
    ## the angle can be described as sin(theta) = opposite / hypoteneuse 
    ## so: sin(theta) = (1/2) sidelength / r 
    ## **you can look at Parent et al. 2017 Figure 1E for a diagram; the sidelength
    ## is the fancy 'l' letter (called 'ell'), and the angle theta here is called
    ## (30deg - theta) in the paper. Also, 'r' here is 'R' in the paper.
    ## putting the equations together and recognizing that theta describes only 
    ## half the arclength of the gap wall we then have:
    ## theta = np.arcsin( sidelength / (2 * r) )
    ## so: (1/2) * gap_sidewall_arclength = r * np.arcsin( sidelength / (2 * r) )
    ## moving that factor of (1/2) from the left side to the right side gives:
    arc0 = 2.0 * r0 * np.arcsin( side0 / (2.0 * r0) )
    arc1 = 2.0 * r1 * np.arcsin( side1 / (2.0 * r1) )
    arc2 = 2.0 * r2 * np.arcsin( side2 / (2.0 * r2) )
    
    ## also get the curved perimeter and area:
    curv_perim = arc0 + arc1 + arc2
    
    
    ## to get the area within the gap (i.e. the curved area), we find the area of
    ## the chord of each circle defined by the gap side length and the gap arc
    ## (e.g. side0 and arc0 for circle0), and add/subtract it to/from the area
    ## of the triangle defining the gap:
    ## the area of a circular sector is "Area = (1/2) *R * s"
    ## where R is the radius of the circular sector and s is the arclength defined 
    ## by the angle theta (see https://mathworld.wolfram.com/CircularSector.html)
    ## from the circular sector area we subtract the area of the triangle with 
    ## two sides of length R that have an angle 'theta' between them:
    ## Note: theta = arclength / radius, and the third side of the triangle is
    ## equal to the sidelength:
    c0_circ_sect_area = (1/2) * r0 * arc0
    ## using Heron's formula we can find the area of a triangle using just the 3
    ## sidelengths (see: https://mathworld.wolfram.com/HeronsFormula.html)
    ## Heron's formula: 
        ## Area = np.sqrt(s*(s - a)*(s - b)*(s - c))
        ## where a,b,c are the triangle sidelengths and 's' is the semiperimeter
        ## s = (1/2)*(a + b + c)
    c0_s = (1/2) * (r0 + r0 + side0)
    c0_tri_area = np.sqrt(c0_s * (c0_s - r0) * (c0_s - r0) * (c0_s - side0))
    ## lastly, the area of the chord:
    c0_curv_area = c0_circ_sect_area - c0_tri_area
    
    ## repeating for the other two circles:
    c1_circ_sect_area = (1/2) * r1 * arc1
    c1_s = (1/2) * (r1 + r1 + side1)
    c1_tri_area = np.sqrt(c1_s * (c1_s - r1) * (c1_s - r1) * (c1_s - side1))
    c1_curv_area = c1_circ_sect_area - c1_tri_area
    
    c2_circ_sect_area = (1/2) * r2 * arc2
    c2_s = (1/2) * (r2 + r2 + side2)
    c2_tri_area = np.sqrt(c2_s * (c2_s - r2) * (c2_s - r2) * (c2_s - side2))
    c2_curv_area = c2_circ_sect_area - c2_tri_area
    
    
    ## now we need to identify the signs of the curved areas (i.e. should it be
    ## added to or subtracted from the area tri_area?)
    ## in order to do so, we check if the shortest line from the centre of the 
    ## triangle formed by , c02_inters, and c12_inters to a side of the 
    ## triangle formed by the same 3 points is less than or greater than that 
    ## triangle center point to the arc of that same side:
    
    ## first, find the triangles center-point by taking the average of the three
    ## intersection points:
    tri_center = np.mean([np.asarray(c01_inters), 
                          np.asarray(c02_inters), 
                          np.asarray(c12_inters)], axis=0)
    ## next, define 3 vectors that connect each pair of intersection points
    ## (note that which of the 2 possible vector directions you get doesn't matter
    ## since they are parallel and therefore their dot product with the line made
    ## by the triangle center-point and the nearest point on the sidelength will
    ## still be 0):
    vec0 = np.asarray(c01_inters) - np.asarray(c02_inters)
    vec1 = np.asarray(c01_inters) - np.asarray(c12_inters)
    vec2 = np.asarray(c02_inters) - np.asarray(c12_inters)
    
    
    
    
    ## get the point on each side closest to the triangle center:
    # side0_closest_pt =  get_closest_point(c01_inters, vec0, tri_center)
    # side1_closest_pt =  get_closest_point(c12_inters, vec1, tri_center)
    # side2_closest_pt =  get_closest_point(c02_inters, vec2, tri_center)
    
    ## the closest point can lie outside of the gap side, therefore it might be
    ## better to take the midpoint of the line and draw a vector from the 
    ## triangle center to that midpoint. The vector won't be perpendicular 
    ## anymore, but it will still pass through both the curved and straight
    ## sides of the gap wall:
    side0_mid_pt = np.mean([c01_inters, c02_inters], axis=0)
    side1_mid_pt = np.mean([c01_inters, c12_inters], axis=0)
    side2_mid_pt = np.mean([c02_inters, c12_inters], axis=0)
    
    
    
    
    ## great, so we have the closest points on the lines, and now we need to draw
    ## a line from the triangle center through those points and find where that 
    ## line intersects with each circle. Then we compare the length of the line 
    ## from triangle center to a side vs the length from triangle center to an arc
    ## and ask which is smaller.
    
    ## first, get the vectors that start at the triangle center and end at the 
    ## closest points:
    # arc_vec0 = side0_closest_pt - tri_center
    # arc_vec1 = side1_closest_pt - tri_center
    # arc_vec2 = side2_closest_pt - tri_center
    
    ## using the mid points instead:
    arc_vec0 = side0_mid_pt - tri_center
    arc_vec1 = side1_mid_pt - tri_center
    arc_vec2 = side2_mid_pt - tri_center

    
    ## now find the intersections of the line with the circle:
    c0_arc_vec0_inters_pt1, c0_arc_vec0_inters_pt2 = get_circle_line_intersection_pts(ori0, r0, tri_center, arc_vec0)
    c1_arc_vec1_inters_pt1, c1_arc_vec1_inters_pt2 = get_circle_line_intersection_pts(ori1, r1, tri_center, arc_vec1)
    c2_arc_vec2_inters_pt1, c2_arc_vec2_inters_pt2 = get_circle_line_intersection_pts(ori2, r2, tri_center, arc_vec2)
    
    
    ## now we need to identify which of the two intersection points is the correct
    ## one to use... we can test to see which point intersects the line formed by
    ## the sidelength at the gap if connected to the circle origin. 
    # def get_line_line_intersection(line_pt1, line_vec1, line_pt2, line_vec2):
    
    
    ## well... for our purposes we can just test which one is closest to the 
    ## sidelength closest point (eg. side0_closest_pt)
    # c0_dist1 = np.linalg.norm(vector(c0_arc_vec0_inters_pt1, side0_closest_pt))
    # c0_dist2 = np.linalg.norm(vector(c0_arc_vec0_inters_pt2, side0_closest_pt))
    ## using mid points instead:
    c0_dist1 = np.linalg.norm(vector(c0_arc_vec0_inters_pt1, side0_mid_pt))
    c0_dist2 = np.linalg.norm(vector(c0_arc_vec0_inters_pt2, side0_mid_pt))

    if c0_dist1 <= c0_dist2:
        c0_arc_vec0_inters_pt = c0_arc_vec0_inters_pt1
    else:
        c0_arc_vec0_inters_pt = c0_arc_vec0_inters_pt2
    
    
    # c1_dist1 = np.linalg.norm(vector(c1_arc_vec1_inters_pt1, side1_closest_pt))
    # c1_dist2 = np.linalg.norm(vector(c1_arc_vec1_inters_pt2, side1_closest_pt))
    c1_dist1 = np.linalg.norm(vector(c1_arc_vec1_inters_pt1, side1_mid_pt))
    c1_dist2 = np.linalg.norm(vector(c1_arc_vec1_inters_pt2, side1_mid_pt))
    if c1_dist1 <= c1_dist2:
        c1_arc_vec1_inters_pt = c1_arc_vec1_inters_pt1
    else:
        c1_arc_vec1_inters_pt = c1_arc_vec1_inters_pt2
    
    
    # c2_dist1 = np.linalg.norm(vector(c2_arc_vec2_inters_pt1, side2_closest_pt))
    # c2_dist2 = np.linalg.norm(vector(c2_arc_vec2_inters_pt2, side2_closest_pt))
    c2_dist1 = np.linalg.norm(vector(c2_arc_vec2_inters_pt1, side2_mid_pt))
    c2_dist2 = np.linalg.norm(vector(c2_arc_vec2_inters_pt2, side2_mid_pt))
    if c2_dist1 <= c2_dist2:
        c2_arc_vec2_inters_pt = c2_arc_vec2_inters_pt1
    else:
        c2_arc_vec2_inters_pt = c2_arc_vec2_inters_pt2
    
    
    
    ## finally, we can use these points to determine which line segments are 
    ## smallest between the one formed by connecting tri_center to side#_closest_pt
    ## or connecting tri_center to c#_arc_vec#_inters_pt, and therefore determine
    ## whether the arc points inwards or outwards, therefore telling us if we need
    ## to add or subtract the area of the chord:
    # side0_magn = np.linalg.norm(vector(side0_closest_pt, tri_center))
    ## using side mid point instead
    side0_magn = np.linalg.norm(vector(side0_mid_pt, tri_center))    
    arc_vec0_magn = np.linalg.norm(vector(c0_arc_vec0_inters_pt, tri_center))
    ## if the length of side0_magn is longer than arc_vec0_magn, then the curve
    ## must be cutting in to the triangle, and therefore we need to subtract the
    ## area of the chord, so the sign is negative, otherwise we add area:
    c0_curv_sign = np.sign(arc_vec0_magn - side0_magn)
    
    ## now for the other two sides:
    # side1_magn = np.linalg.norm(vector(side1_closest_pt, tri_center))
    side1_magn = np.linalg.norm(vector(side1_mid_pt, tri_center))
    arc_vec1_magn = np.linalg.norm(vector(c1_arc_vec1_inters_pt, tri_center))
    c1_curv_sign = np.sign(arc_vec1_magn - side1_magn)
    
    # side2_magn = np.linalg.norm(vector(side2_closest_pt, tri_center))
    side2_magn = np.linalg.norm(vector(side2_mid_pt, tri_center))
    arc_vec2_magn = np.linalg.norm(vector(c2_arc_vec2_inters_pt, tri_center))
    c2_curv_sign = np.sign(arc_vec2_magn - side2_magn)
    
    
    
    ## Finally, we can calculate the area of the gap when curved surfaces are
    ## accounted for: 
    curv_area = tri_area + c0_curv_sign * c0_curv_area + c1_curv_sign * c1_curv_area + c2_curv_sign * c2_curv_area
    
    
    
    
    ## the 'vector' function calculates a vector that runs parallel to side0, 
    ## side1, or side2:
    ## These two vectors form one angle:
    vec01to02 = vector(c01_inters, c02_inters)
    vec01to12 = vector(c01_inters, c12_inters)
    ## ...and so do these two:
    vec02to01 = vector(c02_inters, c01_inters)
    vec02to12 = vector(c02_inters, c12_inters)
    ## ...and these two as well:
    vec12to01 = vector(c12_inters, c01_inters)
    vec12to02 = vector(c12_inters, c02_inters)
    
    ## Now get the angle associated with each corner of the triangle that defines
    ## the gap:
    angle01 = np.rad2deg(angle(vec01to02, vec01to12))
    angle02 = np.rad2deg(angle(vec02to01, vec02to12))
    angle12 = np.rad2deg(angle(vec12to01, vec12to02))
    
    ## Check that the 3 angles add up to 180deg:
    assert np.round(angle01 + angle02 + angle12, decimals=2) == 180
    
    
    
    
    ## Now, get the angles associated with each corner of the gap when the curved
    ## surface is accounted for, i.e. get the angle at the circle-circle 
    ## intersection points as defined by the vectors that are tangent to the 
    ## circles at those points:
    
    ## first, find the slopes that are tangent to those points:
    ## for the intersection point between circle0 and circle1:
    c0_slope01 = circle_deriv(c01_inters, ori0)
    c1_slope01 = circle_deriv(c01_inters, ori1)
    ## for the intersection point between circle0 and circle2:
    c0_slope02 = circle_deriv(c02_inters, ori0)
    c2_slope02 = circle_deriv(c02_inters, ori2)
    ## for the intersection point between circle1 and circle2:
    c1_slope12 = circle_deriv(c12_inters, ori1)
    c2_slope12 = circle_deriv(c12_inters, ori2)
    
    
    ##Let's put these intersect-slope pairs along with the other two intersection 
    ## points in a list for later:
    derivs_at_inters = [(c01_inters, c0_slope01, c02_inters, c12_inters), 
                        (c01_inters, c1_slope01, c02_inters, c12_inters), 
                        (c02_inters, c0_slope02, c01_inters, c12_inters), 
                        (c02_inters, c2_slope02, c01_inters, c12_inters), 
                        (c12_inters, c1_slope12, c01_inters, c02_inters), 
                        (c12_inters, c2_slope12, c01_inters, c02_inters)
                        ]
    
    
    
    ## now use these slopes to construct a unit vector in each direction:
    unit_vecs_at_inters = list()
    for point, slope, other_pt1, other_pt2 in derivs_at_inters:
        ## construct the unit vectors:
        u_vec1, u_vec2 = slope_to_unit_vecs(point, slope)
        
        ## pick the unit vector that has the smallest distance to one of the other
        ## two points, as the unit vectors form a triangle that may or may not
        ## be squished or bloated:
        pt1 = point + u_vec1
        pt2 = point + u_vec2
        
        ## get magnitude of vectors formed by connecting either pt1 or pt2 with 
        ## each 'other_pt'
        test_vecs = np.array((pt1 - other_pt1, pt1 - other_pt2, 
                              pt2 - other_pt1, pt2 - other_pt2))
        ## because the first two indices of text_vecs use pt1 and the last two use
        ## pt2, that means the first two correspond to u_vec1 and the last two
        ## correspond to u_vec2
        u_vecs = [u_vec1, u_vec1, u_vec2, u_vec2]
    
        ## get the magnitude of each test_vector:
        dists = [np.linalg.norm(vec) for vec in test_vecs]
        
        ## pick the vector with the smallest magnitude
        u_vec = u_vecs[np.argmin(dists)]
        
        unit_vecs_at_inters.append((point, u_vec))
        
        
    
    ## keep in mind that because derivs_at_inters has the order of gap corners as:
    ## c01_inters, c01_inters, c02_inters, c02_inters, c12_inters, c12_inters
    ## so does unit_vecs_at_inters.
    ## We will be unpacking these vector-intersection_point pairs and then 
    ## calculating the angle between the two vectors at each intersection point
    ## next.
    
    ## unpacking vector-intersection_point pairs:
    _, (c01_vec0, c01_vec1, c02_vec0, c02_vec2, c12_vec1, c12_vec2) = list(zip(*unit_vecs_at_inters))
    
    ## Now get the angle associated with each corner of the gap when curvature is
    ## accounted for:
    curv_angle01 = np.rad2deg(angle(c01_vec0, c01_vec1))
    curv_angle02 = np.rad2deg(angle(c02_vec0, c02_vec2))
    curv_angle12 = np.rad2deg(angle(c12_vec1, c12_vec2))
    
    
        
    
    
    
    
    
    
    # ## 
    # ## The plots produced here are for quality-checking purposes only, thus they 
    # ## are commented out:
    # ##
    # ## I think we have reached a point where I need to plot the circles and vectors
    # ## to see if they are correct:  
    # def get_circle_points_xy(origin_xy, radius, num_points=3601):
    
    #     ## amount of circle to draw, 2*pi = the whole circle:
    #     theta = np.linspace(0, 2*np.pi, num_points)
        
    #     ## unpack circle origin:
    #     x0, y0 = origin_xy
        
    #     ## a circle defined by polar coordinates:
    #     x = radius * np.cos(theta) + x0
    #     y = radius * np.sin(theta) + y0
        
    #     return (x,y)
     
    # ## get circle x,y points:
    # c0_x, c0_y = get_circle_points_xy(ori0, r0)
    # c1_x, c1_y = get_circle_points_xy(ori1, r1)
    # c2_x, c2_y = get_circle_points_xy(ori2, r2)
    
    
    # plt.subplots(figsize=(6,6))
    # ## draw circles:
    # plt.plot(c0_x, c0_y, c='k')
    # plt.plot(c1_x, c1_y, c='k')
    # plt.plot(c2_x, c2_y, c='k')
    # ## plot circle origins:
    # plt.scatter(*ori0, c='tab:blue', marker='.')
    # plt.scatter(*ori1, c='tab:blue', marker='.')
    # plt.scatter(*ori2, c='tab:blue', marker='.')
    # ## plot all intersection points:
    # plt.scatter(c01_inters_pts_x, c01_inters_pts_y, c='tab:orange', marker='.', zorder=9)
    # plt.scatter(c02_inters_pts_x, c02_inters_pts_y, c='tab:orange', marker='.', zorder=9)
    # plt.scatter(c12_inters_pts_x, c12_inters_pts_y, c='tab:orange', marker='.', zorder=9)
    # ## plot gap corners
    # plt.scatter(*c01_inters, c='red', marker='.', zorder=10)
    # plt.scatter(*c02_inters, c='red', marker='.', zorder=10)
    # plt.scatter(*c12_inters, c='red', marker='.', zorder=10)
    # ## draw gap triangle center:
    # plt.scatter(*tri_center, c='tab:green', marker='.')
    # ## plot line0 and closest point on line0 to tri_center
    # plt.plot(*list(zip(*[c01_inters, c02_inters])), c='r', lw=0.5)
    # plt.plot(*list(zip(*[c01_inters, c12_inters])), c='r', lw=0.5)
    # plt.plot(*list(zip(*[c02_inters, c12_inters])), c='r', lw=0.5)
    # # plt.scatter(*side0_closest_pt, c='c', marker='.', zorder=10)
    # # plt.scatter(*side1_closest_pt, c='c', marker='.', zorder=10)
    # # plt.scatter(*side2_closest_pt, c='c', marker='.', zorder=10)
    # plt.scatter(*side0_mid_pt, c='c', marker='.', zorder=10)
    # plt.scatter(*side1_mid_pt, c='c', marker='.', zorder=10)
    # plt.scatter(*side2_mid_pt, c='c', marker='.', zorder=10)
    # ## plot the points on the arcs that intersect with the line that passe through
    # ## the triangle center and the cloests point on the sidelengths to the triangle
    # ## center:
    # plt.scatter(*c0_arc_vec0_inters_pt, c='m', marker='.')
    # plt.scatter(*c1_arc_vec1_inters_pt, c='m', marker='.')
    # plt.scatter(*c2_arc_vec2_inters_pt, c='m', marker='.')
    
    # plt.xlim(0.66, 0.74)
    # plt.ylim(0.66, 0.74)
    
    # print(np.mean([side0, side1, side2]), np.sum([curv_angle01, curv_angle02, curv_angle12]))
    
    
    
    
    
    gap.append((r0, r1, r2, 
                c01_inters_x, c01_inters_y, 
                c02_inters_x, c02_inters_y, 
                c12_inters_x, c12_inters_y, 
                side0, side1, side2, tri_area, tri_perim, 
                *ori0, *ori1, *ori2, 
                angle01, angle02, angle12, 
                arc0, arc1, arc2, curv_area, curv_perim, 
                curv_angle01, curv_angle02, curv_angle12))


gap_arr = np.asarray(gap)


np.save(Path.joinpath(out_dir, 'gap_data.npy'), gap_arr)














    
    
############################################################################
################                                            ################
################          Unused code starts here.          ################
################                                            ################
############################################################################
    
## I sped up the code sufficiently that multiprocessing was not necessary


## Define the number of processes to use: 
## processes = how many processors you want to use (if you make if 'None' then
## all the computers processors will be used)
# processes = multiproc.cpu_count() - 1



# if __name__ == '__main__':
    
#     ## spawn a pool of processes and distribute the function circle_intersects  
#     ## to those processes:
#     with multiproc.Pool(processes=processes) as pool:
#     # pool = multiproc.Pool(processes = processes)#, initargs=(filepath, total_possibilities, data_length, global_index_start, global_index_end, processes, circle0_yxr, circle1_yxr, circle2_yxr))
#         pool.map(circle_intersects, index_start_list)
#     # pool.close()
#     # pool.join()
    




# ## get the vectors perpendicular to the gap sidelength ones:
# def perpendic_xy(vec_xy):
#     ## takes a 2D vector and return an orthogonal vector of equal magnitude
#     ## (see https://mathworld.wolfram.com/PerpendicularVector.html)
#     x, y = vec_xy
#     perp = (-1 * y, x)
#     return perp
    
# perp_vec0 = np.asarray(perpendic_xy(vec0))
# perp_vec1 = np.asarray(perpendic_xy(vec1))
# perp_vec2 = np.asarray(perpendic_xy(vec2))

    
    
    
    