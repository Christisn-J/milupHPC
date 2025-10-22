#!/usr/bin/env python3
'created by Anne Vera Jeschke 15 January 2024'
import numpy as np
import matplotlib.pyplot as plt
import h5py

"""
this program creates two 3-D cylinders around the origin which are then shifted to their final positions.
used for colliding cylinder testcase
"""
# Dim of cylinders
dim = 3

# cylinder properties: inner and outer radius
r_inner = 3.0
r_outer = 4.0

width = 10  # width of cylinders will be width*delta_p

# particle spacing
delta_p = 0.1
# 0.1   --> 2 * 2.196 * width   = 4.392 *width  particles
# 0.07  --> 2 * 4.488   = 8.976   particles
# 0.05  --> 2 * 8.804   = 17.608  particles
# 0.04  --> 2 * 13.736  = 27.472  particles
# 0.03  --> 2 * 24.420  = 48.840  particles
# 0.02  --> 2 * 54.988  = 109.976 particles
# 0.015 --> 2 *
# 0.01  --> 2 * 219.860 = 439.720 particles
# 0.008 --> 2 * 343.668 = 687.336 particles
# 0.007 --> 2 * 448.772 = 897.544 particles
# 0.006 --> 2 * 610.948 = 1.221.896 particles
# 0.005 --> 2 * 879.624 = 1.759.248 particles
# 0.001 --> 2 * 21.991.108 particles

# shift of the cylinders from origin on x-axis
# shift = 6    # for delta_p <= 0.1
shift = 5
# shift = 4.75  # for delta_p <= 0.05
# shift = 4.85 # for delta_p <= 0.01, 0.005, 0.001
# shift = 4.2

# projected speed
v_p = 0.015 # 0.059

density = 1

mass = delta_p**dim * density

# create initial distribution through creating a 2d grid with dims 2*r_outer x 2*r_outer
# then delete particles which are not on the ring

# calc number of particles within square
N_length = int(2*r_outer/delta_p)
# number of particles in square
N_square = int(N_length**2)

# coordinates of particles in square
r = np.zeros((N_square, dim))

# 2D meshgrid
a = np.mgrid[0:N_length, 0:N_length]

# create square
for i in range(dim-1):
    k = 0
    for j in range(N_square):
        if j % N_length == 0 and j > 0:
            k += 1
            k = k % N_length
        # print(i, k, j)

        r[j, i] = (a[i, k, j % N_length]-(N_length-1)/2)*delta_p

# count particles in one ring
N = 0
arr = np.zeros(N_square) 
for i in range(N_square):
    radius = np.sqrt(r[i, 0]**2 + r[i, 1]**2)
    if r_outer >= radius >= r_inner:
        N += 1
        arr[i] = 1

# construct two rings with N particles which then are shifted along the x-axis

r_ring = np.zeros((N, dim))  # first ring
r_ring2 = np.zeros((N, dim))  # second ring
v = np.zeros((N, dim))
v2 = np.zeros((N, dim))
NoRings = width
m = np.ones(2*N*NoRings)*mass
rho = np.ones(2*N*NoRings)*density
materialId = np.zeros(2*N*NoRings, dtype=np.int8)
#Sxx = np.zeros(2*N)
#Sxy = np.zeros(2*N)

# create ring 1
counter = 0
for i in range(N_square):
    if arr[i] == 1:
        r_ring[counter, 0] = r[i, 0] - shift # normal
        r_ring[counter, 1] = r[i, 1]  # normal
        r_ring[counter, 2] = 0


        # r_ring[counter, 0] = r[i, 0] + shift # second quandrant
        # r_ring[counter, 0] = r[i, 0] - 2.5*shift # first quadrant
        # r_ring[counter, 1] = r[i, 1] + shift # positive

        v[counter, 0] = v_p
        counter += 1
# create ring 2
counter = 0
for i in range(N_square):
    if arr[i] == 1:
        r_ring2[counter, 0] = r[i, 0] + shift  # normal
        r_ring2[counter, 1] = r[i, 1]  # normal
        r_ring2[counter, 2] = 0

        # r_ring2[counter, 0] = r[i, 0] + 2.5*shift # second quadrant
        # r_ring2[counter, 0] = r[i, 0] - shift #first quadrant
        # r_ring2[counter, 1] = r[i, 1] + shift #positive

        v2[counter, 0] = -v_p
        counter += 1
# put two rings in one array
r_rings = np.concatenate((r_ring, r_ring2))
v_rings = np.concatenate((v, v2))

# to create 3-D cylinders the already produces particles distribution
# of the two rings is copied and shift along the z-axis
for i in range(1, NoRings):
    r_ring[:, 2] = i*delta_p
    r_ring2[:, 2] = i*delta_p
    r_rings = np.concatenate((r_rings, r_ring, r_ring2))
    v_rings = np.concatenate((v_rings, v, v2))


#h5f = h5py.File("rings_N{}-2D.h5".format(2*N), "w")
#print("Saving to rings_N{}-2D.h5...".format(2*N))
h5f = h5py.File("cyl_deltap{}width{}.h5".format(delta_p, NoRings), "w")
print("Saving to cyl_deltap{}width{}.h5...".format(delta_p, NoRings))

# write to hdf5 data set
h5f.create_dataset("x", data=r_rings)
h5f.create_dataset("v", data=v_rings)
h5f.create_dataset("m", data=m)
h5f.create_dataset("materialId", data=materialId)
h5f.create_dataset("rho", data=rho)
#h5f.create_dataset("Sxx", data=Sxx)
#h5f.create_dataset("Sxy", data=Sxy)

h5f.close()
print("Number of particles: ", 2*N*NoRings)
print("Finished")