#! /usr/bin/env python

from re import U
import numpy as np
import argparse
import time
from multiprocessing import Pool, TimeoutError
from julia_curve import c_from_group
import multiprocess as mp

# Update according to your group size and number (see TUWEL)
GROUP_SIZE   = 2
GROUP_NUMBER = 14

# do not modify BENCHMARK_C
BENCHMARK_C = complex(-0.2, -0.65)

def compute_julia_set_sequential(xmin, xmax, ymin, ymax, im_width, im_height, c):

    zabs_max = 10
    nit_max = 300

    xwidth  = xmax - xmin
    yheight = ymax - ymin

    julia = np.zeros((im_width, im_height))
    for ix in range(im_width):
        for iy in range(im_height):
            nit = 0
            # Map pixel position to a point in the complex plane
            z = complex(ix / im_width * xwidth + xmin,
                        iy / im_height * yheight + ymin)
            # Do the iterations
            while abs(z) <= zabs_max and nit < nit_max:
                z = z**2 + c
                nit += 1
            ratio = nit / nit_max
            julia[ix,iy] = ratio

    return julia

# this function is for computing a specific patch
# function is almost identical to compute_julia_set_sequential() but it accounts for the fact
# that it is computing a patch which has starting coordinates which arent necessarily 0, 0
def compute_julia_patch(xmin, xmax, ymin, ymax, im_width, im_height, c, 
                        x_start, y_start, patch_width, patch_height):
    zabs_max = 10
    nit_max = 300

    xwidth  = xmax - xmin
    yheight = ymax - ymin

    julia = np.zeros((patch_width, patch_height))
    for ix in range(patch_width):
        for iy in range(patch_height):
            nit = 0
            # Map pixel position to a point in the complex plane
            z = complex((ix + x_start) / im_width * xwidth + xmin,    # only difference between compute_julia_set_sequential() is this row
                        (iy + y_start) / im_height * yheight + ymin)   # ... and this row
            # Do the iterations
            while abs(z) <= zabs_max and nit < nit_max:
                z = z**2 + c
                nit += 1
            ratio = nit / nit_max
            julia[ix,iy] = ratio

    return julia

# function for adding elements from the result_queue to the global (shared) variable (julia_set)
def add_patch_to_whole_set(julia_set, patch, x_start, y_start):
        julia_set[x_start:x_start+patch.shape[0], y_start:y_start+patch.shape[1]] = patch


def compute_julia_in_parallel(size, xmin, xmax, ymin, ymax, patch, nprocs, c):

    # splitting the image into patches
    # initializing some key variables
    full_patches_per_row = size // patch
    partial_patch_size = size % patch
    partial_patches = partial_patch_size != 0
    patch_start_positions = []
    patch_dimensions = []
    inputs = []
    julia_img = np.zeros((size, size))


    # finding out if the image will have partial patches
    if partial_patches:
        rng = full_patches_per_row+1
    else:
        rng = full_patches_per_row


    # generating start coordinates of each patch
    # alongside
    # generating the dimension of each patch
    for x in range(rng):
        for y in range(rng):

            ######## PATCH START POSITIONS ############
            patch_start_positions.append((x*patch, y*patch))

            ########## PATCH DIMENSIONS #############
            # case when patches fit perfectly in the image
            if not partial_patches:
                patch_dimensions.append((patch, patch))
            # case when we do have partial patches
            else:
                # case when we are at the vertical border
                if x == rng-1:
                    partial_width = partial_patch_size
                #case when we are not at the vertcal border
                else:
                    partial_width = patch
                # case when we are at horizontal border
                if y == rng-1:
                    partial_height = partial_patch_size
                # case when we are not at the horizontal border
                else:
                    partial_height = patch
                patch_dimensions.append((partial_width, partial_height))


    ########### CALCULATING INPUTS ################
    #task arguments are:
        #xmin, xmax, ymin, ymax, im_width, im_height, c, \
                        #x_start, y_start, patch_width, patch_height

        # xmin, xmax, ymin, ymax, im_width, im_height, c are CONSTANT
        # x_start, y_start, is different for each patch
        # patch_width, patch_height is maybe only different for the last patch if it is not standard dimension
    for i in range(len(patch_start_positions)):
            x_start = patch_start_positions[i][0]
            y_start = patch_start_positions[i][1]
            patch_width = patch_dimensions[i][0]
            patch_height = patch_dimensions[i][1]
            inputs.append((xmin, xmax, ymin, ymax, size, size, c,
                           x_start, y_start,
                           patch_width, patch_height))

    pool = mp.Pool(processes = nprocs)
    completed_patches = pool.starmap(compute_julia_patch, inputs, chunksize = 1)


    ########## PUTTING PATCHES TOGETHER ##############
    for i, p in enumerate(completed_patches):
        p_start_position = patch_start_positions[i]
        x_strt = p_start_position[0]
        y_strt = p_start_position[1]
        add_patch_to_whole_set(julia_img, p, x_strt, y_strt)


    return julia_img


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", help="image size in pixels (square images)", type=int, default=500)
    parser.add_argument("--xmin", help="", type=float, default=-1.5)
    parser.add_argument("--xmax", help="", type=float, default=1.5)
    parser.add_argument("--ymin", help="", type=float, default=-1.5)
    parser.add_argument("--ymax", help="", type=float, default=1.5)
    parser.add_argument("--group-size", help="", type=int, default=None)
    parser.add_argument("--group-number", help="", type=int, default=None)
    parser.add_argument("--patch", help="patch size in pixels (square images)", type=int, default=20)
    parser.add_argument("--nprocs", help="number of workers", type=int, default=1)
    parser.add_argument("--draw-axes", help="Whether to draw axes", action="store_true")
    parser.add_argument("-o", help="output file")
    parser.add_argument("--benchmark", help="Whether to execute the script with the benchmark Julia set", action="store_true")
    args = parser.parse_args()

    #print(args)
    if args.group_size is not None:
        GROUP_SIZE = args.group_size
    if args.group_number is not None:
        GROUP_NUMBER = args.group_number

    # assign c based on mode
    c = None
    if args.benchmark:
        c = BENCHMARK_C 
    else:
        c = c_from_group(GROUP_SIZE, GROUP_NUMBER) 

    stime = time.perf_counter()
    julia_img = compute_julia_in_parallel(
        args.size,
        args.xmin, args.xmax, 
        args.ymin, args.ymax, 
        args.patch,
        args.nprocs,
        c)
    rtime = time.perf_counter() - stime

    print(f"{args.size};{args.patch};{args.nprocs};{rtime}")

    if not args.o is None:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig, ax = plt.subplots()
        ax.imshow(julia_img, interpolation='nearest', cmap=plt.get_cmap("hot"))

        if args.draw_axes:
            # set labels correctly
            im_width = args.size
            im_height = args.size
            xmin = args.xmin
            xmax = args.xmax
            xwidth = args.xmax - args.xmin
            ymin = args.ymin
            ymax = args.ymax
            yheight = args.ymax - args.ymin

            xtick_labels = np.linspace(xmin, xmax, 7)
            ax.set_xticks([(x-xmin) / xwidth * im_width for x in xtick_labels])
            ax.set_xticklabels(['{:.1f}'.format(xtick) for xtick in xtick_labels])
            ytick_labels = np.linspace(ymin, ymax, 7)
            ax.set_yticks([(y-ymin) / yheight * im_height for y in ytick_labels])
            ax.set_yticklabels(['{:.1f}'.format(-ytick) for ytick in ytick_labels])
            ax.set_xlabel("Imag")
            ax.set_ylabel("Real")
        else:
            # disable axes
            ax.axis("off") 

        plt.tight_layout()
        plt.savefig(args.o, bbox_inches='tight')
        #plt.show()