import math
import numpy as np
from scipy.io import loadmat
from utils.utils import cart2sph, pol2cart
from sklearn.preprocessing import scale
from scipy.interpolate import griddata

def to_alpha0(data, args):
    alpha_data = []
    for window in data:
        window_data0 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data0 = np.abs(window_data0)
        window_data0 = np.sum(np.power(window_data0[args.point0_low:args.point0_high, :], 2), axis=0)
        window_data0 = np.log2(window_data0 / args.window_length)
        alpha_data.append(window_data0)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha1(data, args):
    alpha_data = []
    for window in data:
        window_data1 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data1 = np.abs(window_data1)
        window_data1 = np.sum(np.power(window_data1[args.point1_low:args.point1_high, :], 2), axis=0)
        window_data1 = np.log2(window_data1 / args.window_length)
        alpha_data.append(window_data1)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha2(data, args):
    alpha_data = []
    for window in data:
        window_data2 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data2 = np.abs(window_data2)
        window_data2 = np.sum(np.power(window_data2[args.point2_low:args.point2_high, :], 2), axis=0)
        window_data2 = np.log2(window_data2 / args.window_length)
        alpha_data.append(window_data2)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha3(data, args):
    alpha_data = []
    for window in data:
        window_data3 = np.fft.fft(window, n=args.window_length, axis=0)
        window_data3 = np.abs(window_data3)
        window_data3 = np.sum(np.power(window_data3[args.point3_low:args.point3_high, :], 2), axis=0)
        window_data3 = np.log2(window_data3 / args.window_length)
        alpha_data.append(window_data3)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def to_alpha4(data, args):
    alpha_data = []
    for window in data:
        window_data4= np.fft.fft(window, n=args.window_length, axis=0)
        window_data4 = np.abs(window_data4)
        window_data4 = np.sum(np.power(window_data4[args.point4_low:args.point4_high, :], 2), axis=0)
        window_data4 = np.log2(window_data4 / args.window_length)
        alpha_data.append(window_data4)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data

def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, math.pi / 2 - elev)

def gen_images(data, args):
    locs = loadmat('./locs_orig.mat')
    locs_3d = locs['data']
    locs_2d = []
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    locs_2d_final = np.array(locs_2d)
    grid_x, grid_y = np.mgrid[
                     min(np.array(locs_2d)[:, 0]):max(np.array(locs_2d)[:, 0]):args.image_size * 1j,
                     min(np.array(locs_2d)[:, 1]):max(np.array(locs_2d)[:, 1]):args.image_size * 1j]

    images = []
    for i in range(data.shape[0]):
        images.append(griddata(locs_2d_final, data[i, :], (grid_x, grid_y), method='cubic', fill_value=np.nan))
    images = np.stack(images, axis=0)

    images[~np.isnan(images)] = scale(images[~np.isnan(images)])
    images = np.nan_to_num(images)
    return images


def filter_signal_by_fft(data, point0_low, point0_high, window_length): 
    filtered_data = []
    for window in data:
        fft_data = np.fft.fft(window, n=window_length, axis=1)
        mask = np.zeros_like(fft_data)
        mask_slice = [slice(None)] * fft_data.ndim
        mask_slice[1] = slice(point0_low, point0_high)
        mask[tuple(mask_slice)] = 1.0

        n_fft = window_length
        if np.isrealobj(window):
            neg_low = n_fft - (point0_high - point0_low)
            neg_high = n_fft - point0_low
            mask_slice[1] = slice(neg_low, neg_high)
            mask[tuple(mask_slice)] = 1.0
        
        filtered_fft = fft_data * mask
        
        ifft_data = np.fft.ifft(filtered_fft, n=window_length, axis=1)
        
        if np.isrealobj(window):
            ifft_data = np.real(ifft_data)
        
        orig_length = window.shape[1]
        if window_length > orig_length:
            crop_slice = [slice(None)] * ifft_data.ndim
            crop_slice[1] = slice(0, orig_length)
            ifft_data = ifft_data[tuple(crop_slice)]
        
        filtered_data.append(ifft_data)
    
    filtered_data = np.stack(filtered_data, axis=0)
    
    return filtered_data