from heapq import heapify, heappop

import numpy as np
from scipy.signal import convolve

# helper functions for centerline processing

def evenly_spaced_resample(xs, ys, step):
    x_0 = xs[0]
    xs = xs - x_0
    resampled_xs = [0]
    resampled_ys = [ys[0]]
    i = 1
    while True:
        x = len(resampled_xs) * step
        while x > xs[i]:
            i += 1
            if i == len(xs):
                resampled_xs = x_0 + np.array(resampled_xs, dtype=np.float64)
                resampled_ys = np.array(resampled_ys, dtype=np.float64)
                return resampled_xs, resampled_ys
        resampled_xs.append(x)
        a = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
        resampled_ys.append(a * ys[i] + (1 - a) * ys[i - 1])


def smoothing(xs, ys, kernel_len):
    kernel = [1 / kernel_len] * kernel_len
    xs = convolve(xs, kernel, mode="valid")
    ys = convolve(ys, kernel, mode="valid")
    return xs, ys


def double_intersection(xs, ys):
    if ys[-1] < ys[0]:
        xs, ys = double_intersection(xs[::-1], ys[::-1])
        return xs[::-1], ys[::-1]
    heap = list(ys)
    heapify(heap)
    i = 0
    try:
        while heappop(heap) == ys[i]:
            i += 1
    except IndexError:
        return xs, ys
    if i:
        i -= 1
    return xs[i:], ys[i:]


def derivative_cut(xs, ys, std_cut, window):
    derivative = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    mean = np.mean(derivative)
    std = np.std(derivative)
    is_steep = abs(derivative - mean) >= std_cut * std
    start = is_steep[:window].argmax()
    while is_steep[start]:
        start += 1
        if start == len(ys):
            return xs, ys
    end = len(xs) - 2 - is_steep[: -window - 1 : -1].argmax()
    while is_steep[end]:
        end -= 1
        if end == -1:
            return xs, ys
    return xs[start : end + 2], ys[start : end + 2]


def preprocess_centerline(xs, ys, kernel_len, std_cut, window, resolution):
    xs, ys = evenly_spaced_resample(xs, ys, resolution)
    xs_smooth, ys_smooth = smoothing(xs, ys, kernel_len)
    return xs_smooth, ys_smooth



def keep_centerline(
    xs,
    ys,
    pixel_size,
    min_len,
    kernel_len,
    std_cut,
    window,
    min_prep_len,
    max_der_std,
    max_der,
    max_var_der,
):
    if xs[-1] - xs[0] < min_len:
        return np.bool_(False)
    if np.ptp(ys) <= 1.0e-8:
        return np.bool_(False)
    
    n_corrupted = np.count_nonzero(np.logical_or(ys == ys.min(), ys == ys.max())) - 2
    if 10 * n_corrupted > len(ys):
        return np.bool_(False)
    
    xs_p, _ = preprocess_centerline(xs, ys, kernel_len, std_cut, window, pixel_size)
    if xs_p[-1] - xs_p[0] < min_prep_len:
        return np.bool_(False)
    start = np.searchsorted(xs, xs_p[0], "right") - 1
    end = np.searchsorted(xs, xs_p[-1], "left") + 1
    dx = xs[start + 1 : end] - xs[start : end - 1]
    dy = ys[start + 1 : end] - ys[start : end - 1]
    der = dy / dx
    mean = np.mean(der)
    std = np.std(der)
    abnormal_variation = abs(der - mean) >= min(max_der_std * std, max_der)
    if 10 * np.count_nonzero(abnormal_variation) > len(ys):
        return np.bool_(False)
    if np.max(np.abs(der[1:]-der[:-1]))>max_var_der:
        return np.bool_(False)
    return np.bool_(True)
