import numpy as np
import tifffile as tf
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def fft_propagation(fx, fy, z, wavelength):
    h = np.exp(
        1j * (2 * np.pi / wavelength) * z * np.sqrt(1 - (wavelength ** 2 * fx ** 2) - (wavelength ** 2 * fy ** 2)))
    h[np.isnan(h)] = 0  # replace nan's with zeros
    return h
def lens_transformation(x, y, f, wl):
    g = np.exp(-1j * np.pi * (x ** 2 + y ** 2) / (wl * f))
    return g
def cylindrical_lens(x, y, f, wl, direction):
    if direction == 0:
        g = np.exp(-1j * np.pi * (x ** 2) / (wl * f))
    elif direction == 1:
        g = np.exp(-1j * np.pi * (y ** 2) / (wl * f))    
    return g
def gaussian(sigma, x, y):
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return g
wl = 0.530  # um
nx = 1280
ny = 1280
dx = 1  # micron
xr = np.arange(-nx, nx)
yr = np.arange(-ny, ny)
xv, yv = np.meshgrid(xr, yr, indexing='ij', sparse=True)
xc = dx * xv
yc = dx * yv
fm = 1 / (2 * dx)  # Spatial frequency, inverse microns
df = fm / nx  # Spacing between discrete frequency coordinates, inverse microns
fx = df * xv
fy = df * yv
k = 2 * np.pi / wl
radius = nx * 0.60


msk = np.sqrt(xv ** 2 + yv ** 2) <= radius
result = msk * gaussian(nx * 2 * dx, xc, yc)
tf.imshow(np.abs(result) ** 2)
# z = 100000
# h = fft_propagation(fx, fy, z, wl)
# objf = fftshift(fft2(result))
# temp = objf * h
# result = ifft2(ifftshift(temp))
# tf.imshow(np.abs(result) ** 2)
# tf.imsave('propogation.tif', np.abs(result) ** 2)
# f = 50000
# phase = lens_transformation(xc, yc, f, wl) * msk
# object = object * phase
f = 100000
phase = cylindrical_lens(xc, yc, f, wl, 1) * msk
result = result * phase #* np.exp(1j * 0.2*zer.Zm(7, radius, Nx=nx*2))
z = 100000
h = fft_propagation(fx, fy, z, wl)
objf = fftshift(fft2(result))
temp = objf * h
result = ifft2(ifftshift(temp))
tf.imshow(np.abs(result) ** 2)
# tf.imsave('propogation.tif', np.abs(result) ** 2)
f = 100000
phase = lens_transformation(xc, yc, f, wl) * msk
result = result * phase
z = 200000
h = fft_propagation(fx, fy, z, wl)
objf = fftshift(fft2(result))
temp = objf * h
result = ifft2(ifftshift(temp))
tf.imshow(np.abs(result) ** 2)
f = 100000
phase = lens_transformation(xc, yc, f, wl) * msk
result = result * phase
z = 300000
h = fft_propagation(fx, fy, z, wl)
objf = fftshift(fft2(result))
temp = objf * h
result = ifft2(ifftshift(temp))
tf.imshow(np.abs(result) ** 2)
f = 200000
phase = lens_transformation(xc, yc, f, wl)
result = result * phase
z = 200000
h = fft_propagation(fx, fy, z, wl)
objf = fftshift(fft2(result))
temp = objf * h
result = ifft2(ifftshift(temp))
tf.imshow(np.abs(result) ** 2)
tf.imsave('result2.tif', np.abs(result) ** 2)