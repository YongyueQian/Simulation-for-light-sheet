import matplotlib.pylab as plt
import tifffile as tf
import numpy as np
from scipy.optimize import curve_fit

na = 0.5 #8.225mm
nh = 512
n = 2 * nh
wl = 0.53
dxy = 0.1
df = 1/(nh*dxy)
radius = (2*na/wl)/df
xyr = np.roll(np.arange(-nh, nh), nh)
xyv = np.meshgrid(xyr, xyr, indexing='ij', sparse=True)
rho = np.sqrt(xyv[0]**2 + xyv[1]**2)
bpp = (rho<=radius).astype(np.float32)
tf.imshow(bpp)
f = np.where( (np.abs(xyv[0]) <= 8) & (np.abs(xyv[1]) <= 2/11*radius), 1, 0)
msk = (f * bpp)==1
# phm = np.exp(1j  * msk)
new_bpp = f * bpp 
tf.imshow(f)
tf.imshow(msk)
tf.imshow(np.fft.fftshift(new_bpp))
psf2d = np.fft.fft2(new_bpp)
tf.imshow(np.fft.fftshift(psf2d))
tf.imsave('single lightsheet.tif', np.fft.fftshift(np.abs(psf2d)))
# Calculate thickness of light-sheet 

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt

fip = tf.imread('single lightsheet.tif')
plt.imshow(fip)
plt.title('image')
line = fip[512,:] 
line_length = 200
crop_away = 1024-line_length
line = line[crop_away//2:1024-int(np.ceil(crop_away/2))]
xc = np.arange(line.shape[0])
H, A, x0, sigma = gauss_fit(xc, line)
FWHM = abs(2.35482 * sigma) * 0.1

print('The offset of the gaussian baseline is', H)
print('The center of the gaussian fit is', x0)
print('The sigma of the gaussian fit is', sigma)
print('The maximum intensity of the gaussian fit is', H + A)
print('The Amplitude of the gaussian fit is', A)
print('The FWHM of the gaussian fit is', FWHM)
print('The Width of the gaussian fit is', FWHM * 1.18)

plt.figure()
plt.plot(xc, line, 'k', label='data')
plt.plot(xc, gauss(xc, *gauss_fit(xc, line)), '--r', label='fit')
plt.legend()
plt.xlabel('Position')
plt.ylabel('Intensity (A)')

# F = 164.5 / 20
# M = 200 / 150
# T = 4 * 0.53 * F / (np.pi * 10.96 * M) #um
