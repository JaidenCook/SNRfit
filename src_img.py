#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Python module for performing image manipulation on FITS files.

TODO:
Make this a package.
Upgrade the background esimtation method.
Improve the boundary conditions.
Implement Gaussian deconvolution.
"""

__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@curtin.edu.au"

# Generic stuff:
import os,sys
import time
import warnings

import logging
logging.captureWarnings(True) 

# Array stuff:
import numpy as np

# Plotting stuff:
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 12})

plt.rc('xtick', color='k', labelsize='medium', direction='out')
plt.rc('xtick.major', size=6, pad=4)
plt.rc('xtick.minor', size=4, pad=4)

plt.rc('ytick', color='k', labelsize='medium', direction='out')
plt.rc('ytick.major', size=6, pad=4)
plt.rc('ytick.minor', size=4, pad=4)

# Astropy stuff:
from astropy import units as u
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from astropy.io.votable import writeto as writetoVO
from astropy.wcs import WCS

# Image processing packages:
from skimage.feature import blob_dog, blob_log

from src_fit import FWHM2sig, Gaussian2D, NGaussian2D
from src_plot import astro_plot_2D

def calc_img_bkg_rms(image,mask_arr=None,Niter=5,sigma_thresh=2.5,
                     mask_cond=False,plot_cond=False,abscond=False):
    """
    Calculates a constant background and rms for an input image. 
    Can accept input mask images.
    
    Parameters:
    ----------
    image : (array)
        Numpy array containing the image.
    mask_arr : numpy array, default=None
        Numpy array containing masked regions.
    Niter : int, default = 5
        Number of sigma thresholding iterations.
    sigma_thresh : float, default=2.5
        Sigma threshold statistic.
    mask_cond : bool, default=False
        If True return the sigma clipped mask.
    plot_cond : bool, default=False
        If True plot the image with the mask overlaid.
    abscond : bool, default=False
        If True the threshold is on both negative and positive values.
        Default False only thresholds positive values.

            
    Returns:
    ----------
    bkg : float
        Expected background value.
    rms : float
        Expected image root mean squared.
    """
    # Output image gets altered otherwise.
    image = np.ones(image.shape)*image

    if np.any(mask_arr):
        # If mask is provided.
        image[mask_arr==False] = np.NaN

    bkg = np.nanmedian(image)
    rms = np.nanstd(image)

    for i in range(Niter):

        # Calculating the threshold mask.
        if abscond==False:
            threshold_pve = bkg + sigma_thresh*rms
            threshold_nve = bkg - sigma_thresh*rms
            thresh_mask = np.logical_or((image > threshold_pve),(
                image < threshold_nve))
        else:
            thresh_mask = np.abs(image) > threshold_pve
            
        # Subsetting the data.
        image[thresh_mask] = np.NaN
    
        # Recalculate the bkg and rms:
        bkg = np.nanmedian(image)
        rms = np.nanstd(image)

        #print(f'Max pixel = {np.nanmax(image):5.4f}')
        
        if i == (Niter-1) and plot_cond:
            plt.imshow(image,origin='lower')
            plt.show()

    if mask_cond:
        thresh_mask = np.isnan(image)
        return bkg,rms,thresh_mask
    else:
        return bkg,rms


def calc_footprint(a,b,pa):
    """
    If given then calculate the footprint. The footprint is 
    the PSF.

    Parameters:
    ----------
    a : float
        Gaussian major axis. in Pixel units.
    b : float
        Gaussian mino axis. in Pixel units.
    pa : float
        Position angle in units of degrees.

    Returns:
    ----------
    footrpint : numpy array, bool
        2D numpy array containing 1 where the footprint is and zero elsewhere.
    """
    
    # Determined by the input variables.
    Naxis = np.rint(a + b).astype(int)

    if Naxis % 2 == 0:
        Naxis += 1
        
    xx_psf,yy_psf = np.mgrid[0:Naxis,0:Naxis]

    # Should be in degrees. This is for consistency purposes.
    pa = np.radians(pa)

    # Centre coordinates.
    x0 = int(Naxis/2)
    y0 = int(Naxis/2)

    # Calculate the footprint.
    footprint =  Gaussian2D((xx_psf,yy_psf), 1, 
                            x0, y0, FWHM2sig(a), 
                            FWHM2sig(b), pa)
    
    # Setting the footprint values.
    footprint[footprint >= 0.25] = 1.
    footprint[footprint < 0.25] = 0

    return footprint


def island_calc(img,peaks_vec,eps=0.7,foot_params=None,verbose=False,**kwargs):
    """
    Function for determining the island around a given set of sources. 
    Each source is represented by a peak value which has an associated x and y 
    coordinate. Each source also has an expected scale which is determined by 
    the peak detection software. We can use this scale to estimate the number of 
    pixels that should be in the island. If the island has many more pixels than 
    the expected value, the tolerance is probably incorrect. That or the source 
    could be some kind of artefact noise or otherwise. 

    Parameters:
    ----------
    img : numpy array, float
        2D numpy array containing the image data.
    peaks_vec : numpy array, float
        2D numpy array containing the x and y coordinates of all peaks, as 
        well as the scale in terms of sigma. This is how large the source 
        is expected to be. These should be in pixel units.
    eps : float, default=0.7
        Epsilon is the tolerance factor, this is a relative factor, relative
        to the peak intensity.
    foot_params : tuple, default=None
        The major (a), minor (b) and position angle of
        an elliptical Gaussian. Should be the image PSF. Used to calculate
        the footprint of the peaks. PA should be in degrees.
    **kwargs :
        Keyword arguments for scikit-images function flood().
    
            
    Returns:
    ----------
    island_cube : numpy array, bool
        3D numpy array with each slice having the same shape as img. Each
        slice is the boolean mask for each identified island.
    
    """
    from skimage.segmentation import flood

    ## TODO
    # Need to figure out how to group point that are associated to the same
    # source. 

    if np.any(foot_params):
        a,b,pa = foot_params
        footprint = calc_footprint(a,b,pa)
    else:
        footprint = None
    
    # Number of sources/peaks.
    Npeaks = peaks_vec.shape[0]

    #
    if (peaks_vec.shape[-1] < 3) or foot_params:
        sigmas = FWHM2sig(a)*np.ones(Npeaks)
        Npix_est = np.ceil(2*np.pi*FWHM2sig(a)*FWHM2sig(b))*np.ones(Npeaks)
    else:
        # If the peak vec has a third column assume it is the size column.
        sigmas = peaks_vec[:,-1]
    
        Npix_est = np.ceil(2*np.pi*sigmas**2) # Npixels estimate.

    # Creating the island mask cube. Each slice is for a given
    # source. 
    island_cube = np.zeros(img.shape + (Npeaks,)).astype(bool)
    for ind in range(Npeaks):

        # Getting the xcoord/index values.
        xcoord = int(peaks_vec[ind,0])
        ycoord = int(peaks_vec[ind,1])
        
        # Calculating unique tolerance.
        tol = img[xcoord,ycoord]*eps

        # Calculating the flood mask. The byteswap is a bug fix for the
        # data types passed in scikit-images.
        flood_mask = flood(img.byteswap().newbyteorder(),(xcoord,ycoord),
                    footprint=footprint,
                    tolerance=tol,**kwargs)
        
        # Calc the number of pixels in the island.
        Npix_island = img[flood_mask].size

        # Assign the mask to the island.
        island_cube[:,:,ind] = flood_mask

        if verbose:
            print(Npix_island,Npix_est[ind])

    return island_cube



def generate_correlated_noise(std,psf_params,img_dims,
                               verbose=False,threshold=1e-1,w=None,
                               return_cond=False):
    """
    For a given input standard deviation (calculated from some image noise),
    input image psf parameters, and image dimensions, generate spatially 
    correlated noise.

    The psf here is assumed to be Gaussian, therefore noise is correlated on 
    spatial scales related to the Gaussian size. This is an accurate first order 
    approximation of interferometric noise.

    Parameters:
    ----------
    std : float
        Standard deviation of image noise.
    psf_params : list
        List of psf major, minor and position angle. Major and minor axes should 
        be units of pixels.
    img_dims : tuple
        Tuple containing the image X and Y axis pixel sizes. 
    verbose : bool, default=False
        If True prints extra information. 
    w : astropy object, default=None
        Astropy world coordinate system, if given this function plots the 
        uncorrelated noise and the psf.
    return_cond : bool,default=False
        If True return the PSF and uncorrelated noise maps.
    """
    from scipy.signal import convolve2d

    # Getting the indices of the coordinate grid. 
    # Numpy grid ordering is different to the ordering I specify.
    # There is an option to change this.
    yy_pix,xx_pix = np.indices(img_dims)

    # Defining the centre of the coordinate grid. 
    x_cent = int(img_dims[0]/2)
    y_cent = int(img_dims[1]/2)

    # Creating the coordinate grid. 
    yy_pix = yy_pix - y_cent
    xx_pix = xx_pix - x_cent

    # Creating the PSF function.
    a_psf,b_psf,theta_psf = psf_params

    sigx_psf = a_psf/(2*np.sqrt(2*np.log(2))) 
    sigy_psf = b_psf/(2*np.sqrt(2*np.log(2))) 

    ppsf = np.array([1,0,0,sigx_psf,sigy_psf,theta_psf])

    # Generating the image. 
    img_psf = NGaussian2D((xx_pix,yy_pix),ppsf,fit=False).reshape(img_dims[0],
                                                                  img_dims[1])

    if np.any(w):
        print('Plotting psf image...')
        #astro_plot_2D(img_psf, w, figsize=(7.5,6),scale=0.6)
        astro_plot_2D(img_psf, w, figsize=(7.5,6))

    # Generating the uncorrelated noise.
    # The uncorrelated noise needs to be scaled by the area of the PSF beam.
    # We assume the mean is zero, the background can be added to the output image later. 
    img_noise = np.random.normal(0,scale=std,size=img_dims)

    if np.any(w):
        print('Plotting uncorrelated noise image...')
        #astro_plot_2D(img_noise, w, figsize=(7.5,6),scale=0.6)
        astro_plot_2D(img_noise, w, figsize=(7.5,6))

    # Calculating the scaling factor. 
    w_a = (a_psf)/np.sqrt(2*np.log(2))
    w_b = (b_psf)/np.sqrt(2*np.log(2))

    rescale = np.sqrt(np.pi*w_a*w_b)

    # Normalising the PSF.
    img_psf = img_psf/np.sum(img_psf)

    # Calculating the rescaled image. 
    img_scaled = img_noise*rescale

    # Convoling the psf with the noise.
    img_correlated_noise = convolve2d(img_psf,img_scaled,mode='same',
                                      boundary='wrap')

    std_cor = np.nanstd(img_correlated_noise)

    # Error checking. The standard deviations should be similar.
    std_diff = np.abs(std_cor - std)
    if std_diff > threshold:
        print(std_cor,std)
        err_msg = f'Difference standard deviation above threshold {std_diff:5.3e}'
        img_correlated_noise = (std/std_cor)*img_correlated_noise
        #raise ValueError(err_msg)
    
    if verbose:
        print(f'Rescaling factor = {rescale:5.3f}')
        print(f'Input standard deviation is {std:5.3f}')
        print(f'Correlated standard deviation is {std_cor:5.3f}')
    
    if return_cond:
        return img_correlated_noise,img_psf,img_noise
    else:
        return img_correlated_noise


def determine_peaks_bkg(image_nu,constants,maj_fac=1,num_sigma=20,
            thresh_fac=1,overlap=1,log_cond=False):
    """
    Determines the number of peaks, their locations, sizes and amplitudes for an
    input image. Builds the core of the model parameter guess.
    
    Parameters
    ----------
    image_nu : numpy array 
        2D numpy array containing the image. Should be background subtracted.
    constants : tuple
        tuple containing (a_psf,Major,dx,rms), which is the psf major axis, 
        source major axis and pixel size in degrees, as well as the image rms.
    maj_frac : float, default=1 
        Major axis fraction scaling. Should be less than 1.
    num_sigma : int, default=20
        Number of sigma for laplacian of Gaussians method.
    thresh_fac : float, default=1
        Multiplicative factor for rms threshold for peak detection.
    overlap : float, default=1
        Blob overlap factor, shouldn't be greater than 1.
    log_cond : bool, default=False
        If True use log scale for determining blob sizes.
    
    Returns
    ----------
    coord_sigma_amp_vec : numpy array
        numpy array containing the coordinates (x,y), sigma and amp for each 
        peak.
    """
    (a_psf,Major,dx,rms) = constants

    num_sigma = int(num_sigma)

    if overlap > 1:
        raise ValueError("'overlap' variable cannot be greater than 1.")
    
    # Calculating the minimum sigma in pixel coordinates.
    min_sigma = (a_psf/dx)/(2*np.sqrt(2*np.log(2)))

    print('Min sigma =  %5.3f [pix]' % min_sigma)

    #max_angle = 0.5*Major/60 # deg
    max_angle = Major/60 # deg

    # Max sigma is determined by the SNR size. 
    max_sigma = maj_fac*(max_angle/dx)/(2*np.sqrt(2*np.log(2)))

    print('Maximum sigma = %5.3f' % max_sigma)

    # coord_log = [xcoord,ycoord,blob_size]
    # Get the location of the peaks in x,y as well as the expected blob size. 
    coordinates_sigma_vec = blob_log(image_nu,min_sigma=min_sigma,
                                     max_sigma=max_sigma,log_scale=log_cond,
                                     threshold=rms*thresh_fac,
                                     num_sigma=num_sigma,overlap=overlap)

    print('Performing peak detection with %5.3e threshold.' % (rms*thresh_fac))

    try:
        N_peaks = len(coordinates_sigma_vec)
        print('Detected an initial %s peaks.' % N_peaks)
    except TypeError:
        print('No peaks detected. Adjust input parameters.')
        N_peaks = 0
        return None
    else:

        # Creating new coord object. 
        coord_sigma_amp_vec = np.zeros((len(coordinates_sigma_vec),4))
        # First three columns are equal to coord_log.
        coord_sigma_amp_vec[:,:3] = coordinates_sigma_vec

        # Guessing the Gaussian amplitude as the pixel value closest to the peak 
        # coordinates.
        coord_sigma_amp_vec[:,3] = \
            image_nu[(coordinates_sigma_vec[:,0]-1).astype(int),
                     (coordinates_sigma_vec[:,1]-1).astype(int)]

        return coord_sigma_amp_vec
    
def convolve_image(image,header,Gauss_size):
    """
    Simple image convolver. Takes input image, header, and new Gaussian size in 
    degrees. 

    Parameters:
    ----------
    image : numpy array
        Numpy array containing the image.
    header : astropy object
        Astropy FITS header image object.
    Gauss_size : float
        Convolving Gaussian size in degrees.
            
    Returns:
    ----------
    image_nu : numpy array
        Numpy array containing the convolved image.
    header : astropy object
        Updated header.

    """

    from scipy.ndimage import gaussian_filter

    dx = header['CD2_2'] #pixel size in degrees [deg]

    N_pix = int(Gauss_size/dx)

    print('Gaussian width = %3i [pix]' % N_pix)

    image_nu = gaussian_filter(image, sigma=N_pix)

    # Getting header Major and minor 
    bmaj = float(header['BMAJ']) # [deg]
    bmin = float(header['BMIN']) # [deg]
    
    # Width of the convolving Gaussian.
    sigma = N_pix*dx # [deg]
    major = 2*np.sqrt(2*np.log(2))*sigma # Major axis [deg]

    # Calculating the new restoing beam major and minor axes in degrees.
    bmaj_nu = np.sqrt(major**2 + bmaj**2)
    bmin_nu = np.sqrt(major**2 + bmin**2)

    header['BMAJ'] = bmaj_nu
    header['BMIN'] = bmin_nu

    return image_nu,header