#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Python module for fitting cutout radio images. 

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


# Scipy stuff:
import scipy.optimize as opt

# Astropy stuff:
from astropy import units as u
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from astropy.io.votable import writeto as writetoVO
from astropy.wcs import WCS

# Image processing packages:
from skimage.feature import blob_dog, blob_log

def FWHM2sig(FWHM):
    """
    Calculate the sigma width of a Gaussian from the full width half maximum.
    Parameters:
    ----------
    FWHM : float 
        Gaussian full widht half maximum.
            
    Returns:
    ----------
    sig : float
        Gaussian width.
    """ 
    sig = FWHM/(2*np.sqrt(2*np.log(2)))
    return sig

def sig2FWHM(sig):
    """
    Calculate the full width half maximum from the sigma width of a Gaussian.

    Parameters:
    ----------
    sig : float 
        Major axis size of the beam.
            
    Returns:
    ----------
    FWHM : float
        Full widht half maximum.
    """ 
    FWHM = sig*(2*np.sqrt(2*np.log(2)))
    return FWHM

def Beam_solid_angle(major,minor):
    """
    Calculates the solid angle of a Gaussian beam.
            
    Parameters:
    ----------
    major : float 
        Major axis size of the beam.
    minor : float 
        Minor axis size of the beam.
            
    Returns:
    ----------
    solid_angle : float
        Gaussian beam solid angle.
    """

    #return np.pi*major*minor/(4*np.log(2))
    solid_angle = np.pi*FWHM2sig(major)*FWHM2sig(minor)
    return solid_angle


def Gaussian2D(xdata_tuple, amplitude, x0, y0, sigma_x, sigma_y, theta):
    """
    Generalised 2DGaussian function.
    
    Parameters:
    ----------
    xdata_Tuple : tuple
        Tuple containing the X-data and Y-data arrays.
    amplitude : float
        Gaussian amplitude.
    x0 : float
        Gaussian peak x-cooridinate.
    y0 : float
        Gaussian peak y-cooridinate.
    sigma_x : float
        2D Gaussian x width.
    sigma_y : float
        2D Gaussian y width.
    theta : float
        2D Gaussian position angle. In radians.
            
    Returns:
    ----------
    g : numpy array
        2D numpy array, the N_Gaussian image.
    """
    
    (x,y) = xdata_tuple
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))

    return g

## Gaussian and fitting functions.
def NGaussian2D(xdata_tuple, *params, fit=True):
    """
    Generalised 2DGaussian function, used primarily for fitting Gaussians to 
    islands. Also used to create N-component Gaussian model images. 
    
    Parameters:
    ----------
    xdata_Tuple : tuple
        Tuple containing the X-data and Y-data arrays.
    params : numpy array
        Array of the Gaussian parameters, should be 6*N_Guassians. 1 dimension.
            
    Returns:
    ----------
    zz : numpy array
        2D numpy array, the N_Gaussian image.
    """
    
    # Initialising the data array.
    (x,y) = xdata_tuple
    zz = np.zeros(np.shape(x))

    if fit == False:
        params = params[0]
    else:
        pass
    
    # Looping through all the Gaussians. Parameter array has to be 1D, 
    # each Gaussian parameter set separated by 6 places in the list.
    for i in range(0, len(params), 6):
        amp_temp = params[i]
        x0_temp = params[i + 1]
        y0_temp = params[i + 2]
        sigx_temp = params[i + 3]
        sigy_temp = params[i + 4]
        theta_temp = params[i + 5]

        zz = zz + Gaussian2D(xdata_tuple, amp_temp, x0_temp, y0_temp, 
                                sigx_temp, sigy_temp, theta_temp)

    if fit:
        return zz.ravel()
    else:
        return zz