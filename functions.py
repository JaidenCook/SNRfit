#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Python module for fitting cutout radio images. 

TODO:
Make this a package.
"""

__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook@curtin.edu.au"

import logging
logging.captureWarnings(True) 

# Array stuff:
import numpy as np

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

def paRA2paXY(pa,degrees=True):
    """
    Rotate the position angle from RA and DEC orientation to
    pixel coordinates.
    """

    if degrees:
        pa = 360 - (pa + 90)
    else:
        pa = 2*np.pi - (pa + np.pi/2)

    return pa

def paXYpaRA2(pa,degrees=True):
    """
    Rotate the position angle from RA and DEC orientation to
    pixel coordinates.
    """

    if degrees:
        pa = 360 - (pa + 90)
    else:
        pa = 2*np.pi - (pa + np.pi/2)

    return pa

def Beam_solid_angle(major,minor,degrees=True):
    """
    Calculates the solid angle of a Gaussian beam.
            
    Parameters:
    ----------
    major : float 
        Major axis size of the beam in radians.
    minor : float 
        Minor axis size of the beam in radians.
    degrees : bool, default=True
        If True the input units are in degrees.
            
    Returns:
    ----------
    solid_angle : float
        Gaussian beam solid angle.
    """
    if degrees:
        major = np.radians(major)
        minor = np.radians(minor)

    solid_angle = 2*np.pi*FWHM2sig(major)*FWHM2sig(minor)
    return solid_angle

def Sint_calc(Speak,Maj,Min,omegaPSF,e_Speak=None,e_Maj=None,e_Min=None,
              degrees=True):
    """
    Function to propagate the integrated flux density uncertainty.
    
    Parameters:
    ----------
    Speak : float, numpy ndarray
        Float or numpy array of peak flux densities in Jy/beam
    Maj : float, numpy ndarray
        Major axis of Gaussian component(s) in degrees.
    Min : float, numpy ndarray
        Minor axis of Gaussian component(s) in degrees.
    omegaPSF : float, numpy ndarray
        Solid angle of the PSF. Used to determine the number of beams for a 
        given Gaussian component.
    e_Speak : float, numpy ndarray, default=None
        Uncertainty in peak flux density (Jy/beam).
    e_Maj : float, numpy ndarray, default=None
        Uncertainty in Major axis (deg).
    e_Min : float, numpy ndarray, default=None
        Uncertainty in Minor axis (deg).
    
    Returns:
    ----------
    Sint : float, numpy ndarray
        Flux density in Jy.
    e_Sint : optional, float, numpy ndarray
        Uncertainty in the flux density.
    """
    # Const in units of beam.
    Nbeam = Beam_solid_angle(Maj,Min,degrees=degrees)/omegaPSF
    Sint = Speak*Nbeam # Jy.

    # If any errors are given calculate the uncertainty in the flux density.
    if np.any(e_Speak) and np.any(e_Min) and np.any(e_Maj):
        e_Sint=Sint*np.sqrt((e_Maj/Maj)**2 +(e_Min/Min)**2 +(e_Speak/Speak)**2)
        
        return Sint, e_Sint
    else:
        return Sint


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
    for i in range(0,len(params),6):
        amp_temp = params[i]
        x0_temp = params[i+1]
        y0_temp = params[i+2]
        sigx_temp = params[i+3]
        sigy_temp = params[i+4]
        theta_temp = params[i+5]

        zz = zz + Gaussian2D(xdata_tuple, amp_temp, x0_temp, y0_temp, 
                             sigx_temp, sigy_temp, theta_temp)

    if fit:
        return zz.ravel()
    else:
        return zz
    

def power_law(freq,S0,alpha,freq0=160e6):
    """
    Power law function for calculating flux density.

    Parameters:
    ----------
    freq : float, or numpy ndarray
        Float or vector of frequencies. Typcally normalised by some reference
        frequency nu0. If not given it is assumed to be nu=1Hz.
    S0 : float
        This is the flux density at the reference frequency.
    alpha : float
        This is the spectral index.
    freq0 : float, default=160e6
        Reference frequency in Hz.
            
    Returns:
    ----------
    Snu : float, or numpy ndarray
        Flux density at frequency nu.
    """

    Snu = S0*(freq/freq0)**(alpha)
    return Snu

def curved_power_law(freq,S0,alpha,beta,freq0=160e6):
    """
    Power law function for calculating flux density.

    Parameters:
    ----------
    freq : float, or numpy ndarray
        Float or vector of frequencies. Typcally normalised by some reference
        frequency nu0. If not given it is assumed to be nu=1Hz.
    S0 : float
        This is the flux density at the reference frequency.
    alpha : float
        This is the spectral index.
    beta : float
        Curvature.
    freq0 : float, default=160e6
        Reference frequency in Hz.
            
    Returns:
    ----------
    Snu : float, or numpy ndarray
        Flux density at frequency nu.
    """

    Snu = S0*((freq/freq0)**(alpha))*np.exp(beta*(np.log(freq/freq0))**2)
    return Snu

def jac_power_law(freq,S0,alpha,freq0=160e6):
    """
    Parameters:
    ----------
    freq : float, or numpy ndarray
        Float or vector of frequencies. Typcally normalised by some reference
        frequency nu0. If not given it is assumed to be nu=1Hz.
    S0 : float
        This is the flux density at the reference frequency.
    alpha : float
        This is the spectral index.
    freq0 : float, default=160e6
        Reference frequency in Hz.
            
    Returns:
    ----------
    """
    dfdS0 = (freq/freq0)**(alpha)
    dfda = S0*np.log(freq/freq0)*(freq/freq0)**alpha

    jacVec = np.array([dfdS0,dfda]).T

    return jacVec


def matern_cov(r,sigma,r0=7.28e6):
    """
    Matern covariance kernel used for estimating the covariance between channels.
    """
    if isinstance(sigma, np.ndarray):
        if len(sigma) != len(r):
            errmsg = f'Length of sigma not equal to r'
            raise ValueError(errmsg)
    sigx,sigy = np.meshgrid(sigma,sigma)
    sigMat = sigx*sigy
    
    # Assuming that the channels are independent.
    # Some channels might be missing.
    #r0 = np.min(np.diff(r))
    rx,ry = np.meshgrid(r,r)
    dr = np.abs(rx-ry)
    
    k= sigMat*(1+np.sqrt(3)*(dr/r0))*np.exp(-np.sqrt(3)*(dr/r0))
    #k=sigMat*(1+np.sqrt(5)*(dr/r0)+(5/3)*(dr/r0)**2)*np.exp(-np.sqrt(5)*(dr/r0))

    return k

