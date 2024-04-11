#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Module for spectral fitting. 

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

# Scipy stuff:
import scipy.optimize as opt

from functions import *

def spec_fit(freqs,fluxVec,func=power_law,sigma=None,bounds=True,
             covmatrix=False,perrcond=True):
    """
    Fit the spectra of a radio source. 
    
    Parameters:
    ----------
    freqs : float, numpy ndarray
        Numpy vector of frequencies.
    fluxVec : float, numpy ndarray
        Numpy vector of flux densities in Jy.
    func : function, default=power_law
        Model function
    sigma : float, or numpy ndarray
        Float of rms, or vector of uncertainties for each frequency.
    bounds : bool, default=True
        If True calculate the bounds.
    covmatrix : bool, default=False
        If True calculate the covariance matrix. Sigma must also not be None.
    perrcond : bool, default=True
        If True only return the diagonal elements of the output covariance 
        matrix, these are also squarerooted.

    Returns:
    ----------
    popt : numpy array
        2D numpy array containing the best fit parameters.
    pcov : numpy array
        2D numpy array containing the covariance matrix for the fit parameters.
    """
    freq0 = freqs[0]
    naninds = np.isnan(fluxVec) == False
    freqs = freqs[naninds]
    fluxVec = fluxVec[naninds]
    
    if covmatrix and np.any(sigma):
        # Calc covariance matrix.
        if isinstance(sigma, np.ndarray):
            # If sigma is an array.
            sigma = sigma[naninds]

        covMat = matern_cov(freqs,sigma)
        sigma = covMat
        print('Covariance matrix calculated...')

    if bounds:
        # Sensible bounds set on the parameters.
        bounds_low = np.array([0,-5])
        bounds_hi = np.array([np.inf,5])
        bounds = (bounds_low,bounds_hi)
    else:
        bounds = None

    popt,pcov = opt.curve_fit(func,freqs/freq0,fluxVec,
                              p0=[1,-0.7],maxfev=int(1e5),sigma=sigma,
                              jac=jac_power_law,bounds=bounds)
    
    if perrcond:
        pcov = np.sqrt(np.diag(pcov))

    return popt,pcov