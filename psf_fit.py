#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Python module for fitting PSF in cutout radio images. 

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
from functions import FWHM2sig,Beam_solid_angle,sig2FWHM
from src_img import create_model_mask,footprint_mask,calc_footprint
from src_fit import SNR_Gauss_fit


def avg_psf(poptArr,e_poptArr,rms,boolcond=False,verbose=False):
    """
    Calculate the average psf from an input array of fit values, their associated,
    errors, and the image rms.
    
    Parameters:
    ----------
    poptArr : numpy array
        Array containing the fitted Gaussian values for each point source.
    e_poptArr : numpy array
        Associated errors for the fit components.
    rms : float
        Image rms.
    boolcond : bool, default=False
        If True return the boolean value vector.
    verbose : bool, defualt=False
        If True print values as output.

    Returns:
    ----------
    avgPSFparams : tuple, float
        Tuple containing the fit psf params (amaj,bmaj,pa) with pa in radians.
    boolVec : numpy array, bool
        Vector of boolean values, True for sources deemed as point sources.
    """
    aMod = sig2FWHM(poptArr[:,3])*dx
    bMod = sig2FWHM(poptArr[:,4])*dx
    paMod = poptArr[:,-1]
    e_aMod = sig2FWHM(e_poptArr[:,3])*dx
    e_bMod = sig2FWHM(e_poptArr[:,4])*dx
    e_paMode = e_poptArr[:,-1]
    
    paModRatio = poptArr[:,-1]/np.radians(theta_PA)
    beamRatio = Beam_solid_angle(aMod,bMod,degrees=True)/omegaPSF
    SNRVec = poptArr[:,0]/rms
    
    # Calculate the boolean vectors, and combine them together.
    SNRBoolVec = SNRVec >= 5
    paModBoolVec = (paModRatio > 1.1) & (paModRatio < 0.9)
    boolVec = SNRBoolVec | paModBoolVec

    # Calculating the weights.
    w = SNRVec/beamRatio

    # Calculate the weighted average of the fit psf values.
    apsfAvg = np.average(aMod[boolVec],weights=w[boolVec]/e_aMod[boolVec])
    bpsfAvg = np.average(bMod[boolVec],weights=w[boolVec]/e_bMod[boolVec])
    paAvg = np.average(paMod[boolVec],weights=w[boolVec]/e_paMode[boolVec])
    # Collate the average parameters into a tuple.
    avgPSFparams = (apsfAvg,bpsfAvg,paAvg)

    if verbose:
        print(SNRVec)
        print(beamRatio)
        print(psfParams)
        print(FWHM2sig(apsfAvg)/dx,FWHM2sig(bpsfAvg)/dx,np.degrees(paAvg))
    
    if boolcond:
        return avgPSFparams,boolVec
    else:
        return avgPSFparams


def fit_psf(pointCoordArr,image,psfParams,maskList,rms,
            verbose=False,boolcond=False):
    """
    Fit the PSF for each of the input sources, assuming they are point sources.
    These are then averaged with a different function that returns the averaged
    psf parameters.

    Parameters:
    ----------
    pointCoordArr : numpy array, float
        Contains the pixel coordinates for all sources to be fit.
    image : numpy array, float,
        2D numpy array containing image information.
    maskList : list
        List contains 2D boolean numpy mask arrays for each source.
    rms : float
        Image rms.
    boolcond : bool, default=False
        If True return the boolean value vector.
    verbose : bool, defualt=False
        If True print values as output.
    
        
    Returns:
    ----------
    avgPSFparams : tuple, float
        Tuple containing the fit psf params (amaj,bmaj,pa) with pa in radians.
    pointCoordArr : numpy array, float, optional
        Contains the pixel coordinates for all sources to be fit. Thresholded 
        for all sources that didn't match the point source criteria.
    """
    xVec = np.arange(image.shape[0])
    yVec = np.arange(image.shape[1])
    xx,yy = np.meshgrid(xVec,yVec)
    
    if pointCoordArr.shape[0] > 20:
        maskList = maskList[:20]
        NpointSource = 20
    else:
        NpointSource = pointCoordArr.shape[0]

    poptArr = np.zeros((NpointSource,6))
    e_poptArr = np.zeros((NpointSource,6))
    for ind,mask in enumerate(maskList):

        imgMasked = np.copy(image) - bkg
        imgMasked[mask==False] = np.NaN
        
        # Getting the data.
        data = imgMasked[mask]

        # Getting the coords for the point source.
        coordTemp = np.array([pointCoordArr[ind,:]])

        popt,perr = SNR_Gauss_fit(xx[mask],yy[mask],data,coordTemp,
                                  psfParams,maj_frac=1,rms=rms,verbose=False)
        
        poptArr[ind,:] = popt
        e_poptArr[ind,:] = perr
        
    # Calculate the average psf.
    if boolcond:
        avgPSFparams,boolVec = avg_psf(poptArr,e_poptArr,rms,
                                       verbose=verbose,boolcond=boolcond)
        pointCoordArr = pointCoordArr[boolVec,:]

        return avgPSFparams,pointCoordArr
    else:
        avgPSFparams = avg_psf(poptArr,e_poptArr,rms,verbose=verbose)

        return avgPSFparams



