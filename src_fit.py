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

from functions import *

def majmin_swap(popt,pcov,degrees=True):
    """
    Function for swapping the major and minor axes of a 2D Gaussian. This is 
    necessary, because the major axis is defined in as the x-axis component of
    the 2D Gaussian. This means that the smaller axis depending on the rotation
    angle will often be swapped. This is fixed by swapping the axes and rotating
    by 90 degrees.

    Parameters:
    ----------
    majVec : ndarray
        Major axis vector.
    minVec : ndarray
        Minor axis vector.
    paVec : ndarray
        Postion axis vector.
    degrees : bool, default=True
        If True paVec is in degrees.

    Returns:
    ----------
    majVec : ndarray
        Major axis vector.
    minVec : ndarray
        Minor axis vector.
    paVec : ndarray
        Postion axis vector.
    """
    
    # Boolean vector where axes are swapped.
    swapVec = popt[:,3] < popt[:,4]


    if np.any(swapVec):
        # Creating temp vectors from copied numpy arrays.
        majVec_temp = np.copy(popt[:,3])
        minVec_temp = np.copy(popt[:,4])
        paVec_temp = np.copy(popt[:,5])
        
        # Swapping axes.
        majVec_temp[swapVec] = popt[:,4][swapVec]
        minVec_temp[swapVec] = popt[:,3][swapVec]

        # Performing the rotation.
        if degrees:
            # Default 
            paVec_temp[swapVec] += 90 
        else:
            paVec_temp[swapVec] += np.pi/2

        # Assigning the temp vectors to input maj and min vecs.
        popt[:,3] = majVec_temp
        popt[:,4] = minVec_temp
        popt[:,5] = paVec_temp
    
    if pcov.shape[0] != pcov.shape[1]:
        # If only the errors given.
        e_majVec_temp = np.copy(pcov[:,3])
        e_minVec_temp = np.copy(pcov[:,4])
        
        # Swapping axes.
        e_majVec_temp[swapVec] = pcov[:,4][swapVec]
        e_minVec_temp[swapVec] = pcov[:,3][swapVec]

        # Assigning the temp vectors to input maj and min vecs.
        pcov[:,3] = majVec_temp
        pcov[:,4] = minVec_temp
    
    elif pcov.shape[0] == pcov.shape[1]:
        # If the full covariance matrix is given.
        indVec = np.arange(popt.size)
        xindArr,yindArr = np.meshgrid(indVec,indVec)

        pcov = pcov[xindArr,yindArr]

    return popt,pcov

def Gaussian_2Dfit(xx,yy,data,pguess,func=NGaussian2D,sigma=None,
                   pbound_low=None,pbound_up=None,
                   maxfev=10000000):
    """
    Wrapper function for the Gaussian_2Dfit function, which fits the NGaussian2D 
    function using scipy.optimise.curve_fit(), which uses a non-linear least 
    squares method. In future this function should be refactored, especially if 
    we want to consider using different fitting methods such as a Bayesian 
    approach.    

    Parameters:
    ----------
    xx : numpy array
        Numpy array containing the x-positions for the data, should have 
        dimension 1.
    yy : numpy array 
        Numpy array containing the y-positions for the data, should have 
        dimension 1.
    data : numpy array
        Numpy array containing the image data, should have dimensions 1.
    pguess : numpy array
        Numpy array containing the initial parameters.
    func : function, default=NGaussian2D()
        Function which the parameters is supplied to. This function is fit to 
        the data.
    sigma : numpy array, float, default=None
        Either a single float or numpy array.
    pbound_low : numpy array, default=None
        Lower bound on the parameter ranges.
    pbound_up : numpy array, default=None
        Upper bound on the parameter ranges.
            
    Returns:
    ----------
    popt : numpy array
        2D numpy array containing the best fit parameters.
    perr : numpy array
        2D numpy array containing the covariance matrix for the fit parameters.
    """  
    
    if np.any(pbound_low) and np.any(pbound_up):
        # If bounds supplied.
        popt,pcov = opt.curve_fit(func,(xx,yy),data.ravel(),p0=pguess.ravel(),
                                bounds=(pbound_low.ravel(),pbound_up.ravel()),
                                maxfev=maxfev,sigma=sigma)
                                #,method="dogbox"
    else:
        # If no bounds supplied.
        popt, pcov = opt.curve_fit(func,(xx,yy),data.ravel(),p0=pguess.ravel(),
                                maxfev=maxfev,sigma=sigma)

    
    popt = popt.reshape(np.shape(pguess)) # Fit parameters.
    
    popt,pcov = majmin_swap(popt,pcov,degrees=False)

    return popt,pcov
    

# // This function needs to be refactored. 
def SNR_Gauss_fit(xx,yy,data,coords,constants,maj_frac=0.125,
                  allow_negative=False,bounds=True,rms=None,perrcond=True):
    """
    Wrapper function for the Gaussian_2Dfit function, which fits the NGaussian2D 
    function using scipy.optimise.curve_fit(), which uses a non-linear least 
    squares method. In future this function should be refactored, especially if 
    we want to consider using different fitting methods such as a Bayesian 
    approach.    

    Parameters:
    ----------
    xx : numpy array
        Numpy array containing the x-positions for the data, should have 
        dimension 1.
    yy : numpy array 
        Numpy array containing the y-positions for the data, should have 
        dimension 1.
    data : numpy array
        Numpy array containing the image data, should have dimensions 1.
    coords : numpy array
        Gaussian component x,y position array, has dimension 2.
    maj_min : tuple
        Contains the major and minor axis of the SNR in arcminutes.
    maj_frac : float, default=0.125
        Fractional size limit of fit Gaussians, as a fraction of the Major axis. 
    rms : float, default=None,
        If given calculate the covariance matrix.
    perrcond : bool, default=True
        If False return the full covariance matrix.
            
    Returns:
    ----------
    popt : numpy array
        2D numpy array containing the best fit parameters.
    perr : numpy array
        2D numpy array containing the covariance matrix for the fit parameters.
    pcov : numpy array, optional
        2D covariance matrix returned by scipy.curve_fit().
    """

    # Major and Minor axis in arcminutes of the SNR.
    major = constants[0]
    #minor = constants[1]
    dx = constants[2] #pixel size in degrees [deg]

    # Restoring beam parameters. Used in the guess.
    pixel_scale = dx*3600 # [arcsec] 
    a_psf = constants[3]*3600 # [arcsec]
    b_psf = constants[4]*3600 # [arsec]
    PA_psf = np.abs(np.radians(constants[5])) # [rads]

    # Restoring beam sizes in pixels:
    sigxPSF = FWHM2sig(a_psf/pixel_scale)
    sigyPSF = FWHM2sig(b_psf/pixel_scale)

    if np.any(rms):
        # If given calculate the covariance matrix.
        psfParams = [sigxPSF,sigyPSF,PA_psf]
        sigma = calc_covMatrix(xx,yy,rms,psfParams)
        print('Covariance matrix calculated...')
    else:
        sigma=None
    
    # PSF initial parameters. Used to determine pguess.
    p0 = np.array([1,0,0,sigxPSF,sigyPSF,PA_psf])

    # Parameter array dimensions.
    N_gauss = len(coords)
    N_params = len(p0)

    # The guess parameter array.
    pguess = np.ones((N_gauss,N_params))*p0[None,:]

    Max_major = (maj_frac*major/60)/dx # [pix]

    # Defining the min and max peak x location limits.
    x_low = np.nanmin(xx)
    x_hi = np.nanmax(xx)

    # Defining the min and max peak y location limits.
    y_low = np.nanmin(yy)
    y_hi = np.nanmax(yy)

    # Assigning initial positions.
    pguess[:,1] = coords[:,1]
    pguess[:,2] = coords[:,0]

    if coords.shape[1] > 2:
        # The blob fitting method returns an expected sigma for a given peak.
        # We can use this as a guess of the actual sigma.
        
        if coords.shape[1] > 3:
            # peak intensity guess for the Gaussians.

            # Set values less than 0 to be 0.
            coords[:,3][coords[:,3] < 0] = 0
            pguess[:,0] = coords[:,3]
        else:
            pass
        pguess[:,3][coords[:,2]>sigxPSF] = coords[:,2][coords[:,2]>sigxPSF]
        pguess[:,4][coords[:,2]>sigyPSF] = coords[:,2][coords[:,2]>sigyPSF]

    # Specifying the lower fit bounds for each Gaussian.
    if allow_negative:
        # If true allow fitting negative peaks.
        # Template 1D array.
        pbound_low = np.array([-np.inf,x_low,y_low,sigxPSF,sigyPSF,0.0]) 
    else:
        #pbound_low = np.array([0.0,x_low,y_low,sigxPSF,sigyPSF,0.0]) 
        pbound_low = np.array([0.0,x_low+sigxPSF/2,y_low+sigxPSF/2,
                               sigxPSF,sigyPSF,0.0]) 

    # Expanding for each Gaussian component.
    pbound_low = np.ones((N_gauss,N_params))*pbound_low[None,:] 
    #pbound_up = np.array([np.inf,x_hi,y_hi,Max_major,Max_major,2*np.pi])
    pbound_up = np.array([np.inf,x_hi-sigxPSF/2,y_hi-sigxPSF/2,
                          Max_major,Max_major,2*np.pi])
    pbound_up = np.ones((N_gauss,N_params))*pbound_up[None,:]

    if bounds:
        # Getting the fit parameters, and their errors.
        popt, pcov = Gaussian_2Dfit(xx,yy,data,pguess,func=NGaussian2D,
                                    pbound_low=pbound_low,pbound_up=pbound_up,
                                    sigma=sigma)
    else:
        popt, pcov = Gaussian_2Dfit(xx,yy,data,pguess,
                                    func=NGaussian2D,sigma=sigma)
    
    if perrcond:
        pcov = np.sqrt(np.diag(pcov)).reshape(np.shape(pguess))

    return popt,pcov


# // This function needs to be refactored. 
def fit_psf(xx,yy,data,coords,psf_params,
            bounds=True,peak_fit=False,perrcond=True):
    """
    Wrapper function for the Gaussian_2Dfit function, which fits the NGaussian2D 
    function using scipy.optimise.curve_fit(), which uses a non-linear least 
    squares method. In future this function should be refactored, especially if 
    we want to consider using different fitting methods such as a Bayesian 
    approach.    

    Parameters:
    ----------
    xx : numpy array
        Numpy array containing the x-positions for the data, should have dim 1.
    yy : numpy array 
        Numpy array containing the y-positions for the data, should have dim 1.
    data : numpy array
        Numpy array containing the image data, should have dimensions 1.
    coords : numpy array
        Gaussian component x,y position array, has dimension 2.
    psf_params : tuple
        Contains the PSF parameters in pixel coords, specifically the sigx, sigy
        and the position angle (pa) in radians.
    bounds : bool, default=True
        If True fit with boundary conditions. Turn off if answers are infeasible.
    peak_fit : bool, default=False
        If True only fit the amplitude. Effectively make the bounds on the other
        parameters small. By defualt sets bounds True if set to False.
    perrcond : bool, default=True
        If False return the full covariance matrix.
            
    Returns:
    ----------
    popt : numpy array
        2D numpy array containing the best fit parameters.
    perr : numpy array
        2D numpy array containing the covariance matrix for the fit parameters.
    """
    # Getting the PSF params.
    sigx, sigy, pa = psf_params

    # PSF initial parameters. Used to determine pguess.
    p0 = np.array([1,0,0,sigx,sigy,pa])# Template inital parameter array.

    # Parameter array dimensions. 
    # This will cause problems if I only fit a single source. 
    if len(coords.shape) > 1:
    #if coords.shape[-1] > 1:
        # If more than one Gaussian is to be fit.
        N_gauss = len(coords)
    else:
        # Case where only one Gaussian is to be fit.
        coords = np.array([coords])
        N_gauss = 1

    # Number of parameters.
    N_params = len(p0)

    if peak_fit:
        bounds = True

    if bounds:
        # Defining the min and max peak x location limits.
        x_low = np.nanmin(xx)
        x_hi = np.nanmax(xx)

        # Defining the min and max peak y location limits.
        y_low = np.nanmin(yy)
        y_hi = np.nanmax(yy)

        # Template lower bound.
        epsilon = 1e-3
        theta_eps = 1e-6
        pbound_low = np.array([0.0,x_low,y_low,
                            sigx - epsilon*sigx,
                            sigy - epsilon*sigy,
                            pa - theta_eps]) 
        # Template upper bound.
        pbound_up = np.array([np.inf,x_hi,y_hi,
                            sigx + epsilon*sigx,
                            sigy + epsilon*sigy,
                            pa + theta_eps])

        
        # Expanding for each Gaussian component.
        #if N_gauss > 1:
            # If there is more than one Gaussian to fit. 
            #pbound_low = np.ones((N_gauss,N_params))*pbound_low[None,:] 
            #pbound_up = np.ones((N_gauss,N_params))*pbound_up[None,:]

        pbound_low = np.ones((N_gauss,N_params))*pbound_low[None,:] 
        pbound_up = np.ones((N_gauss,N_params))*pbound_up[None,:]

        if peak_fit:
            # Setting the peak position bounds to be restricted to inputs.
            pbound_low[:,1] = coords[:,1] - epsilon
            pbound_low[:,2] = coords[:,0] - epsilon

            pbound_up[:,1] = coords[:,1] + epsilon
            pbound_up[:,2] = coords[:,0] + epsilon

    else:
        pbound_low = None
        pbound_up = None

    # The guess parameter array.
    pguess = np.ones((N_gauss,N_params))*p0[None,:]

    # Assigning initial positions.
    pguess[:,1] = coords[:,1] # x-coord
    pguess[:,2] = coords[:,0] # y-coord

    # Getting the fit parameters, and their errors.
    popt,pcov = Gaussian_2Dfit(xx,yy,data,pguess,
                               func=NGaussian2D,pbound_low=pbound_low,
                               pbound_up=pbound_up)

    if N_gauss == 1:
        # Doesn't need to be 2D if only a single Gaussian.
        popt = popt.flatten()
        if perrcond:
            pcov = np.sqrt(np.diag(pcov))
    else:
        if perrcond:
            pcov = np.sqrt(np.diag(pcov)).reshape(pguess.shape)

    return popt,pcov


def Fit_quality(data,p_mod,xx,yy,rms,reduced_cond=False):
    """
    Returns the Chi-squared value for an input model. Model input given by 
    the parameter array p_mod. The input data should have the background 
    subtracted.
    
    Parameters:
    ----------
    data : numpy array
        Numpy array containing the image data.
    p_mod : numpy array
        2D numpy array containing the model parameters. 
    xx : numpy array
        2D numpy array containing the x-grid pixel coordinates.
    yy : numpy array
        2D numpy array containing the y-grid pixel coordinates.
    rms : float
        Root mean squared of the image. 
    reduced_cond : bool
        Default is False, if True return the reduced Chi-squared.
            
    Returns:
    ----------
    chisqd : float
        Chi-squared value for the fit.
    """
    
    # Calculating the model values:
    zz = NGaussian2D((xx,yy),p_mod,fit=False)

    # Calculating the squared residuals and the chisqd value.
    image_residual_squared = (data - zz)**2
    
    if np.any(np.isnan(data)):
        # If the input image is masked. Masked values will be NaNs.
        # If masked values not discarded then Npix is too large. 
        Npix = len(data[np.isnan(data) == False]) # Number of pixels.
    else:
        # Default if no masked values are present.
        Npix = np.size(data) # Number of pixels.

    # Chi-squared:
    chisqd = (1/(rms)**2)*np.nansum(image_residual_squared)
    # Calculating the reduce Chi-squared, add option to return this 
    # Binstead of chisquared.
    red_chisqd = chisqd/(Npix - np.size(p_mod))

    if reduced_cond:
        # Default is False. If True return the reduced Chi-squared instead.
        return red_chisqd
    else:
        # Default option.
        return chisqd


#
def model_select(params1,params2,xx,yy,data,
                 rms=None,perr1=None,perr2=None,verbose=False):
    """
    Takes input parameters from two Gaussian fit models. Calculates the BIC for 
    each model and rejects the model with the higher BIC. This is usually the 
    single Gaussian fit model.
    
    Parameters:
    ----------
    params1 : numpy array
        1D/2D numpy array containing the model1 fit params for each Gaussian.
    params2 : numpy array
        1D/2D numpy array containing the model2 fit params for each Gaussian.
    xx : numpy array
        Array of x-values.
    yy : numpy array
        Array of y-values.
    data : numpy array
        Data for which to compare the models to.
    rms : float, default=None
        Uncertainty in the data. Assumed to be constant.
    perr1 : numpy array, default=None
        1D/2D numpy array containing the error model1 fit params for each 
        Gaussian.
    perr2 : numpy array, default=None
        1D/2D numpy array containing the error model2 fit parameters for each 
        Gaussian.
    verbose : bool, default=False
        If True print additional information.
            
    Returns:
    ----------
    params : numpy array
        Selected parameter array.
    perr : numpy array, default=None
        Selected parameter error array.
    """

    if np.any(rms):
        pass
    else:
        # If not given calculate from the input data. Result will likely be 
        # biased.
        from src_img import calc_img_bkg_rms
        _,rms = calc_img_bkg_rms(data,plot_cond=True)

    # Calculating the chi squared and reduced chi squared values.
    chi1 = Fit_quality(data,params1.ravel(),xx,yy,rms)
    chi2 = Fit_quality(data,params2.ravel(),xx,yy,rms)

    # Number of parameters.
    k1 = np.size(params1) # model 1
    k2 = np.size(params2) # model 2

    # Number of data points.
    n = np.size(data)

    # Calculating the BIC for both models.
    BIC_1 = chi1 + k1*np.log(n)
    BIC_2 = chi2 + k2*np.log(n)

    # Calculating the Delta BIC value.
    dBIC = BIC_2 - BIC_1

    if verbose:
        print(f'BIC1 = {BIC_1:5.3f}')
        print(f'BIC2 = {BIC_2:5.3f}')
        print(f'dBIC = {dBIC:5.3f}')

    perr = None
    if dBIC > 0.0:
        params = params1
        if np.any(perr1):
            perr = perr1
    elif dBIC < 0.0:
        params = params2
        if np.any(perr2):
            perr = perr2
    else:
        err_msg = 'Error calculating dBIC, cannot perform model selection.'
        raise ValueError(err_msg)
    
    if np.any(perr):
        return params, perr
    else:
        return params

def calc_covMatrix(xx,yy,rms,psfParams):
    """
    This function estimates the covariance matrix assuming the data is spatially
    correlated with a 2D Gaussian function. 

    Parameters:
    ----------
    xx : numpy array
        Numpy array containing the x-positions for the data, should have 
        dimension 1.
    yy : numpy array 
        Numpy array containing the y-positions for the data, should have 
        dimension 1.
    rms : float
        Standard deviation of the data. Calculated from the background image.
    psfParams : numpy array
        List, tuple or numpy array containing the PSF params (sigx,sigy,pa). pa
        is in radians.
            
    Returns:
    ----------
    covMat : numpy array
        Covariance matrix (2D numpy array).
    """

    # Creating the col and row matrices for calculating the sx and dy offsets.
    #xxcol,xxrow = np.meshgrid(xx+1,xx+1)
    #yycol,yyrow = np.meshgrid(yy+1,yy+1)
    xxcol,xxrow = np.meshgrid(xx,xx)
    yycol,yyrow = np.meshgrid(yy,yy)

    # Getting the PSF parameters.
    sigxpsf = psfParams[0]
    sigypsf = psfParams[1]
    PA = psfParams[2]

    # Offset matrices for each position vector. Diagnonals are 0.
    dxMat = xxcol-xxrow
    dyMat = yycol-yyrow

    # Having issues with singular matrices.
    gaussMat = (Gaussian2D((dxMat,dyMat),1,0,0,sigxpsf,sigypsf,PA))**2
    covMat = (rms**2)*gaussMat

    return covMat