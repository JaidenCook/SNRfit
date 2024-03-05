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
    I'm tired of doing this manually here is a function
    to convert from FWHM to sigma. Units don't matter.
    """ 

    return FWHM/(2*np.sqrt(2*np.log(2)))

def sig2FWHM(sig):
    """
    I'm tired of doing this manually here is a function
    to convert from FWHM to sigma. Units don't matter.
    """ 

    return sig*(2*np.sqrt(2*np.log(2)))

def majmin_swap(majVec,minVec,paVec,degrees=True):
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
    swapVec = majVec < minVec

    if np.any(swapVec):
        # Creating temp vectors from copied numpy arrays.
        majVec_temp = np.copy(majVec)
        minVec_temp = np.copy(minVec)
        paVec_temp = np.copy(paVec)
        
        # Swapping axes.
        majVec_temp[swapVec] = minVec[swapVec]
        minVec_temp[swapVec] = majVec[swapVec]

        # Performing the rotation.
        if degrees:
            # Default 
            paVec_temp[swapVec] += 90 
        else:
            paVec_temp[swapVec] += np.pi/2

        # Assigning the temp vectors to input maj and min vecs.
        majVec = majVec_temp
        minVec = minVec_temp
        paVec = paVec_temp
    
    return majVec,minVec,paVec

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
        2D Gaussian position angle.
            
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


# // This can be generalised to fit other functions shapelets etc.
# // We can also generalise the fitting approach, MCMC etc.
def Gaussian_2Dfit(xx,yy,data,pguess,
                func=NGaussian2D,pbound_low=None,pbound_up=None,
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
    pbound_low : numpy array, default=None
        Lower bound on the parameter ranges.
    pbound_up : numpy array, defaul=None
        
            
    Returns:
    ----------
    popt : numpy array
        2D numpy array containing the best fit parameters.
    pcov : numpy array
        2D numpy array containing the covariance matrix for the fit parameters.
    """
    if np.any(pbound_low) and np.any(pbound_up):
        # If bounds supplied.
        popt, pcov = opt.curve_fit(func,(xx,yy),data.ravel(),p0=pguess.ravel(),
                    bounds=(pbound_low.ravel(),pbound_up.ravel()),maxfev=maxfev)
    else:
        # If no bounds supplied.
        popt, pcov = opt.curve_fit(func,(xx,yy),data.ravel(),p0=pguess.ravel(),
                    maxfev=maxfev)
    
    popt = popt.reshape(np.shape(pguess)) # Fit parameters.
    perr = np.array([np.sqrt(pcov[i,i]) \
                     for i in range(len(pcov))]).reshape(np.shape(pguess))

    return popt,perr

# // This function needs to be refactored. 
def SNR_Gauss_fit(xx,yy,data,coords,constants,maj_frac=0.125,
                  allow_negative=False,bounds=True):
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
            
    Returns:
    ----------
    popt : numpy array
        2D numpy array containing the best fit parameters.
    pcov : numpy array
        2D numpy array containing the covariance matrix for the fit parameters.
    """

    # Major and Minor axis in arcminutes of the SNR.
    major = constants[0]
    minor = constants[1]
    dx = constants[2] #pixel size in degrees [deg]

    # Restoring beam parameters. Used in the guess.
    pixel_scale = dx*3600 # [arcsec] 
    a_psf = constants[3]*3600 # [arcsec]
    b_psf = constants[4]*3600 # [arsec]
    PA_psf = np.abs(np.radians(constants[5])) # [rads]

    # Restoring beam sizes in pixels:
    sig_x_psf = FWHM2sig(a_psf/pixel_scale)
    sig_y_psf = FWHM2sig(b_psf/pixel_scale)

    print(sig_x_psf,sig_y_psf)

    # PSF initial parameters. Used to determine pguess.
    p0 = np.array([1,0,0,sig_x_psf,sig_y_psf,PA_psf])

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

            # Set values less than 0 to be 1.
            coords[:,3][coords[:,3] < 0] = 0
            pguess[:,0] = coords[:,3]
        else:
            pass
        pguess[:,3][coords[:,2]>sig_x_psf] = coords[:,2][coords[:,2]>sig_x_psf]
        pguess[:,4][coords[:,2]>sig_y_psf] = coords[:,2][coords[:,2]>sig_y_psf]

    # Specifying the lower fit bounds for each Gaussian.
    if allow_negative:
        # If true allow fitting negative peaks.
        # Template 1D array.
        pbound_low = np.array([-np.inf,x_low,y_low,sig_x_psf,sig_y_psf,0.0]) 
    else:
        pbound_low = np.array([0.0,x_low,y_low,sig_x_psf,sig_y_psf,0.0]) 

    # Expanding for each Gaussian component.
    pbound_low = np.ones((N_gauss,N_params))*pbound_low[None,:] 
    pbound_up = np.array([np.inf,x_hi,y_hi,Max_major,Max_major,np.pi])
    pbound_up = np.ones((N_gauss,N_params))*pbound_up[None,:]

    if bounds:
        # Getting the fit parameters, and their errors.
        popt, perr = Gaussian_2Dfit(xx,yy,data,pguess,
                func=NGaussian2D,pbound_low=pbound_low,pbound_up=pbound_up)
    else:
        popt, perr = Gaussian_2Dfit(xx,yy,data,pguess,
                func=NGaussian2D)

    return popt,perr


# // This function needs to be refactored. 
def fit_psf(xx,yy,data,coords,psf_params,
            bounds=True,peak_fit=False):
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
    popt,perr = Gaussian_2Dfit(xx,yy,data,pguess,
                               func=NGaussian2D,pbound_low=pbound_low,pbound_up=pbound_up)

    if N_gauss == 1:
        # Doesn't need to be 2D if only a single Gaussian.
        popt = popt.flatten()
        perr = perr.flatten()

    return popt,perr



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


def write_model_table(popt,perr,constants,alpha,SNR_ID,w,outname=None):
    """
    Converts model Gaussian fit parameters and errors to Astropy tabel object. 
    Add option for saving the table.

    Parameters:
    ----------
    popt : numpy array
        Numpy array containing the Gaussian fit parameters, should have two 
        axes.
    perr : numpy array
        Numpy array containing the Gaussian fit parameter errors, should have 
        two axes.
    header : astropy object
        Astropy image header object. Contains information about the image. 
    alpha : float
        Spectral index of the model. Assumed to be constant for all components.
    SNR_ID : int 
        Integer ID for the input SNR.
    w : astropy object
        Astropy world coordinate system.
    outname : str
        Output filename. Default is None, if given writes an astropy fits file.

    
    Returns:
    ----------
    t : astropy object
        Astropy table containing the model position and fit data. 

    """
    
    from astropy.table import QTable,Table

    # Defining the conversion factor. Calculating the integrated flux density.
    a_psf = constants[3] # [deg]
    b_psf = constants[4] # [deg]
    dx = constants[2] #pixel size in degrees [deg]

    # Calculating the pixel and beam solid angles.
    Omega_pix = dx**2 #[deg^2]
    Omega_beam = np.pi*a_psf*b_psf/(4*np.log(2)) #[deg^2]
    dOmega = Omega_pix/Omega_beam

    # ensuring type.
    popt = popt.astype('float')
    perr = perr.astype('float')

    if len(np.shape(popt)) > 1:

        # Creating columns.
        # Name column, SNID and the component number.
        Names = np.array(['SN{0}-{1}'.format(SNR_ID,i) \
                          for i in range(len(popt))]).astype('str')
        # Model ID is the same as the SNR_ID. Might change this at a later date. 
        #B Useful for now.
        ModelID = (np.ones(len(Names))*SNR_ID).astype('int')

        ## Calculating the integrated flux density.
        Sint = popt[:,0]*(2*np.pi*popt[:,3]*popt[:,4])*dOmega*u.Jy # [Jy]

        # Calculating the spectral coefficients. For modelling source spectrum.
        alpha = -1*np.ones(len(Sint))*np.abs(alpha)

        # Need to recalculate.
        u_Sint = Sint*(perr[:,0]/popt[:,0]) # [Jy]

        # Setting the centroid X and Y pixel values.
        X_pos = (popt[:,1]) # [pix]
        #u_X = X_pos*(perr[:,1]/popt[:,1]) # [pix]
        u_X = perr[:,1] # [pix]

        Y_pos = (popt[:,2]) # [pix]
        #u_Y = Y_pos*(perr[:,2]/popt[:,2]) # [pix]
        u_Y = perr[:,2] # [pix]

        # Getting the RA and DEC information from the WCS.
        # Pixels offset by 1.
        RA, DEC = w.wcs_pix2world(popt[:,1] + 1,popt[:,2] + 1, 1)

        RA = RA*u.degree # FK5 [deg]
        u_RA = RA*(u_X/X_pos) # FK5 [deg]

        DEC = DEC*u.degree # FK5 [deg]
        u_DEC = np.abs(DEC*(u_Y/Y_pos)) # FK5 [deg]

        Maj = popt[:,3]*(2*np.sqrt(2*np.log(2)))*(dx*60)*u.arcmin # [arcmins]
        u_Maj = Maj*(perr[:,3]/popt[:,3]) # [arcmins]

        Min = popt[:,4]*(2*np.sqrt(2*np.log(2)))*(dx*60)*u.arcmin # [arcmins]
        u_Min = Min*(perr[:,4]/popt[:,4]) # [arcmins]

        PA = np.degrees(popt[:,5])*u.deg # [deg]
        u_PA = (perr[:,5])*u.deg # [deg]

        # Constructing the table list structure.
        proto_table = [Names, RA, u_RA, DEC, u_DEC, Sint, u_Sint, Maj, u_Maj, 
        Min, u_Min, PA, u_PA, alpha, ModelID]

    else:
        
        # Creating columns.
        # Name column, SNID and the component number.
        Names = 'SN{0}'.format(SNR_ID)
        # Model ID is the same as the SNR_ID. Might change this at a later date. 
        # Useful for now.
        ModelID = int(SNR_ID)

        ## Calculating the integrated flux density.
        Sint = popt[0]*(2*np.pi*popt[3]*popt[4])*dOmega*u.Jy # [Jy]

        # Need to recalculate.
        u_Sint = Sint*(perr[0]/popt[0]) # [Jy]

        # Setting the centroid X and Y pixel values.
        X_pos = (popt[1]) # [pix]
        u_X = perr[1] # [pix]

        Y_pos = (popt[2]) # [pix]
        u_Y = perr[2] # [pix]

        # Getting the RA and DEC information from the WCS.
        RA, DEC = w.wcs_pix2world(popt[1] + 1,popt[2] + 1, 1)

        RA = RA*u.degree # FK5 [deg]
        u_RA = RA*(u_X/X_pos) # FK5 [deg]

        DEC = DEC*u.degree # FK5 [deg]
        u_DEC = np.abs(DEC*(u_Y/Y_pos)) # FK5 [deg]

        Maj = popt[3]*(2*np.sqrt(2*np.log(2)))*(dx*60)*u.arcmin # [arcmins]
        u_Maj = Maj*(perr[3]/popt[3]) # [arcmins]

        Min = popt[4]*(2*np.sqrt(2*np.log(2)))*(dx*60)*u.arcmin # [arcmins]
        u_Min = Min*(perr[4]/popt[4]) # [arcmins]

        PA = np.degrees(popt[5])*u.deg # [deg]
        u_PA = (perr[5])*u.deg # [deg]

        alpha = -1*np.abs(alpha)

        # Constructing the table list structure.
        proto_table = [[Names], [RA], [u_RA], [DEC], [u_DEC], 
                        [Sint], [u_Sint], [Maj], [u_Maj], [Min], [u_Min], 
                        [PA], [u_PA], [alpha], [ModelID]]

    # Column names.
    col_names = ['Name','RA','u_RA','DEC','u_DEC','Sint','u_Sint',
               'Maj','u_Maj','Min','u_Min','PA','u_PA','alpha','ModelID']

    t = QTable(proto_table,names=col_names,meta={'name':'first table'})
    #t = Table(proto_table,names=col_names,meta={'name':'first table'})

    #print(t)

    if outname:
        # Condition for writing the file.
        # Default returns table.
        t.write(outname,overwrite=True)
    else:
        pass

    return t



def model_select(params1,params2,perr1,perr2,xx,yy,data,rms):
    """
    Takes input parameters from two Gaussian fit models. Calculates the BIC for 
    each model and rejects the model with the higher BIC. This is usually the 
    single Gaussian fit model.
    
    Parameters:
    ----------
    params1 : numpy array
        1D/2D numpy array containing the model1 fit params for each Gaussian.
    perr1 : numpy array
        1D/2D numpy array containing the error model1 fit params for each 
        Gaussian.
    params2 : numpy array
        1D/2D numpy array containing the model2 fit params for each Gaussian.
    perr2 : numpy array
        1D/2D numpy array containing the error model2 fit parameters for each 
        Gaussian.
    xx : numpy array
        Array of x-values.
    yy : numpy array
        Array of y-values.
    data : numpy array
        Data for which to compare the models to.
    rms : numpy array
        Uncertainty in the data. Assumed to be constant.
            
    Returns:
    ----------
    params : numpy array
        Selected parameter array.
    perr : numpy array
        Selected parameter error array.
    """

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

    if dBIC > 0.0:
        params = params1
        perr = perr1
    elif dBIC < 0.0:
        params = params2
        perr = perr2
    else:
        err_msg = 'Error calculating dBIC, cannot perform model selection.'
        raise ValueError(err_msg)
    
    return params, perr


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
    solid_angle
    """

    return np.pi*major*minor/(4*np.log(2))

def deg_2_pixel(w,header,RA,DEC,Maj=None,Min=None):
    """
    Calcuates the pixel coordinates for an input wcs and RA and DEC array. 
    Converts, the major and minor axis values of Gaussian fits to pixel values.

    Maj and Min are now optional parameters.

    Parameters:
    ----------
    w : astropy object
        Astropy world coordinate system object.
    header : astropy object
        Astropy FITS header image object.
    RA : numpy array
        Array of RA positions.
    DEC : numpy array
        Array of DEC positions.
    Maj : numpy array
        Array of Major axis sizes (in degrees).
    Min : numpy array
        Array of Minor axis sizes (in degrees).
            
    Returns:
    ----------
    x_vec : numpy array
        SNR x pixel coordinate values.
    y_vec : numpy array
        SNR y pixel coordinate values.
    Maj_pix : float, optional
        SNR Major axis pixel size.
    Min_pix : float, optional
        SNR Minor axis pixel size.
    """

    try:
        # Checking if the CDELTA attribute exists.
        dx = header['CD2_2'] #pixel scale in degrees [deg]
    except KeyError:
        try:
            dx = np.abs(header['CDELT2']) #pixel scale in degrees [deg]
        except KeyError:
            err_msg='Header "CDELT2" and "CD2_2" do not exist. ' + \
                    'Check FITS header.'
            raise KeyError(err_msg)

    x_vec, y_vec = w.wcs_world2pix(RA,DEC, 1)

    if np.any(Maj) and np.any(Min):
        Maj_pix = Maj/dx
        Min_pix = Min/dx

        return x_vec,y_vec,Maj_pix,Min_pix
    else:
        return x_vec,y_vec