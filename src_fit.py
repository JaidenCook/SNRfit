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

##TODO: Refactor input output functions.
def J2000_name(RA,DEC,verbose=False):
    """
    Function that converts the source RA and DEC into a JRA+-DEC name format.

    Parameters:
    ----------
    RA : numpy array
        Numpy array containing the RA of each source.
    DEC : numpy array 
        Numpy array containing the DEC of each source. 
    verbose : bool, default=False
        If True print the formatted J2000 names compared to the J2000 coords.
            
    Returns:
    ----------
    J2000_names : numpy char.array
        Contains the new names for each source.

    """
    def check_string(string):
        """
        Simple function that checks the length of an input string, if it is 
        equal to 1, it adds a zero at the start of the string.
        
        Parameters:
        ----------
        string : str
            String.
        
        Returns:
        ----------
        string : str
            String.
        """

        nchar = len(string)
        if nchar == 1:
            string = '0' + string

        return string

    from astropy.coordinates import SkyCoord

    # Get the SkyCoord object for each RA and DEC.
    coords = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree)

    # Getting the hms and dms strings.
    JstringRA = coords.ra.hms
    JstringDEC = coords.dec.dms

    #
    # Splitting the right ascension component into three parts.
    JRA_h = np.char.array(JstringRA[0].astype(int)).decode('UTF-8')
    JRA_m = np.char.array(JstringRA[1].astype(int)).decode('UTF-8')
    JRA_s = np.char.array(np.rint(JstringRA[2]).astype(int)).decode('UTF-8')

    # Splitting the declination component into three parts.
    JDEC_d = np.char.array(JstringDEC[0].astype(int)).decode('UTF-8')
    JDEC_m = np.char.array(np.abs(JstringDEC[1].astype(int))).decode('UTF-8')
    JDEC_s = np.char.array(np.abs(np.rint(JstringDEC[2]).astype(int))).decode('UTF-8')

    J2000_names = np.chararray(RA.size,itemsize=14,unicode=True)
    for i in range(RA.size):
        rah = check_string(JRA_h[i])
        ram = check_string(JRA_m[i])
        ras = check_string(JRA_s[i])
        decd = check_string(JDEC_d[i])
        decm = check_string(JDEC_m[i])
        decs = check_string(JDEC_s[i])

        # Creating new string.
        J2000_name = 'J'+rah+ram+ras+decd+decm+decs

        J2000_names[i] = J2000_name

        if verbose:
            # For comparision check the names are formatted to the actual 
            # coordinates.
            print(J2000_name,coords.to_string('hmsdms')[i])

    return J2000_names


def write_model_table(popt,perr,constants,w,ID,
                      alpha=None,deconv=False,outname=None):
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
    ID : int 
        Integer ID for the input SNR.
    w : astropy object
        Astropy world coordinate system.
    alpha : float, default=None
        Spectral index of the model. Assumed to be constant for all components.
    deconv : bool, default=False
        If True calculate the deconvolution parameters for the fitted Gaussian 
        models.
    outname : str
        Output filename. Default is None, if given writes an astropy fits file.
  
    Returns:
    ----------
    t : astropy object
        Astropy table containing the model position and fit data. 

    """
    
    from astropy.table import QTable,Table
    from gaus_decv import deconv2,deconv_gauss

    # Defining the conversion factor. Calculating the integrated flux density.
    a_psf = constants[3] # [deg]
    b_psf = constants[4] # [deg]
    dx = constants[2] # pixel size in degrees [deg]

    # Calculating the pixel and beam solid angles.
    Omega_pix = dx**2 #[deg^2]
    Omega_beam = np.pi*a_psf*b_psf/(4*np.log(2)) #[deg^2]
    dOmega = Omega_pix/Omega_beam

    # ensuring type.
    popt = popt.astype('float')
    perr = perr.astype('float')

    # Initialising the column names:
    # Column names.
    col_names = ['ID','Name','RA','u_RA','DEC','u_DEC','Sint','u_Sint',
                 'Maj','u_Maj','Min','u_Min','PA','u_PA']

    if len(np.shape(popt)) > 1:

        # Creating columns.
        # Setting the centroid X and Y pixel values.
        X_pos = popt[:,1] # [pix]
        u_X = perr[:,1] # [pix]

        Y_pos = popt[:,2] # [pix]
        u_Y = perr[:,2] # [pix]

        # Getting the RA and DEC information from the WCS.
        # Pixels offset by 1.
        RA, DEC = w.wcs_pix2world(X_pos + 1,Y_pos + 1, 1)

        # Name column, SNID and the component number.
        Names = J2000_name(RA,DEC)

        u_RA = RA*(u_X/X_pos)*u.degree # FK5 [deg]
        RA = RA*u.degree # FK5 [deg]

        #
        u_DEC = np.abs(DEC*u_Y/Y_pos)*u.degree # FK5 [deg]
        DEC = DEC*u.degree # FK5 [deg]

        # Model ID associates model components with one source.
        ModelID = (np.ones(len(Names))*ID).astype('int')

        ## Calculating the integrated flux density.
        Sint = popt[:,0]*(2*np.pi*popt[:,3]*popt[:,4])*dOmega*u.Jy # [Jy]
        
        # Need to recalculate.
        u_Sint = Sint*(perr[:,0]/popt[:,0]) # [Jy]
        Sint = Sint

        # Getting the Major and Minor axes.
        Maj = sig2FWHM(popt[:,3])*(dx*60)*u.arcmin # [arcmins]
        u_Maj = Maj*(perr[:,3]/popt[:,3]) # [arcmins]
        Min = sig2FWHM(popt[:,4])*(dx*60)*u.arcmin # [arcmins]
        u_Min = Min*(perr[:,4]/popt[:,4]) # [arcmins]

        # Rotate the position angle so it matches the cosmological ref frame.
        PA = (270 - np.degrees(popt[:,5]))*u.deg # [deg]
        u_PA = (perr[:,5])*u.deg # [deg]

        # Constructing the table list structure. Rounding for formatting.
        proto_table = [ModelID,Names,np.round(RA,5),np.round(u_RA,7), 
                       np.round(DEC,5),np.round(u_DEC,7),np.round(Sint,3), 
                       np.round(u_Sint,5),np.round(Maj,4),np.round(u_Maj,5),
                       np.round(Min,4),np.round(u_Min,5),
                       np.round(PA,3),np.round(u_PA,5)]

        if deconv:
            # If deconvolution option is true, calculate the deconvolved params.
            sigxpsf_pix = FWHM2sig(a_psf)/dx
            sigypsf_pix = FWHM2sig(b_psf)/dx
            BPA = constants[-1]
            theta_PA = 360 - (BPA + 90)
            Maj_DC,Min_DC,PA_DC = deconv_gauss((sigxpsf_pix,sigypsf_pix,theta_PA),
                                               (popt[:,3],popt[:,4],np.degrees(popt[:,5]))) 
            
            # Converting the DC Major and Minor axes.
            Maj_DC = sig2FWHM(Maj_DC)*(dx*60)*u.arcmin # [arcmins]
            Min_DC = sig2FWHM(Min_DC)*(dx*60)*u.arcmin # [arcmins]
            # Rotate the position angle so it matches the cosmological ref frame.
            PA_DC = (270 - np.degrees(PA_DC))*u.deg # [deg]

            # Appending to the proto_table.
            proto_table.append(np.round(Maj_DC,3))
            proto_table.append(np.round(Min_DC,3))
            proto_table.append(np.round(PA_DC,3))
            # Add the PSF params so you can check the deconvolution.
            proto_table.append(np.round(np.ones(RA.size)*a_psf*60,3)*u.arcmin)
            proto_table.append(np.round(np.ones(RA.size)*b_psf*60,3)*u.arcmin)
            proto_table.append(np.round(np.ones(RA.size)*BPA,3)*u.deg)
            col_names.append('Maj_DC')
            col_names.append('Min_DC')
            col_names.append('PA_DC')
            col_names.append('Maj_PSF')
            col_names.append('Min_PSF')
            col_names.append('PA_PSF')

        if np.any(alpha):
            proto_table.append(alpha)
            col_names.append('alpha')    

    else:
        # Creating columns.
        # Setting the centroid X and Y pixel values.
        X_pos = (popt[1]) # [pix]
        u_X = perr[1] # [pix]

        Y_pos = (popt[2]) # [pix]
        u_Y = perr[2] # [pix]

        # Getting the RA and DEC information from the WCS.
        RA, DEC = w.wcs_pix2world(X_pos + 1,Y_pos + 1, 1)

        # Name column, SNID and the component number.
        Names = J2000_name(RA,DEC)

        RA = RA*u.degree # FK5 [deg]
        u_RA = RA*(u_X/X_pos) # FK5 [deg]

        DEC = DEC*u.degree # FK5 [deg]
        u_DEC = np.abs(DEC*(u_Y/Y_pos)) # FK5 [deg]

        # Model ID associates model components with one source.
        ModelID = int(ID)

        ## Calculating the integrated flux density.
        Sint = popt[0]*(2*np.pi*popt[3]*popt[4])*dOmega*u.Jy # [Jy]

        # Need to recalculate.
        u_Sint = Sint*(perr[0]/popt[0]) # [Jy]

        Maj = popt[3]*(2*np.sqrt(2*np.log(2)))*(dx*60)*u.arcmin # [arcmins]
        u_Maj = Maj*(perr[3]/popt[3]) # [arcmins]

        Min = popt[4]*(2*np.sqrt(2*np.log(2)))*(dx*60)*u.arcmin # [arcmins]
        u_Min = Min*(perr[4]/popt[4]) # [arcmins]

        # Rotate the position angle so it matches the cosmological ref frame.
        #PA = np.degrees(popt[5])*u.deg # [deg]
        PA = (270 - np.degrees(popt[5]))*u.deg # [deg]
        u_PA = (perr[5])*u.deg # [deg]

        # Constructing the table list structure.
        proto_table = [[ModelID],[Names],[np.round(RA,3)],[np.round(u_RA,5)], 
                       [np.round(DEC,3)],[np.round(u_DEC,4)],[np.round(Sint,3)], 
                       [np.round(u_Sint,5)],[np.round(Maj,3)],[np.round(u_Maj,5)],
                       [np.round(Min,3)],[np.round(u_Min,5)],
                       [np.round(PA,3)],[np.round(u_PA,5)]]

        if deconv:
            # If deconvolution option is true, calculate the deconvolved params.
            sigxpsf_pix = FWHM2sig(a_psf)/dx
            sigypsf_pix = FWHM2sig(b_psf)/dx
            BPA = constants[-1]
            theta_PA = 360 - (BPA + 90)

            Maj_DC,Min_DC,PA_DC = deconv_gauss((sigxpsf_pix,sigypsf_pix,theta_PA),
                                               (popt[3],popt[4],np.degrees(popt[5])))
            
            # Converting the DC Major and Minor axes.
            Maj_DC = sig2FWHM(Maj_DC)*(dx*60)*u.arcmin # [arcmins]
            Min_DC = sig2FWHM(Min_DC)*(dx*60)*u.arcmin # [arcmins]
            # Rotate the position angle so it matches the cosmological ref frame.
            #PA_DC = np.degrees(PA_DC)*u.deg # [deg]
            PA_DC = (270 - np.degrees(PA_DC))*u.deg # [deg]

            # Appending to the proto_table.
            proto_table.append([np.round(Maj_DC,3)])
            proto_table.append([np.round(Min_DC,3)])
            proto_table.append([np.round(PA_DC,3)])
            # Add the PSF params so you can check the deconvolution.
            proto_table.append([np.round(a_psf*60,3)*u.arcmin])
            proto_table.append([np.round(b_psf*60,3)*u.arcmin])
            proto_table.append([np.round(theta_PA,3)*u.deg])
            col_names.append('Maj_DC')
            col_names.append('Min_DC')
            col_names.append('PA_DC')
            col_names.append('Maj_PSF')
            col_names.append('Min_PSF')
            col_names.append('PA_PSF')

        if np.any(alpha):
            proto_table.append([alpha])
            col_names.append('alpha')    

    t = QTable(proto_table,names=col_names,meta={'name':'first table'})
    #t = Table(proto_table,names=col_names,meta={'name':'first table'})

    if outname:
        # Condition for writing the file.
        # Default returns table.
        t.write(outname,overwrite=True)

    return t


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

def deg_2_pixel(w,header,RA,DEC,Maj=None,Min=None,pixoffset=1):
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
    pixoffset : float, default=1
        If pyBDSF grid set to 1, if SNRFIT set 0.
            
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

    x_vec, y_vec = w.wcs_world2pix(RA,DEC,pixoffset)

    if np.any(Maj) and np.any(Min):
        Maj_pix = Maj/dx
        Min_pix = Min/dx

        return x_vec,y_vec,Maj_pix,Min_pix
    else:
        return x_vec,y_vec

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