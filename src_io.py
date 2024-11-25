#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Python module for reading and writing FITS tables/images for snrfit. 

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

# Astropy stuff:
from astropy import units as u
from astropy.io import fits

from functions import *

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

def write_model_table(popt,perr,psfParams,w,ID,Names=None,alpha=None,
                      deconv=False,outname=None,precision=6,pixoffset=1):
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
    psfParams : astropy object
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
    precision : int,
        Precision to which to write out the table values.
    pixoffset : int
        Pixel origin for WCS, either 1 or 0. Should be 0, if x and y coordinates
        taken from numpy array indices.
  
    Returns:
    ----------
    t : astropy object
        Astropy table containing the model position and fit data. 

    """
    
    from astropy.table import QTable,Table
    from gaus_decv import deconv2,deconv_gauss

    # Defining the conversion factor. Calculating the integrated flux density.
    dx = np.abs(w.wcs.cd[0,0]) # pixel size in degrees [deg]
    a_psf,b_psf,BPA = psfParams

    # Calculating the pixel and beam solid angles.
    omegaPSF = Beam_solid_angle(a_psf,b_psf)

    # ensuring type.
    popt = popt.astype('float')
    perr = perr.astype('float')

    # Initialising the column names:
    # Column names.
    col_names = ['ID','Name','RA','u_RA','DEC','u_DEC','Sint','u_Sint',
                 'Maj','u_Maj','Min','u_Min','PA','u_PA']

    if len(np.shape(popt)) == 1:
        # Setting popt to be of appropriate shape.
        popt = np.ones((1,popt.size))*popt[None,:]
        perr = np.ones((1,popt.size))*perr[None,:]

    # Creating columns.
    # Setting the centroid X and Y pixel values.
    X_pos = popt[:,1] + pixoffset # [pix]
    u_X = perr[:,1] # [pix]

    Y_pos = popt[:,2] + pixoffset# [pix]
    u_Y = perr[:,2] # [pix]

    # Getting the RA and DEC information from the WCS.
    # Pixels offset by 1.
    RA,DEC = w.wcs_pix2world(X_pos,Y_pos,pixoffset)

    # Name column, SNID and the component number.
    Names = J2000_name(RA,DEC)

    u_RA = RA*(u_X/X_pos)*u.degree # FK5 [deg]
    RA = RA*u.degree # FK5 [deg]

    #
    u_DEC = np.abs(DEC*u_Y/Y_pos)*u.degree # FK5 [deg]
    DEC = DEC*u.degree # FK5 [deg]

    # Model ID associates model components with one source.
    ModelID = (np.ones(len(Names))*ID).astype('int')

    # Getting the Major and Minor axes.
    Maj = sig2FWHM(popt[:,3])*(dx*60)*u.arcmin # [arcmins]
    u_Maj = Maj*(perr[:,3]/popt[:,3]) # [arcmins]
    Min = sig2FWHM(popt[:,4])*(dx*60)*u.arcmin # [arcmins]
    u_Min = Min*(perr[:,4]/popt[:,4]) # [arcmins]

    ## Calculating the integrated flux density.
    Sint,u_Sint = Sint_calc(popt[:,0],Maj.value/60,Min.value/60,omegaPSF,
                            e_Speak=perr[:,0],e_Maj=u_Min.value/60,
                            e_Min=u_Maj.value/60)
    Sint = Sint*u.Jy
    u_Sint = u_Sint*u.Jy

    # Rotate the position angle so it matches the cosmological ref frame.
    PA = (270 - np.degrees(popt[:,5]))*u.deg # [deg]
    u_PA = (perr[:,5])*u.deg # [deg]

    # Constructing the table list structure. Rounding for formatting.
    proto_table = [ModelID,Names,np.round(RA,precision),
                    np.round(u_RA,precision),np.round(DEC,precision),
                    np.round(u_DEC,precision),np.round(Sint,precision), 
                    np.round(u_Sint,precision),np.round(Maj,precision),
                    np.round(u_Maj,precision),np.round(Min,precision),
                    np.round(u_Min,precision),np.round(PA,precision),
                    np.round(u_PA,precision)]

    if deconv:
        # If deconvolution option is true, calculate the deconvolved params.
        sigxpsf_pix = FWHM2sig(a_psf)/dx
        sigypsf_pix = FWHM2sig(b_psf)/dx
        theta_PA = paRA2paXY(BPA)
        Maj_DC,Min_DC,PA_DC = deconv_gauss((sigxpsf_pix,sigypsf_pix,theta_PA),
                                            (popt[:,3],popt[:,4],np.degrees(popt[:,5]))) 
        
        # Converting the DC Major and Minor axes.
        Maj_DC = sig2FWHM(Maj_DC)*(dx*60)*u.arcmin # [arcmins]
        Min_DC = sig2FWHM(Min_DC)*(dx*60)*u.arcmin # [arcmins]
        # Rotate the position angle so it matches the cosmological ref frame.
        PA_DC = paXYpaRA2(np.degrees(PA_DC))*u.deg # [deg]

        # Appending to the proto_table.
        proto_table.append(np.round(Maj_DC,precision))
        proto_table.append(np.round(Min_DC,precision))
        proto_table.append(np.round(PA_DC,precision))
        # Add the PSF params so you can check the deconvolution.
        proto_table.append(np.round(np.ones(RA.size)*a_psf*60,precision)*u.arcmin)
        proto_table.append(np.round(np.ones(RA.size)*b_psf*60,precision)*u.arcmin)
        proto_table.append(np.round(np.ones(RA.size)*BPA,precision)*u.deg)
        col_names.append('Maj_DC')
        col_names.append('Min_DC')
        col_names.append('PA_DC')
        col_names.append('Maj_PSF')
        col_names.append('Min_PSF')
        col_names.append('PA_PSF')

    if np.any(alpha):
        proto_table.append(alpha)
        col_names.append('alpha')    

    t = QTable(proto_table,names=col_names,meta={'name':'first table'})

    if outname:
        # Condition for writing the file.
        # Default returns table.
        print(f'File saved {outname}')
        t.write(outname,overwrite=True)

    return t

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

    x_vec,y_vec = w.wcs_world2pix(RA,DEC,pixoffset)

    if np.any(Maj) and np.any(Min):
    #if np.any(Maj.size) and np.any(Min.size):
        Maj_pix = Maj/dx
        Min_pix = Min/dx

        return x_vec,y_vec,Maj_pix,Min_pix
    else:
        return x_vec,y_vec

def open_fits(filepath,return_hdul=False):
    """
    Wrapper function for opening a fits image file. Discerns between two 
    different types of fits images with different dimensions.

    Parameters:
    ----------
    filepath : str
        File and path location.
    return_hdul : bool, default=False
        If True return hdul.
 
            
    Returns:
    ----------
    header : astropy object
        Astropy FITS header image object.
    img_arr : numpy array
        Image array.
    hdul : astropy object, optional
        Astropy hdul object. Contains image and header.
    """

    with fits.open(filepath) as hdul:
        header = hdul[0].header

        # 2D image Arrays come in several sizes depending on the format.
        # Below are the most common.
        if len(hdul[0].data.shape) == 4:
            img_arr = hdul[0].data[0,0,:,:]
        elif len(hdul[0].data.shape) == 2:
            img_arr = hdul[0].data
        else:
            err_msg = f'hdul Shape no 2 or 4.'
            raise ValueError(err_msg)

        ### TODO maybe just return the HDU object for return_hdul.
        if return_hdul:
            # If True return the hdul object.
            return header,img_arr,hdul
        else:
            return header,img_arr
