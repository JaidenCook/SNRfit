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
__email__ = "Jaiden.Cook@student.curtin.edu"

# Generic stuff:
import os,sys
import time
import warnings

import logging
logging.captureWarnings(True) # Cleans up some of the annoying derecation warnings. 
#warnings.filterwarnings('always', category=DeprecationWarning,
#                        module=r'^{0}\.'.format(re.escape(__name__)))

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

# Parser options:
from optparse import OptionParser

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


def great_circle_dist(lat1,lat2,lon1,lon2,degrees=False):
    """
    Calculate the distance between two points on a spherical surface.
    This function is generalised to calculate the distance between a 
    fixed point with lat1 and lon1, and an array of points designated
    lat2 and lon2.
    
    Parameters:
    ----------
    lat1 : float
        Latitude value of reference point.
    lat2 : float/numpy array
        Latitude value or array of points.
    lon1 : float
        Longitude value of reference point.
    lon2 : float/numpy array
        Longitude value or array of points.
    
    Returns:
    ----------
    theta_dist : float/numpy array
        float or vector of distance values relative to 
        lat1 and lon1.
    """

    if degrees:
        # Condition if the values are in degrees.
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)
        lon1 = np.radians(lon1)
        lon2 = np.radians(lon2)
    else:
        pass

    # Haversine formula.
    hav1 = np.sin((lat2-lat1)/2)**2
    hav2 = np.sin((lon2-lon1)/2)**2
    
    theta_dist = 2*np.sqrt(hav1 + np.cos(lat1)*np.cos(lat2)*hav2)

    if degrees:
        # If true return the angle distance array in units of degrees.
        theta_dist = np.degrees(theta_dist)
    else:
        pass
    
    return theta_dist

## Plotting functions.
def point_plot(zz,img_nu,resid_img,wcs,vmax=None,vmin=None,filename=None,
                scale=1,dpi=70):
    """
    Plots the data image, the model image and the residual image for comparison.
    
    Parameters:
    ----------
    zz : numpy array
        Numpy array containing the model image.
    image : numpy array
        Numpy array containing the image. Background subtracted.
    image : numpy array
        Numpy array containing the residual image.
    wcs : astropy object
        Object containing the world coordinate system for the map projection.
    vmax : float, default=None
        Max colourbar value.
    vmin : float, default=None
        Min colourbar value.
    filename : str, default=None
        If not None, write png with provided filename.
    scale : float, default=1
        Scale size of the figure.
            
    Returns:
    ----------
    None
    """

    # Adds limits to the colourbar. Default is none.
    # Limits only added if the user inputs their own limits.
    extend='neither'
    if (vmax != None) and (vmin != None):
        extend = 'both'
    else:
        if vmax != None:
            extend = 'max'
        else:
            vmax = np.nanmax(img_nu)*0.6

        if vmin != None:
            extend = 'min'
        else:
            vmin = np.nanmin(img_nu)        

    # Creating figure object.
    fig = plt.figure(figsize=(scale*15, scale*4))

    # Defining the width of each figure.
    epsilon = 0.02
    width = 1/3 - epsilon

    # Adding axes objects to the figure. 
    ax1 = fig.add_axes([0,0,width,1],projection=wcs.celestial, 
            slices=('x','y'))
    ax2 = fig.add_axes([0.33,0,width,1],projection=wcs.celestial, 
            slices=('x','y'),sharey=ax1)
    ax3 = fig.add_axes([0.66,0,width,1],projection=wcs.celestial, 
            slices=('x','y'),sharey=ax1)

    # sharey isn't working, so set the y-lables to be false.
    ax2.tick_params('y',labeltop=False)
    ax3.tick_params('y',labeltop=False)

    # Creating image objects for each axis. 
    im1 = ax1.imshow(img_nu, cmap='cividis', vmin=vmin, vmax=vmax, aspect='auto')
    im2 = ax2.imshow(zz, cmap='cividis', vmin=vmin, vmax=vmax, aspect='auto')
    im3 = ax3.imshow(resid_img, cmap='cividis', vmin=vmin, vmax=vmax, aspect='auto')

    # Adding titles to the subfigures. 
    ax1.set_title('Data')
    ax2.set_title('Model')
    ax3.set_title('Residuals')

    # Adding colourbars, no label added, but that can be fixed.
    cb1 = fig.colorbar(im1, ax=ax1, pad =0.002, extend=extend)
    cb2 = fig.colorbar(im2, ax=ax2, pad =0.002, extend=extend)
    cb3 = fig.colorbar(im3, ax=ax3, pad =0.002, extend=extend)

    # Adding labels to the x and y-axis. 
    ax1.set_xlabel(r'RAJ2000',fontsize=14)
    ax2.set_xlabel(r'RAJ2000',fontsize=14)
    ax3.set_xlabel(r'RAJ2000',fontsize=14)
    ax1.set_ylabel(r'DEJ2000',fontsize=14)
    ax1.tick_params(axis='both', color = 'k', which='major', labelsize=12)

    # Setting tick labels size for the colourbar.
    cb1.ax.tick_params(labelsize=12)
    cb2.ax.tick_params(labelsize=12)
    cb3.ax.tick_params(labelsize=12)

    fig.tight_layout()

    if filename:
        #plt.savefig('{0}.png'.format(filename),dpi=dpi,
        #    overwrite=True,bbox_inches='tight')
        plt.savefig('{0}.png'.format(filename),dpi=dpi,bbox_inches='tight')
    else:
        plt.show()

    plt.close()

def array_plot(img_list,wcs_list,scale=1,filename=None):
    """
    Makes Nx3 plots. Images should be ordered appropriately. N is the number
    of rows. WCS coordinate should be the same for each image in a row. 
    In future will generalise plot. 

    Parameters:
    ----------
    img_list : list, numpy array
        List containing images. Each image is a 2D numpy array. Image list
        should be order from last 3 image set to first 3 image set.
    wcs_list : list, astropy object 
        Single wcs coordinate, or list of wcs coordinates. List length
        should match the number of rows. wcs ordering should match the image
        set.
    filename : str, default=None
        If not None, write png with provided filename.
    scale : float, default=1
        Scale size of the figure.
            
    Returns:
    ----------
    None
    """

    # Grid is populated from bottom to top, sort list so it is ordered
    # properly.
    figx = 12
    figy = 4
    fontsize=22
    labelsize=20

    # Number of images in the image list.
    Nimags = len(img_list)

    # Number of columns.
    Ncols = 3

    # Defining the width of each figure.
    epsilonx = 0.02
    epsilony = 0.055

    # Checking that the number of images is divisible by 3. If not then raise
    # a ValueError. Image grid should be Nx3. Might generalise this in future.
    if (Nimags % Ncols) != 0.:
        raise ValueError('Number of images should be divisible by 3.')
    else:
        Nrows = int(Nimags/Ncols)
    
    # If the input wcs list doesn't match the image number then raise a ValueError.
    # If the input wcs list is not a list then ignore this. All images might share
    # the same wcs. 
    if type(wcs_list) == list:
        if (len(wcs_list) % Nrows) != 0:
            err_str = (f"Number of wcs coordinate systems {len(wcs_list)}"\
                f", doesn't match the image array number {Nrows}.")
            raise ValueError(err_str)
    
    # In the case that all images share the same wcs, create a list
    # with the same number of elements as Nimags, but with each element
    # being the input wcs.
    wcs_identical_cond = False
    if type(wcs_list) != list:
        wcs_list = [wcs_list for i in range(Nrows)]
        wcs_identical_cond = True
        # Defining the width of each figure.
        figx = 13
        epsilonx = 0.02
        epsilony = 0.02

    # Adds limits to the colourbar. Default is none.
    # Limits only added if the user inputs their own limits.
    extend='max'    
    vmax_list = []
    vmin_list = []
    for i in range(Nrows):
        max_temp = np.max(np.array([np.nanmax(img_list[i*Ncols+0]),
                np.nanmax(img_list[i*Ncols+1]),np.nanmax(img_list[i*Ncols+2])]))

        vmax_list.append(max_temp*0.8)

        min_temp = np.min(np.array([np.nanmin(img_list[i*Ncols+0]),
                np.nanmin(img_list[i*Ncols+1]),np.nanmin(img_list[i*Ncols+2])]))

        vmin_list.append(min_temp)


    # Creating figure object.
    fig = plt.figure(figsize=(scale*figx, scale*figy*Nrows))

    dx = 1/Ncols
    dy = 1/Nrows
    # Sub image widths.
    width = 1/Ncols - epsilonx
    # Sub image heights.
    height = 1/Nrows - epsilony

    counter = 0
    for i in range(Nrows):
        for j in range(Ncols):

            # Getting the wcs coord system.
            wcs = wcs_list[i]
            img = img_list[counter]
            vmin = vmin_list[i]
            vmax = vmax_list[i]

            # x and y positions of the bottom left corner of axes in the figure.
            xpos = j*dx
            ypos = i*dy

            # Setting the temp axis dimensions. 
            axes_dimensions = [xpos,ypos,width,height]
            
            # Adding axes objects to the figure. 
            axs_temp = fig.add_axes(axes_dimensions,projection=wcs.celestial, 
                    slices=('x','y'))

            if j==0:
                axs_temp.set_ylabel(r'DEJ2000',fontsize=fontsize*scale)
            else:
                # sharey isn't working, so set the y-lables to be false.
                axs_temp.tick_params('y',labeltop=False)

            # If using the same wcs for each image, set all axes to false,
            # except for the bottom row.
            if i != 0 and wcs_identical_cond:
                axs_temp.tick_params('x',labeltop=False)
            else:
                pass

            if i == 0:
                axs_temp.set_xlabel(r'RAJ2000',fontsize=fontsize*scale)
            else:
                axs_temp.set_xlabel(' ',fontsize=fontsize*scale)

            axs_temp.tick_params(axis='both', color = 'k', which='major', labelsize=labelsize*scale)

            im_temp = axs_temp.imshow(img, cmap='cividis', vmin=vmin, vmax=vmax, aspect='auto')

            cb_temp = fig.colorbar(im_temp, ax=axs_temp, pad =0.002, extend=extend)        
            cb_temp.ax.tick_params(labelsize=labelsize*scale)

            if j == (Ncols-1):
                cb_temp.set_label(r'Intensity $\rm{[Jy/Beam]}$',fontsize=fontsize*scale)
            else:
                cb_temp.ax.tick_params(labelright=False)
                

            counter += 1

    fig.tight_layout()

    if filename:
        plt.savefig('{0}.png'.format(filename),bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def astro_plot_2D(image, wcs, figsize=(10,10), scatter_points=None, lognorm=False, 
                    clab=None, vmin=None, vmax=None, filename=None, cmap='cividis',
                    scale=1, point_area=1, abs_cond=False, ellipes=None):
    """
    2D Astro image plotter. Takes an input image array and world coordinate system
    and plots the image. 

    TODO: Add a condition for the colourmap scale. support for log, power, sqrt methods.
    
    Parameters:
    ----------
    image : numpy array
        Numpy array containing the image.
    wcs : astropy object
        Object containing the world coordinate system for the map projection.
    figsize : tuple, default=(10,10)
        Tuple which determines the size of the image. 
    scatter_points : numpy array, default=None
        2D array of peak points, [x,y].
    lognorm : bool, default=False
        If True colourscale is logscale.
    clab : str, default=None
        If given, colourbar label.
    vmin : float, default=None
        Min colour bar scale.
    vmax : float, default=None
        Max colour bar scale.
    filename : str, defualt=None
        If given, output png with filename.
    cmap : str, default='cividis'
        Colourmap colour scheme.
    scale : float, default=1
        Figure size scale.
    point_area : float, default=1
        Area of the scatter point pixel.
    abs_cond : bool, default=False
        Condition for plotting the absolute values.
    ellipes : numpy array, float, default=None
        If given, plot ellipses, uses the same parameter format as the output for
        SNR_Gauss_fit.
            
    Returns:
    ----------
    None
    """
    
    # If array issues, use wcs.celestial
    if lognorm:
        from matplotlib.colors import LogNorm
        norm = LogNorm()
    else:
        norm=None

    if scale != 1:
        # If scale is not default, rescale the figure size.
        (figx,figy) = figsize
        figx *= scale
        figy *= scale

        figsize = (figx,figy)

    fig = plt.figure(figsize=figsize)

    # Create the axis object.
    ax = plt.subplot(projection=wcs.celestial, slices=('x','y'))
    
    # Adds limits to the colourbar. Default is none.
    # Limits only added if the user inputs their own limits.
    extend='neither'
    if vmax != None:
        extend = 'max'
    else:
        pass

    if vmin != None:
        extend = 'min'
    else:
        pass
    
    if vmax != None and vmin != None:
        extend = 'both'

    if abs_cond:
        im = ax.imshow(np.abs(image), origin='lower', cmap=cmap,norm=norm, 
                    vmin=vmin, vmax=vmax, aspect='auto')
    else:
        im = ax.imshow(image, origin='lower', cmap=cmap,norm=norm, 
                       vmin=vmin, vmax=vmax, aspect='auto')
    
    cb = fig.colorbar(im, ax=ax, pad =0.002, extend=extend)
    
    if clab:
        cb.set_label(label = clab, fontsize=20*scale, color='k')
    else:
        cb.set_label(label = 'Jy/beam', fontsize=20*scale)
    
    if np.any(scatter_points):
        # Plot peak detection peaks if coordinates provided.
        ax.scatter(scatter_points[:,1],scatter_points[:,0],
                   color='r',s=point_area)
    else:
        pass

    ax.set_xlabel(r'RAJ2000',fontsize=20*scale)
    ax.set_ylabel(r'DEJ2000',fontsize=20*scale)
    ax.tick_params(axis='both', color = 'k', which='major', labelsize=18*scale)
    cb.ax.tick_params(labelsize=18*scale)

    plt.grid()

    if np.any(ellipes):
        # If given overlay ellipses. This is good for comparing model parameters
        # to the fitted image. 
        from matplotlib.patches import Ellipse

        # Conversion factor. 
        FWHM = 2*np.sqrt(2*np.log(2))

        for i in range(ellipes.shape[0]):
            # Creating ellipse object.
            etemp = Ellipse((ellipes[i,1],ellipes[i,2]),
                            FWHM*ellipes[i,3],FWHM*ellipes[i,4],
                            np.degrees(ellipes[i,5])+90,fc='none',
                            edgecolor='r',lw=1.5)
            
            # Adding to axis.
            ax.add_artist(etemp)

    if filename:
        #plt.savefig('{0}.png'.format(filename),overwrite=True,bbox_inches='tight')
        plt.savefig('{0}.png'.format(filename),bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def hist_residual_plot(res_data,res_data2=None,N_peaks=None,figsize=(8,7),bins=40,alpha=0.35,
                filename=None,label1=None,label2=None,min_val=None,max_val=None,scale=1,**kwargs):
    """
    Plots a histogram of the multi-component residuals and the single Gaussian fit residuals.

    Parameters:
    ----------
    res_data : numpy array
        Numpy array containing the residual data for the multi-component Gaussian fit.
    res_data2 : numpy array, default=None
        Numpy array containing the residual data for the single Gaussian fit.
    N_peaks : int, default=None
        Number of peaks, if not None adds this to plot legend.
    figsize : tuple, default=(8,7)
        Default (8,7) determines the dimensions of the image.
    bins : int, default=40
        Number of histogram bins.
    alpha : float, default=0.35
        Histogram transparency number ranging from 0 to 1.
    filename : str, default=None
        If not None, outputs figure to filename.
    label1 : str, default=None
        res_data label. If not None adds to legend.
    label2 : str, default=None
        res_data2 label. If not None adds to legend.
    min_val : float, default=None
        Min value for plot range. 
    max_val : float, default=None
        Max value for plot range.
    scale : float, default=1
        Figure size scale.
            
    Returns:
    ----------
    None
    """
    if max_val == None:
        max_val = np.max(res_data)
    else:
        pass

    if min_val == None:
        min_val = np.min(res_data)
    else:
        pass

    res_data = res_data.flatten()

    if scale != 1:
        # If scale is not default, rescale the figure size.
        (figx,figy) = figsize
        figx *= scale
        figy *= scale

        figsize = (figx,figy)

    fig,axs = plt.subplots(1,figsize=figsize)

    if label1 == None:
        label1 = '{0} Gaussian Fit'.format(N_peaks)
    else:
        pass

    axs.hist(res_data,bins=bins,label=label1,histtype='stepfilled',color='g',
                edgecolor='g',alpha=alpha,lw=4)

    if np.any(res_data2):
        res_data2 = res_data2.flatten()

        if label2 == None:
            label2 = 'Gaussian Fit'
        else:
            pass
        axs.hist(res_data2,bins=bins,label=label2,histtype='stepfilled',alpha=alpha,lw=2,
                density=True)
    else:
        pass
    
    axs.set_xlabel('Residuals', fontsize=22*scale)
    axs.set_ylabel('Density', fontsize=22*scale)
    axs.set_xlim([min_val,max_val])

    plt.legend(fontsize=18*scale)

    if filename:
        #plt.savefig(filename,overwrite=True,bbox_inches='tight')
        plt.savefig(filename,bbox_inches='tight')
    else:
        plt.show()
    plt.close()

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
    Generalised 2DGaussian function, used primarily for fitting Gaussians to islands.
    Also used to create N-component Gaussian model images. 
    
    Parameters:
    ----------
    xdata_Tuple : tuple
        Tuple containing the X-data and Y-data arrays.
    params : numpy array
        Array of the Gaussian parameters, should be 6*N_Guassians. One dimension.
            
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

def calc_img_bkg_rms(image,mask_arr=None,Niter=5,sigma_thresh=2.5,mask_cond=False,
                     plot_cond=False,abscond=False):
    """
    Calculates a constant background and rms for an input image. Can accept input mask
    images.
    
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
            thresh_mask = np.logical_or((image > threshold_pve),(image < threshold_nve))
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

def generate_correlated_noise(std,psf_params,img_dims,
                               verbose=False,threshold=1e-1,w=None,
                               return_cond=False):
    """
    For a given input standard deviation (calculated from some image noise),
    input image psf parameters, and image dimensions, generate spatially correlated
    noise.

    The psf here is assumed to be Gaussian, therefore noise is correlated on spatial
    scales related to the Gaussian size. This is an accurate first order approximation
    of interferometric noise.

    Parameters:
    ----------
    std : float
        Standard deviation of image noise.
    psf_params : list
        List of psf major, minor and position angle. Major and minor axes should be
        units of pixels.
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
    img_psf = NGaussian2D((xx_pix,yy_pix),ppsf,fit=False).reshape(img_dims[0],img_dims[1])

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
    img_correlated_noise = convolve2d(img_psf,img_scaled,mode='same',boundary='wrap')

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
        numpy array containing the coordinates (x,y), sigma and amp for each peak.
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
    # Getting the location of the peaks in x,y as well as the expected blob size. 
    coordinates_sigma_vec = blob_log(image_nu,min_sigma=min_sigma,max_sigma=max_sigma,
        log_scale=log_cond,threshold=rms*thresh_fac,num_sigma=num_sigma,overlap=overlap)

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

        # Guessing the Gaussian amplitude as the pixel value closest to the peak coordinates.
        coord_sigma_amp_vec[:,3] = image_nu[(coordinates_sigma_vec[:,0]-1).astype(int),
                                (coordinates_sigma_vec[:,1]-1).astype(int)]

        return coord_sigma_amp_vec

# // This can be generalised to fit other functions shapelets etc.
# // We can also generalise the fitting approach, MCMC etc.
def Gaussian_2Dfit(xx,yy,data,pguess,
                func=NGaussian2D,pbound_low=None,pbound_up=None,maxfev=10000000):
    """
    Wrapper function for the Gaussian_2Dfit function, which fits the NGaussian2D function
    using scipy.optimise.curve_fit(), which uses a non-linear least squares method.
    In future this function should be refactored, especially if we want to consider using
    different fitting methods such as a Bayesian approach.    

    Parameters:
    ----------
    xx : numpy array
        Numpy array containing the x-positions for the data, should have dimension 1.
    yy : numpy array 
        Numpy array containing the y-positions for the data, should have dimension 1.
    data : numpy array
        Numpy array containing the image data, should have dimensions 1.
    pguess : numpy array
        Numpy array containing the initial parameters.
    func : function, default=NGaussian2D()
        Function which the parameters is supplied to. This function is fit to the data.
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
        popt, pcov = opt.curve_fit(func, (xx, yy), data.ravel(), p0 = pguess.ravel(),
                    bounds = (pbound_low.ravel(),pbound_up.ravel()), maxfev = maxfev)
    else:
        # If no bounds supplied.
        popt, pcov = opt.curve_fit(func, (xx, yy), data.ravel(), p0 = pguess.ravel(),
                    maxfev = maxfev)
    
    popt = popt.reshape(np.shape(pguess)) # Fit parameters.
    perr = np.array([np.sqrt(pcov[i,i]) for i in range(len(pcov))]).reshape(np.shape(pguess))

    return popt,perr

# // This function needs to be refactored. 
def SNR_Gauss_fit(xx,yy,data,coordinates,constants,maj_frac=0.125,
                  allow_negative=False,bounds=True):
    """
    Wrapper function for the Gaussian_2Dfit function, which fits the NGaussian2D function
    using scipy.optimise.curve_fit(), which uses a non-linear least squares method.
    In future this function should be refactored, especially if we want to consider using
    different fitting methods such as a Bayesian approach.    

    Parameters:
    ----------
    xx : numpy array
        Numpy array containing the x-positions for the data, should have dimension 1.
    yy : numpy array 
        Numpy array containing the y-positions for the data, should have dimension 1.
    data : numpy array
        Numpy array containing the image data, should have dimensions 1.
    coordinates : numpy array
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
    sig_x_psf = (a_psf/pixel_scale)/(2*np.sqrt(2*np.log(2)))
    sig_y_psf = (b_psf/pixel_scale)/(2*np.sqrt(2*np.log(2)))

    print(sig_x_psf,sig_y_psf)

    # PSF initial parameters. Used to determine pguess.
    p0 = np.array([1,0,0,sig_x_psf,sig_y_psf,PA_psf])# Template inital parameter array.

    # Parameter array dimensions.
    N_gauss = len(coordinates)
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
    pguess[:,1] = coordinates[:,1]
    pguess[:,2] = coordinates[:,0]

    if coordinates.shape[1] > 2:
        # The blob fitting method returns an expected sigma for a given peak.
        # We can use this as a guess of the actual sigma.
        
        if coordinates.shape[1] > 3:
            # peak intensity guess for the Gaussians.

            # Set values less than 0 to be 1.
            coordinates[:,3][coordinates[:,3] < 0] = 0
            pguess[:,0] = coordinates[:,3]
        else:
            pass
        pguess[:,3][coordinates[:,2] > sig_x_psf] = coordinates[:,2][coordinates[:,2] > sig_x_psf]
        pguess[:,4][coordinates[:,2] > sig_y_psf] = coordinates[:,2][coordinates[:,2] > sig_y_psf]

    # Specifying the lower fit bounds for each Gaussian.
    if allow_negative:
        # If true allow fitting negative peaks.
        # Template 1D array.
        pbound_low = np.array([-np.inf,x_low,y_low,sig_x_psf,sig_y_psf,0.0]) 
    else:
        pbound_low = np.array([0.0,x_low,y_low,sig_x_psf,sig_y_psf,0.0]) 

    pbound_low = np.ones((N_gauss,N_params))*pbound_low[None,:] # Expanding for each Gaussian component.

    pbound_up = np.array([np.inf,x_hi,y_hi,Max_major,Max_major,np.pi])

    #pbound_up = np.array([np.sum(data),x_hi,y_hi,Max_major,Max_major,np.pi])
    pbound_up = np.ones((N_gauss,N_params))*pbound_up[None,:]

    if bounds:
        # Getting the fit parameters, and their errors.
        popt, perr = Gaussian_2Dfit(xx,yy,data,pguess,
                func=NGaussian2D,pbound_low=pbound_low,pbound_up=pbound_up)
    else:
        popt, perr = Gaussian_2Dfit(xx,yy,data,pguess,
                func=NGaussian2D)

    return popt,perr

def Fit_quality(data,p_mod,xx,yy,rms,reduced_cond=False):
    """
    Returns the Chi-squared value for an input model. Model input given by 
    the parameter array p_mod. The input data should have the background subtracted.
    
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
    # Calculating the reduce Chi-squared, add option to return this instead of chisquared.
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
        Numpy array containing the Gaussian fit parameters, should have two axes.
    perr : numpy array
        Numpy array containing the Gaussian fit parameter errors, should have two axes.
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
        Names = np.array(['SN{0}-{1}'.format(SNR_ID,i) for i in range(len(popt))]).astype('str')
        # Model ID is the same as the SNR_ID. Might change this at a later date. Useful for now.
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
        RA, DEC = w.wcs_pix2world(popt[:,1] + 1,popt[:,2] + 1, 1)# Pixels offset by 1.

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
        # Model ID is the same as the SNR_ID. Might change this at a later date. Useful for now.
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


def convolve_image(image,header,Gauss_size):
    """
    Simple image convolver. Takes input image, header, and new Gaussian size in degrees. 

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

def model_select(params1,params2,perr1,perr2,xx,yy,data,rms):
    """
    Takes input parameters from two Gaussian fit models. Calculates the BIC for each model and
    rejects the model with the higher BIC. This is usually the single Gaussian fit model.
    
    Parameters:
    ----------
    params1 : numpy array
        1D/2D numpy array containing the model1 fit parameters for each Gaussian.
    perr1 : numpy array
        1D/2D numpy array containing the error model1 fit parameters for each Gaussian.
    params2 : numpy array
        1D/2D numpy array containing the model2 fit parameters for each Gaussian.
    perr2 : numpy array
        1D/2D numpy array containing the error model2 fit parameters for each Gaussian.
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
        raise ValueError('Error calculating dBIC, cannot perform model selection.')
    
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

def deg_2_pixel(w,header,RA,DEC,Maj,Min):
    """
    Calcuates the pixel coordinates for an input wcs and RA and DEC array. Converts,
    the major and minor axis values of Gaussian fits to pixel values.

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
    Maj_pix : float
        SNR Major axis pixel size.
    Min_pix : float
        SNR Minor axis pixel size.
    """

    try:
        # Checking if the CDELTA attribute exists.
        dx = header['CD2_2'] #pixel scale in degrees [deg]
    except KeyError:
        try:
            dx = np.abs(header['CDELT2']) #pixel scale in degrees [deg]
        except KeyError:
            raise KeyError('Header "CDELT2" and "CD2_2" do not exist. Check FITS header.')

    x_vec, y_vec = w.wcs_world2pix(RA,DEC, 1)

    Maj_pix = Maj/dx
    Min_pix = Min/dx

    return x_vec,y_vec,Maj_pix,Min_pix

if __name__ == '__main__':

    #import gleam_client
    import logging
    warnings.filterwarnings("ignore", category=DeprecationWarning)


    # Initialising and setting logger parameters.
    #log_file_name = 'snrfit.log'

    #logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    #    datefmt='%d-%m-%Y:%H:%M:%S',
    #    level=logging.INFO,
    #    filename=log_file_name, filemode='w', force=True)

    #logger = logging.getLogger('snrfit')

    start0 = time.perf_counter()



    end0 = time.perf_counter()

    #logger.info('Runtime = %7.3f [s]' % (end0-start0))
    print('Runtime = %7.3f [s]' % (end0-start0))

else:
    pass