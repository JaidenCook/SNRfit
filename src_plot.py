#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Python module for plotting Gaussian fits to FITS format images. 

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
logging.captureWarnings(True) # Cleans up the annoying derecation warnings. 

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
    im1 = ax1.imshow(img_nu,cmap='cividis',vmin=vmin,vmax=vmax,aspect='auto')
    im2 = ax2.imshow(zz,cmap='cividis',vmin=vmin,vmax=vmax,aspect='auto')
    im3 = ax3.imshow(resid_img,cmap='cividis',vmin=vmin,vmax=vmax,aspect='auto')

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
    
    # If the input wcs list doesn't match the image number then raise a 
    # ValueError. If the input wcs list is not a list then ignore this. 
    # All images might share the same wcs. 
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

            axs_temp.tick_params(axis='both',color='k',which='major', 
                                 labelsize=labelsize*scale)

            im_temp = axs_temp.imshow(img,cmap='cividis',vmin=vmin,vmax=vmax,
                                      aspect='auto')

            cb_temp = fig.colorbar(im_temp,ax=axs_temp,pad=0.002,extend=extend)        
            cb_temp.ax.tick_params(labelsize=labelsize*scale)

            if j == (Ncols-1):
                cb_temp.set_label(r'Intensity $\rm{[Jy/Beam]}$',
                                  fontsize=fontsize*scale)
            else:
                cb_temp.ax.tick_params(labelright=False)
                

            counter += 1

    fig.tight_layout()

    if filename:
        plt.savefig('{0}.png'.format(filename),bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def astro_plot_2D(image,wcs,figsize=(10,10),scatter_points=None,lognorm=False,
                  clab=None,vmin=None,vmax=None,filename=None,cmap='cividis',
                  scale=1,point_area=1,abs_cond=False,ellipes=None):
    """
    2D Astro image plotter. Takes an input image array and world coordinate 
    system and plots the image. 

    TODO: Add a condition for the colourmap scale. support for log, power, sqrt 
    methods.
    
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
        If given, plot ellipses, uses the same parameter format as the output 
        for SNR_Gauss_fit.
            
    Returns:
    ----------
    None
    """
    if vmax:
        vmax=vmax
    else:
        vmax = np.nanmax(image)*0.8
    
    if vmin:
        vmin=vmin
    else:
        vmin = np.nanmin(image)*1.2



    # If array issues, use wcs.celestial
    import matplotlib

    if lognorm:
        if vmin < 0:
            # If less than zero make asihn symmetric lognorm colorbar.
            norm = matplotlib.colors.AsinhNorm(vmin=vmin,vmax=vmax,
                                           linear_width=0.1)
        else:
            norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)

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
        im = ax.imshow(np.abs(image),origin='lower',cmap=cmap,norm=norm,
                       aspect='auto')
    else:
        im = ax.imshow(image,origin='lower',cmap=cmap,norm=norm,aspect='auto')
    
    cb = fig.colorbar(im, ax=ax, pad =0.002, extend=extend)
    
    if clab:
        cb.set_label(label = clab, fontsize=20*scale, color='k')
    else:
        cb.set_label(label = 'Jy/beam', fontsize=20*scale)
    
    if np.any(scatter_points):
        # TODO Fix the coordinates. Different for scatter points 
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

        if len(ellipes.shape) < 2:
            # This expects a shape (Nellipses,Nparams). If only one ellipse, w
            # wrap to fit.
            ellipes = np.array([ellipes])
        for i in range(ellipes.shape[0]):
            #
            etemp = Ellipse((ellipes[i,1],ellipes[i,2]),
                            FWHM*ellipes[i,3],FWHM*ellipes[i,4],
                            360-np.degrees(ellipes[i,5]),fc='none',
                            edgecolor='r',lw=1.5)
            
            # Adding to axis.
            ax.add_artist(etemp)
            ax.scatter(ellipes[i,1],ellipes[i,2],
                   color='r',s=point_area)

    if filename:
        #plt.savefig('{0}.png'.format(filename),overwrite=True,bbox_inches='tight')
        plt.savefig('{0}.png'.format(filename),bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def hist_residual_plot(res_data,res_data2=None,N_peaks=None,figsize=(8,7),
                       bins=40,alpha=0.35,filename=None,label1=None,label2=None,
                       min_val=None,max_val=None,scale=1,**kwargs):
    """
    Plots a histogram of the multi-component residuals and the single Gaussian 
    fit residuals.

    Parameters:
    ----------
    res_data : numpy array
        Numpy array containing the residual data for the multi-component 
        Gaussian fit.
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
                edgecolor='g',alpha=alpha,lw=4,density=True)

    if np.any(res_data2):
        res_data2 = res_data2.flatten()

        if label2 == None:
            label2 = 'Gaussian Fit'
        else:
            pass
        axs.hist(res_data2,bins=bins,label=label2,
                 histtype='stepfilled',alpha=alpha,lw=2,density=True)
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