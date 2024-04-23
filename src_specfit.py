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

import emcee

# Array stuff:
import numpy as np

# Scipy stuff:
import scipy.optimize as opt

from functions import *
from bayes import initial_samples,loglikelihood,logposterior,plotposts,calc_logZ

def spec_fit(freqs,fluxVec,func=power_law,sigma=None,bounds=True,
             covmatrix=False,perrcond=True,verbose=False):
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
    #freq0 = freqs[0]
    #freq0 = freqs[-1]
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
        if verbose:
            print('Covariance matrix calculated...')

    if bounds:
        # Sensible bounds set on the parameters.
        bounds_low = np.array([0,-5])
        bounds_hi = np.array([np.inf,5])
        bounds = (bounds_low,bounds_hi)
    else:
        bounds = None

    popt,pcov = opt.curve_fit(func,freqs,fluxVec,
                              p0=[1,-0.7],maxfev=int(1e5),sigma=sigma,
                              jac=jac_power_law,bounds=bounds)
    
    if perrcond:
        pcov = np.sqrt(np.diag(pcov))

    return popt,pcov


def bayes_spec_fit(freqs,flux,paramsDict,sigma=None,
                   Nburnin=500,Nsamples=1250,Nens=100,p0cond=False,
                   corner=False):
    """
    Fit the parameters using Bayesian inference.
    
    Parameters:
    ----------
    freqs : float, numpy ndarray
        Numpy vector of frequencies.
    fluxVec : float, numpy ndarray
        Numpy vector of flux densities in Jy.

    Note:
        We can ignore the normalisations of the prior here.
    
    Returns:
    ----------
    """
    # Get the initial samples.
    inisamples = initial_samples(paramsDict,Nens=Nens,p0cond=p0cond)
    # Number of parameters/dimensions
    ndims = inisamples.shape[1] 

    # For bookkeeping set number of likelihood calls to zero
    loglikelihood.ncalls = 0

    # Set additional args for the posterior 
    # (the data, the noise std. dev., and the abscissa).
    #argslist = (flux,sigma,paramsDict,freqs)
    argslist = (flux,sigma,paramsDict,freqs,power_law)

    # Set up the sampler.
    sampler = emcee.EnsembleSampler(Nens,ndims,logposterior,args=argslist)

    # Pass the initial samples and total number of samples required
    sampler.run_mcmc(inisamples, Nsamples + Nburnin)

    # Extract the samples (removing the burn-in)
    samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)

    # Store the results.
    #popt = np.nanmean(samples_emcee,axis=0)
    popt = np.nanmedian(samples_emcee,axis=0)
    #perr = np.nanstd(samples_emcee,axis=0)
    perr = np.nanquantile(samples_emcee,[0.25,0.5],axis=0)
    perr = perr[1,:] - perr[0,:]
    print(perr)
    #print(perr[1,0] - perr[0,0])
    #print(perr[1,1] - perr[0,1])

    if corner:
        # If True plot the samples.
        plotposts(samples_emcee,paramsDict=paramsDict,popt=popt)

    #
    #logZ=calc_logZ(sampler,Nburnin=Nburnin)-np.log(2*np.pi*np.linalg.det(sigma))
    norm = 2*np.pi*np.linalg.det(sigma)
    logZ=calc_logZ(sampler,Nburnin=Nburnin,norm=norm)
    print(f'logZ={logZ}')
    
    return popt,perr

