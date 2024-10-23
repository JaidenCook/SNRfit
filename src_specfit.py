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

from functions import power_law,matern_cov,jac_power_law,curved_power_law
from bayes import initial_samples,loglikelihood,logposterior,plotposts,calc_logZ

def spec_fit(freqs,fluxVec,func=power_law,sigma=None,
             covmatrix=False,perrcond=True,verbose=False,maxfev=int(1e5),
             **kwargs):
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
        
    if func == power_law:
        p0 = [1,-0.7]
        jac = jac_power_law
        bounds_low = np.array([0,-5])
        bounds_hi = np.array([np.inf,5])
        bounds = (bounds_low,bounds_hi)
    elif func == curved_power_law:
        p0 = [1,-0.7,-0.17]
        jac = None
        bounds_low = np.array([0,-5,-5])
        bounds_hi = np.array([np.inf,5,5])
        bounds = (bounds_low,bounds_hi)
    

    popt,pcov = opt.curve_fit(func,freqs,fluxVec,
                              p0=p0,maxfev=maxfev,sigma=sigma,
                              jac=jac,bounds=bounds,**kwargs)
    
    if perrcond:
        pcov = np.sqrt(np.diag(pcov))

    return popt,pcov


def bayes_spec_fit(freqs,flux,paramsDict,model=power_law,sigma=None,
                   Nburnin=500,Nsamples=1250,Nens=100,p0cond=False,
                   corner=False,verbose=False):
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
    argslist = (flux,sigma,paramsDict,freqs,model)

    # Set up the sampler.
    sampler = emcee.EnsembleSampler(Nens,ndims,logposterior,args=argslist)

    # Pass the initial samples and total number of samples required
    sampler.run_mcmc(inisamples,Nsamples+Nburnin)

    # Extract the samples (removing the burn-in)
    samples_emcee = sampler.get_chain(flat=True,discard=Nburnin)

    # Store the results.
    #popt = np.nanmean(samples_emcee,axis=0)
    popt = np.nanmedian(samples_emcee,axis=0)
    #perr = np.nanstd(samples_emcee,axis=0)
    perr = np.nanquantile(samples_emcee,[0.25,0.75],axis=0)
    perr = (perr[1,:] - perr[0,:])/1.35

    pcov = np.cov(samples_emcee.T)
    if verbose:
        print('+-=-=-=-=-=-=-=-=-=-=-=-=-=-+')
        print(samples_emcee.shape)
        print(pcov.shape)
        print(perr**2)
        print(pcov)

    if corner:
        # If True plot the samples.
        plotposts(samples_emcee,paramsDict=paramsDict,popt=popt)

    return popt,perr

def calc_spec_BIC(pDict,data,freqs,rmsVec):
    """
    Calculates the Bayesian information criterion for the spectral models.

    Parameters:
    ----------
    pDict1 : numpy array
        1D/2D numpy array containing the model1 fit params for each Gaussian.
    pDict2 : numpy array
        1D/2D numpy array containing the model2 fit params for each Gaussian.
    data : numpy array
        Data for which to compare the models to.
    freqs : numpy array
        Contains the frequency values for the models.
    rmsVec : float, default=None
        Uncertainty in the data. Assumed to be constant.
    
    Returns:
    ----------
    BIC : float
        Bayesian Information Criterion.
    """

    pVec = pDict["params"]

    if pDict["model"] == "power_law":
        model = power_law
    elif pDict["model"] == "curved_power_law":
        model = curved_power_law

    chisqd = np.nansum(((data - model(freqs,*pVec))/(rmsVec))**2)

    k = np.size(pVec) # model 1

    # Number of data points.
    n = np.size(data)

    # Calculating the BIC for both models.
    BIC = chisqd + k*np.log(n)

    return BIC

def spec_model_select(pDict1,pDict2,freqs,data,rmsVec,verbose=False):
    """
    Takes input parameters from two Gaussian fit models. Calculates the BIC for 
    each model and rejects the model with the higher BIC. This is usually the 
    single Gaussian fit model.
    
    Parameters:
    ----------
    pDict1 : numpy array
        1D/2D numpy array containing the model1 fit params for each Gaussian.
    pDict2 : numpy array
        1D/2D numpy array containing the model2 fit params for each Gaussian.
    data : numpy array
        Data for which to compare the models to.
    freqs : numpy array
        Contains the frequency values for the models.
    rmsVec : float, default=None
        Uncertainty in the data. Assumed to be constant.
    verbose : bool, default=False
        If True print additional information.
            
    Returns:
    ----------
    params : numpy array
        Selected parameter array.
    """


    # Calculating the BIC for both models.
    BIC1 = calc_spec_BIC(pDict1,data,freqs,rmsVec)
    BIC2 = calc_spec_BIC(pDict2,data,freqs,rmsVec)

    # Calculating the Delta BIC value.
    dBIC = BIC2 - BIC1

    if verbose:
        print(f'BIC1 = {BIC1:5.3f}')
        print(f'BIC2 = {BIC2:5.3f}')
        print('+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=+')
        print(f'dBIC = {dBIC:5.3f}')
        print('+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=+')

    pVec1 = pDict1["params"]
    pVec2 = pDict2["params"]
    e_pVec1 = pDict1["e_params"]
    e_pVec2 = pDict2["e_params"]

    pVec = pVec1
    e_pVec = e_pVec1
    model = pDict1["model"]

    if dBIC < -6:
        pVec = pVec2
        e_pVec = e_pVec2
        model = pDict2["model"]
    
    params = {"params" : pVec, "e_params" : e_pVec, 
              "model" : model}
    
    return params
