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
    freq0 = freqs[0]
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

    popt,pcov = opt.curve_fit(func,freqs/freq0,fluxVec,
                              p0=[1,-0.7],maxfev=int(1e5),sigma=sigma,
                              jac=jac_power_law,bounds=bounds)
    
    if perrcond:
        pcov = np.sqrt(np.diag(pcov))

    return popt,pcov

def logposterior(theta, data, sigma, x):
    """
    The natural logarithm of the joint posterior.

    Parameters:
    ----------
    theta : tuple
        A sample containing individual parameter values.
    data : numpy ndarray
        The set of data/observations.
    sigma : float, or numpy ndarray
        The standard deviation of the data points.
    x : numpy ndarray
        The abscissa values at which the data/model is defined.
    
    Returns:
    ----------
    """

    lp = logprior(theta) # get the prior

    # if the prior is not finite return a probability of zero (log probability of -inf)
    if not np.isfinite(lp):
        return -np.inf

    # return the likeihood times the prior (log likelihood plus the log prior)
    return lp + loglikelihood(theta, data, sigma, x)

def loglikelihood(theta, data, sigma, freqs):
    """
    The natural logarithm of the joint Gaussian likelihood.

    Parameters:
    ----------
    theta : tuple
        A sample containing individual parameter values.
    data : numpy ndarray
        The set of data/observations.
    sigma : float, or numpy ndarray
        The standard deviation of the data points.
    x : numpy ndarray
        The abscissa values at which the data/model is defined.

    Note:
        We do not include the normalisation constants (as discussed above).
    
    Returns:
    ----------
    """

    # unpack the model parameters from the tuple
    S0, alpha = theta

    # evaluate the model (assumes that the straight_line model is defined as above)
    md = power_law(freqs/freqs[0], S0, alpha)

    if isinstance(sigma, np.ndarray):
        if len(sigma.shape) > 1 and (sigma.shape[0]==sigma.shape[1]):
            # If the covariance matrix is given.
            resid = (md - data)
            Kinv = np.linalg.inv(sigma)
            #print(resid.shape,np.matmul(resid,sigma).shape)
            return -0.5*np.matmul(np.matmul(resid,Kinv),resid.T)
        else:
            return -0.5 * np.sum(((md - data) / sigma)**2)
    else:
        return -0.5 * np.sum(((md - data) / sigma)**2)


def logprior(theta,uniform=False):
    """
    The natural logarithm of the prior probability.

    Parameters:
    ----------
    theta : tuple
        A sample containing individual parameter values.
    uniform : bool, default=False

    Note:
        We can ignore the normalisations of the prior here.
    
    Returns:
    ----------
    """

    lp = 0.

    # unpack the model parameters from the tuple
    S0, alpha = theta

    if uniform:
        S0min = 0. # lower range of prior
        S0max = 10.  # upper range of prior
        
        #
        lp = 0. if S0min < S0 < S0max else -np.inf
        
        alpha_min = -3
        alpha_max = 0
        
        #
        lp -= 0. if alpha_min < alpha < alpha_max else -np.inf
    else:
        S0mu = 0.115
        S0sig = 0.164
        logS0mu = np.log(S0mu**2/np.sqrt(S0mu**2 + S0sig**2))
        logS0sig = np.log(1 + (S0sig/S0mu)**2)
        lp = lp - (0.5*((np.log(S0)-logS0mu)/logS0sig)**2)

        # Gaussian prior on m
        alpha_mu = -0.75 # mean of the Gaussian prior
        alpha_sigma = 0.3 # standard deviation of the Gaussian prior
        
        #
        lp = lp - 0.5*((alpha-alpha_mu)/alpha_sigma)**2
    

    return lp

def bayes_spec_fit(freqs,flux,sigma=None):
    """
    
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

    Nens = 100   # number of ensemble points

    # Gaussian prior on m
    alpha_mu = -0.75     # mean of the Gaussian prior
    alpha_sigma = np.sqrt(2) # standard deviation of the Gaussian prior
    #alpha_sigma = 0.6 # standard deviation of the Gaussian prior
    #alpha_min = -3
    #alpha_max = 0

    alpha_ini = np.random.normal(alpha_mu, alpha_sigma, Nens) # initial m points
    #alpah_ini = np.random.uniform(alpha_min, alpha_max, Nens) # initial m points

    S0min = 0.  # lower range of prior
    S0max = 10.   # upper range of prior
    S0mu = 0.1
    S0sig = 1/np.sqrt(2)
    #logS0mu = np.log(S0mu**2/np.sqrt(S0mu**2 + S0sig**2))
    #logS0sig = np.log(1 + (S0sig/S0mu)**2)

    S0ini = np.random.uniform(S0min,S0max,Nens) # initial c points
    #S0ini = np.random.lognormal(logS0mu,logS0sig,Nens)

    inisamples = np.array([S0ini,alpha_ini]).T # initial samples

    ndims = inisamples.shape[1] # number of parameters/dimensions

    Nburnin = 500   # number of burn-in samples
    Nsamples = 1250  # number of final posterior samples

    # for bookkeeping set number of likelihood calls to zero
    loglikelihood.ncalls = 0

    # set additional args for the posterior (the data, the noise std. dev., and the abscissa)
    #argslist = (Sdata1, stdVec, freqs)
    argslist = (flux, sigma, freqs)

    # set up the sampler
    sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)

    # pass the initial samples and total number of samples required
    sampler.run_mcmc(inisamples, Nsamples + Nburnin)

    # extract the samples (removing the burn-in)
    samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)

    # lets store some results for showing together later
    popt = np.array([np.round(np.mean(samples_emcee[:,0]),5),
                     np.round(np.mean(samples_emcee[:,1]),5)])
    perr = np.array([np.round(np.std(samples_emcee[:,0]),5),
                     np.round(np.std(samples_emcee[:,1]),5)])

    #
    return popt,perr