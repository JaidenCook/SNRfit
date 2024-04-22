#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Module for Bayesian fitting functions.

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

def uniformPrior(val,hypParams):
    """
    Returns the natural log of the uniform prior without normalisation.

    Parameters:
    ----------
    val : float
        A sample containing individual parameter values.
    hypParams : tuple, or list, or np.ndarray
        A sample containing individual parameter values.

    Returns:
    ----------
    logPrior : float
        Natural log of the uniform prior.
    """

    low = hypParams[0]
    high = hypParams[1]

    logPrior = 0 if low < val < high else -np.inf

    return logPrior

def normalPrior(val,hypParams):
    """
    Returns the natural log of the normal prior without normalisation.

    Parameters:
    ----------
    val : float
        A sample containing individual parameter values.
    hypParams : tuple, or list, or np.ndarray
        A sample containing individual parameter values.

    Returns:
    ----------
    logPrior : float
        Natural log of the uniform prior.
    """

    mu = hypParams[0]
    sig = hypParams[1]

    logPrior = -0.5*((val-mu)/sig)**2

    return logPrior

def lognormalPrior(val,hypParams):
    """
    Returns the natural log of the normal prior without normalisation.

    Parameters:
    ----------
    val : float
        A sample containing individual parameter values.
    p0Dict : tuple, or list, or np.ndarray
        A sample containing individual parameter values.

    Returns:
    ----------
    logPrior : float
        Natural log of the uniform prior.
    """

    mu = hypParams[0]
    sig = hypParams[1]

    # Log normal distribution satistics.
    logMu = np.log(mu**2/np.sqrt(mu**2 + sig**2))
    logSig = np.log(1 + (sig/mu)**2)

    logPrior = -0.5*((np.log(val)-logMu)/logSig)**2

    return logPrior

def calc_logPriors(thetaVec,paramsDict):
    """
    The natural logarithm of the prior probability.

    Parameters:
    ----------
    thetaVec : tuple
        A sample containing individual parameter values.
    uniform : bool, default=False

    Note:
        We can ignore the normalisations of the prior here.
    
    Returns:
    ----------
    logPrior : float
        Natural log of the priors.
    """

    paramKeys = list(paramsDict.keys())

    logPrior = 0
    for ind,theta in enumerate(thetaVec):
        # Get the prior dictionary for the input parameter.
        p0Dict = paramsDict[paramKeys[ind]]

        if p0Dict['prior'] == 'uniform':
            logPrior += uniformPrior(theta,p0Dict['hyperparams'])
        elif p0Dict['prior'] == 'normal':
            logPrior += normalPrior(theta,p0Dict['hyperparams'])
        elif p0Dict['prior'] == 'lognormal':
            logPrior += lognormalPrior(theta,p0Dict['hyperparams'])

    return logPrior

def logposterior(theta,data,sigma,paramsDict,xVec,model=power_law):
    """
    The natural logarithm of the joint posterior. Not normalised.

    Parameters:
    ----------
    theta : tuple
        A sample containing individual parameter values.
    data : numpy ndarray
        The set of data/observations.
    sigma : float, or numpy ndarray
        The standard deviation of the data points.
    paramsDict : dict
        Dictionary containing the hyperparameters for the model params.
    xVec : numpy ndarray
        The abscissa values at which the data/model is defined.
    model : function, default=power_law
        Model that describes the data. Default is the power law.
    
    Returns:
    ----------
    logPosterior : float
        Natural log of the posterior distribution.
    """

    lp = calc_logPriors(theta,paramsDict) # get the prior

    # if the prior is not finite return a probability of zero (log probability of -inf)
    if not np.isfinite(lp):
        return -np.inf

    # return the likeihood times the prior (log likelihood plus the log prior)
    logPosterior = lp + loglikelihood(theta,data,sigma,xVec,model=model)

    return logPosterior

def loglikelihood(theta,data,sigma,xVec,model=power_law):
    """
    The natural logarithm of the joint Gaussian likelihood.

    TODO add the curved power law model.

    Note:
        We do not include the normalisation constants (as discussed above).

    Parameters:
    ----------
    theta : tuple
        A sample containing individual parameter values.
    data : numpy ndarray
        The set of data/observations.
    sigma : float, or numpy ndarray
        The standard deviation of the data points.
    xVec : numpy ndarray
        The abscissa values at which the data/model is defined.
    model : function, default=power_law
        Model that describes the data. Default is the power law.

    
    Returns:
    ----------
    logl : float
        Evaluated log likelihood.
    """

    # unpack the model parameters from the tuple
    if model==power_law:
        S0, alpha = theta
        # Evaluate the model.
        mod_data = model(xVec/xVec[0], S0, alpha)

    # Residual of the model and the data.
    resid = mod_data - data

    if isinstance(sigma, np.ndarray):
        if len(sigma.shape) > 1 and (sigma.shape[0]==sigma.shape[1]):
            # If the covariance matrix is given.
            Kinv = np.linalg.inv(sigma)
            logl = -0.5*np.matmul(np.matmul(resid,Kinv),resid.T)
        else:
            # If only a vector of uncertainties is given.
            logl = -0.5 * np.sum((resid/sigma)**2)

    return logl

def initial_samples(paramsDict,Nens=100,p0cond=False):
    """
    Create the initial sample array for the ensemblers. Use the hyper parameters
    to determine the intil samples. 

    Parameters:
    ----------
    paramsDict : dict
        Dictionary containing the hyperparameters for the model params.
    Nens : int, default=100
        Number of points in the ensemble.
    
    Returns:
    ----------
    paramsSamples : float, numpy ndarray
        Initial ensemble parameters.
    """
    # Get the parameter dictionary keys.
    paramKeys = list(paramsDict.keys())
    # Set the parameter sample array.
    #paramsSamples = np.zeros((len(paramKeys),Nens))
    paramsSamples = np.zeros((Nens,len(paramKeys)))

    for ind,key in enumerate(paramKeys):
        
        # Get the parameter dictionary.
        p0Dict = paramsDict[key]
        # Get the hyperparameters.
        hypParams = list(p0Dict['hyperparams'])
        
        # If initial values are given, and p0cond = True, then use the initial
        # values in place of the hyperparameters. This is useful if you fit
        # with least squares first to get a good guess.
        try:
            if p0cond:
                hypParams[0] = p0Dict['p0']
        except KeyError:
            pass

        try:
            if p0cond:
                hypParams[1] = p0Dict['e_p0']
        except KeyError:
            pass

        if p0Dict['prior'] == 'uniform':
            pVec = np.random.uniform(hypParams[0],hypParams[1],Nens)
        elif p0Dict['prior'] == 'normal':
            pVec = np.random.normal(hypParams[0],hypParams[1], Nens) 
        elif p0Dict['prior'] == 'lognormal':
            mu = hypParams[0]
            sig = hypParams[1]

            # Log normal distribution satistics.
            logMu = np.log(mu**2/np.sqrt(mu**2 + sig**2))
            logSig = np.log(1 + (sig/mu)**2)

            pVec = np.random.lognormal(logMu,logSig,Nens)

        #paramsSamples[ind,:] = pVec
        paramsSamples[:,ind] = pVec

    return paramsSamples

def calc_logZ(sampler,thresh=None,Nburnin=500,norm=None):
    """
    Estimate the natural log of the evidence.
    """
    samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)

    theta_min = np.nanmin(samples_emcee,axis=0)
    theta_max = np.nanmax(samples_emcee,axis=0)

    samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)

    Nsamp = samples_emcee[:,0].size
    dA = np.prod(theta_max - theta_min)

    if np.any(norm):
        norm = norm
    else:
        norm = 1

    likelihood_samp = 10**(sampler.get_log_prob())
    
    
    
    #likelihood_samp = sampler.compute_log_prob()
    
    if np.any(thresh):
        pass
    else:
        thresh = np.nanstd(likelihood_samp)

    # Get all points above the threshold.
    thresh_inds = np.abs(likelihood_samp-np.nanmean(likelihood_samp))<thresh

    # Number of points above the threshold
    Ngt_thresh = likelihood_samp[thresh_inds].size

    # Estimate the area or olume element.
    dtheta = dA*(Ngt_thresh/Nsamp)
    print(dtheta,Ngt_thresh)

    #
    logZ = np.log(np.sum(likelihood_samp[thresh_inds]*dtheta/norm))

    return logZ

def plotposts(samples,paramsDict,popt=None, **kwargs):
    """
    Function to plot posteriors using corner.py and scipy's gaussian KDE 
    function.
    """
    # functions for plotting posteriors
    import corner
    from scipy.stats import gaussian_kde

    paramsKeys = list(paramsDict.keys())
    labels = [paramsDict[key]['label'] for key in paramsKeys]
    if "truths" not in kwargs:
        if np.any(popt):
            kwargs["truths"] = [popt[0], popt[1]]
        else:
            kwargs["truths"] = [paramsDict[paramsKeys[0]]['bounds'][0], 
                                paramsDict[paramsKeys[1]]['bounds'][0]]

    fig = corner.corner(samples, labels=labels, 
                        hist_kwargs={'density': True}, **kwargs)

    # plot KDE smoothed version of distributions
    for axidx, samps in zip([0, 3], samples.T):
        kde = gaussian_kde(samps)
        xvals = fig.axes[axidx].get_xlim()
        xvals = np.linspace(xvals[0], xvals[1], 100)
        fig.axes[axidx].plot(xvals, kde(xvals), color='firebrick')