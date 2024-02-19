import numpy as np

def deconv2(gaus_bm, gaus_c):
    """ 
    Deconvolves gaus_bm from gaus_c to give gaus_dc.
    Stolen shamelessly from Miriad gaupar.for.
    All PA is in degrees.

    Returns deconvolved gaussian parameters and flag:
     0   All OK.
     1   Result is pretty close to a point source.
     2   Illegal result.

    """
    
    from math import pi, cos, sin, atan2, sqrt

    rad = 180.0/pi

    phi_c = gaus_c[2]+900.0 % 180.0
    phi_bm = gaus_bm[2]+900.0 % 180.0
    theta1 = phi_c / rad
    theta2 = phi_bm / rad
    bmaj1 = gaus_c[0]
    bmaj2 = gaus_bm[0]
    bmin1 = gaus_c[1]
    bmin2 = gaus_bm[1]

    alpha = ( (bmaj1*cos(theta1))**2 + (bmin1*sin(theta1))**2 -
              (bmaj2*cos(theta2))**2 - (bmin2*sin(theta2))**2 )

    beta = ( (bmaj1*sin(theta1))**2 + (bmin1*cos(theta1))**2 -
             (bmaj2*sin(theta2))**2 - (bmin2*cos(theta2))**2 )

    gamma = 2.0 * ( (bmin1**2-bmaj1**2)*sin(theta1)*cos(theta1) -
                  (bmin2**2-bmaj2**2)*sin(theta2)*cos(theta2) )

    s = alpha + beta # 
    t = sqrt((alpha-beta)**2 + gamma**2) # Sqrt of the cosine law
    limit = min(bmaj1, bmin1, bmaj2, bmin2)
    limit = 0.1*limit*limit

    if alpha < 0.0 or beta < 0.0 or s < t:
        if alpha < 0.0 or beta < 0.0:
            bmaj = 0.0
            bpa = 0.0
        else:
            bmaj = sqrt(0.5*(s+t))
            bpa = rad * 0.5 * atan2(-gamma, alpha-beta)
        bmin = 0.0
        if 0.5*(s-t) < limit and alpha > -limit and beta > -limit:
            ifail = 1
        else:
            ifail = 2
    else:
        bmaj = sqrt(0.5*(s+t))
        bmin = sqrt(0.5*(s-t))
        if abs(gamma) + abs(alpha-beta) == 0.0:
            bpa = 0.0
        else:
            bpa = rad * 0.5 * atan2(-gamma, alpha-beta)
        ifail = 0
    return (bmaj, bmin, bpa), ifail

def gausslm_sig2abc(sigl,sigm,pa):
    """
    Calculate the 2D Gaussian parameters in the polynomial ax**2 + 2bxy +cy**2.

    Parameters:
    ----------
    sigl : float
        Gaussian major axis.
    sigm : float
        Gaussian minor axis.
    pa : float
        Gaussian position angle in radians. 

    Returns:
    ----------
    a : float
        x axis coefficient.
    b : float
        correlation coefficient.
    c : float
        y axis coefficient.
    """

    a = (np.cos(pa)**2)/(2*sigl**2) + (np.sin(pa)**2)/(2*sigm**2)
    b = (np.sin(2*pa))/(4*sigl**2) - (np.sin(2*pa))/(4*sigm**2)
    c = (np.sin(pa)**2)/(2*sigl**2) + (np.cos(pa)**2)/(2*sigm**2)


    return a,b,c

def gausslm_abc2sig(a,b,c):
    """
    Calculate the Gaussian sigma, and position angle from the polynomial 
    representation.

    Parameters:
    ----------
    a : float
        x axis coefficient.
    b : float
        correlation coefficient.
    c : float
        y axis coefficient.

    Returns:
    ----------
    sigl : float
        Gaussian major axis.
    sigm : float
        Gaussian minor axis.
    pa : float
        Gaussian position angle in radians. 
    """

    pa=np.pi/2 + 0.5*np.arctan2(2*b,(a-c))
    
    sigl = np.sqrt(0.5/(a*np.cos(pa)**2 + 2*b*np.cos(pa)*np.sin(pa) + \
                    c*np.sin(pa)**2))
    
    sigm = np.sqrt(0.5/(a*np.sin(pa)**2 - 2*b*np.cos(pa)*np.sin(pa) + \
                    c*np.cos(pa)**2))

    return sigl,sigm,pa

def gaussuv_sig2abc(sigl,sigm,pa):
    """
    Calculate the 2D Gaussian parameters in the polynomial ax**2 + 2bxy +cy**2.

    Parameters:
    ----------
    sigl : float
        Gaussian major axis.
    sigm : float
        Gaussian minor axis.
    pa : float
        Gaussian position angle in radians. 

    Returns:
    ----------
    alpha : float
        x axis coefficient.
    beta : float
        correlation coefficient.
    gamma : float
        y axis coefficient.
    """

    alpha = 2*(np.pi*sigl*np.cos(pa))**2 + 2*(np.pi*sigm*np.sin(pa))**2
    beta = (np.sin(2*pa)) * ((np.pi*sigl)**2 - (np.pi*sigm)**2)
    gamma = 2*(np.pi*sigl*np.sin(pa))**2 + 2*(np.pi*sigm*np.cos(pa))**2


    return alpha,beta,gamma

def gaussuv_abc2sig(alpha,beta,gamma):
    """
    Calculate the Gaussian sigma, and position angle from the polynomial 
    representation. This calculated the sigma and pa from the uv space (a,b,c),
    here we call them alpha, beta, gamma to distinguish we are in Fourier space.

    Parameters:
    ----------
    alpha : float
        x axis coefficient.
    beta : float
        correlation coefficient.
    gamma : float
        y axis coefficient.

    Returns:
    ----------
    sigl : float
        Gaussian major axis.
    sigm : float
        Gaussian minor axis.
    pa : float
        Gaussian position angle in radians. 
    """

    pa = 0.5*np.arctan2(2*beta,(alpha - gamma))

    sigl = np.sqrt(alpha*np.cos(pa)**2 + 2*beta*np.cos(pa)*np.sin(pa) + \
                    gamma*np.sin(pa)**2)/(np.sqrt(2)*np.pi)
    
    sigm = np.sqrt(alpha*np.sin(pa)**2 - 2*beta*np.cos(pa)*np.sin(pa) + \
                    gamma*np.cos(pa)**2)/(np.sqrt(2)*np.pi)

    return sigl,sigm,pa