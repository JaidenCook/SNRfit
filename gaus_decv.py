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