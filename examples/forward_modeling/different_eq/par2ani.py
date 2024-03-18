import numpy as np

def get_thomsen_parameters(vp, vs, rho, epsilon, gamma, delta, _theta=45.):

    theta = np.deg2rad(_theta)
    # Calculate the thomsen parameters
    cv11 = rho*vp**2*(1+2*epsilon)
    # cv13 = rho*vp.pow(2)*np.sqrt(f*(f+2*delta))-rho*vs.pow(2)
    cv13 = rho*np.sqrt(((1+2*delta)*vp**2-vs**2)*(vp**2-vs**2))-rho*vs**2
    cv33 = rho*vp**2
    cv44 = rho*vs**2

    # Compute intermediate values
    cos_theta2 = np.power(np.cos(theta),2)
    sin_theta2 = np.power(np.sin(theta),2)
    sin_2theta = np.sin(2*theta)
    sin_2theta_sq = np.power(sin_2theta,2)

    # Compute outputs using intermediate values and Pynp tensor operations
    c11 = (cos_theta2*cv11+sin_theta2*cv13)*cos_theta2 \
        + (cos_theta2*cv13+sin_theta2*cv33)*sin_theta2 \
        + sin_2theta_sq*cv44
    c13 = (cos_theta2*cv11+sin_theta2*cv13)*sin_theta2 \
        + (cos_theta2*cv13+sin_theta2*cv33)*cos_theta2 \
        -  sin_2theta_sq*cv44
    c33 = (sin_theta2*cv11+cos_theta2*cv13)*sin_theta2 \
        + (sin_theta2*cv13+cos_theta2*cv33)*cos_theta2 \
        +  sin_2theta_sq*cv44
    c15 = .5*(cos_theta2*cv11+sin_theta2*cv13)*sin_2theta \
        - .5*(cos_theta2*cv13+sin_theta2*cv33)*sin_2theta \
        - sin_2theta*cv44*(cos_theta2-sin_theta2)
    c35 = .5*(sin_theta2*cv11+cos_theta2*cv13)*sin_2theta \
        - .5*(sin_theta2*cv13+cos_theta2*cv33)*sin_2theta \
        + sin_2theta*cv44*(cos_theta2-sin_theta2)
    c55 = .25*(sin_2theta*cv11-sin_2theta*cv13)*sin_2theta \
        - .25*(sin_2theta*cv13-sin_2theta*cv33)*sin_2theta \
        + cv44*(cos_theta2-sin_theta2)

    return c11, c13, c33, c15, c35, c55

