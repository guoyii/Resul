import time
import numpy
import matplotlib.pyplot as plt


from sympy import *
from sympy.matrices import *
import numpy
import pylab
import warnings
import numpy as np
import numpy as np
from scipy.misc import comb

def Lagrange_interpolation(x, y, u):
    """
    Compute the Lagrange interpolation polynomial.
    # points : input x and y array
    # u : return value
    :var points: A numpy n×2 ndarray of the interpolations points
    :var variable: None, float or ndarray
    :returns:   * P the symbolic expression
                * Y the evaluation result of the polynomial  
    : Equation as : Pn(x)=sum ( Li(x)*yi ), n =0 ~ n ;     
    """
    Numbers = np.size(x)  # detect for how many input points
    dimension = np.size(u)
    report_vector = []  # create the report vectors for input u vector space
    for index in range(0, dimension):
        calculation = u[index]  # feed the input value
        result = 0  # clean the every input value result
        for i in range(0, Numbers):
            numerator = 1  # reset the numerator value on each line
            denominator = 1  # reset the denominator value on each line
            for j in range(0, Numbers):
                if (j != i):
                    numerator = numerator * (calculation - x[0][j])
                    denominator = denominator * (x[0][i] - x[0][j])
                else:
                    # print("not process while i=j")
                    pass
            result = result + y[0][i] * (numerator / denominator)
        report_vector = np.append(report_vector, result)
    return report_vector
def Newton_interpolation(x, y, u):
    x.astype(float)
    y.astype(float)
    n = np.size(x)
    Cn_parameters = []
    report_vector = []
    for i in range(0, n):
        Cn_parameters.append(y[0][i])
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            Cn_parameters[i] = float(Cn_parameters[i] - Cn_parameters[i - 1]) \
                               / float(x[0][i] - x[0][i - j])  # store the Cn parameters
    # print("The Cn Parameters = " + str(Cn_parameters))
    # setup the value calculation routine
    for i in range(0, np.size(u)):
        n = len(Cn_parameters) - 1
        result = Cn_parameters[n]
        for j in range(n - 1, -1, -1):
            # formula=C0+C1(u-x0)+C2(u-x0)(u-x1).....+Cn(u-x0)..(u-x_n-1)
            result = result * (u[i] - x[0][j]) + Cn_parameters[j]
            # print(n - 1)
        report_vector = np.append(report_vector, result)

    return report_vector
def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    # comb(n,i) : represents the n chooses it.
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
def bezier_curve(nPoints, nTimes):
    points = np.random.rand(nPoints, nTimes)  # create the matrix with nPoints x nTimes size
    xpoints = [p[0] for p in points]   # choose the p[0] as the starting point
    ypoints = [p[1] for p in points]   # chosse the p[0] as the ending point
    u = np.linspace(0.0, 1.0, nTimes)   # setup the interval as u vector 1 x n
    # core of the Bezier function calculaton upon the control point ( nPoints) and u vector
    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, u) for i in range(0, nPoints)])
    xvals = np.dot(xpoints, polynomial_array)
    yvals = np.dot(ypoints, polynomial_array)
    return points, xpoints, ypoints, xvals, yvals

def laginter(sinogram_sparse):
    sinogram_sparse = sinogram_sparse.T
    sinogram_inter = np.zeros((4, 10))
    Tx = np.array([np.linspace(0, 10, 3)])
    u = np.linspace(0, 10, 10)
    for i in range(4):
        Ty = np.array([sinogram_sparse[i]])
        result = Lagrange_interpolation(Tx, Ty, u)
        sinogram_inter[i] = result
    return sinogram_inter.T

def newinter(sinogram_sparse):
    sinogram_sparse = sinogram_sparse.T
    sinogram_inter = np.zeros((736, 1160))
    Tx = np.array([np.linspace(0, 1159, 60)])
    u = np.linspace(0, 1159, 1160)
    for i in range(736):
        # print(i)
        Ty = np.array([sinogram_sparse[i]])
        result = Newton_interpolation(Tx, Ty, u)
        # result = Lagrange_interpolation(Tx, Ty, u)
        sinogram_inter[i] = result
    return sinogram_inter.T