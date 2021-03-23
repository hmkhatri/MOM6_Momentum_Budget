"""
Functions for filtering 2D model output to coarser resolution
This puts together the code snippets from Ian's CPT-Snippets repo:
https://github.com/iangrooms/CPT-Snippets 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
import matplotlib.pylab as pylab

def filterSpec(N,dxMin,Lf,shape="Gaussian",X=np.pi):
    """
    Inputs: 
    N is the number of total steps in the filter
    dxMin is the smallest grid spacing - should have same units as Lf
    Lf is the filter scale, which has different meaning depending on filter shape
    shape can currently be one of two things:
        Gaussian: The target filter has kernel ~ e^{-|x/Lf|^2}
        Taper: The target filter has target grid scale Lf. Smaller scales are zeroed out. 
               Scales larger than pi*Lf/2 are left as-is. In between is a smooth transition.
    X is the width of the transition region in the "Taper" filter; per the CPT Bar&Prime doc the default is pi.
    Note that the above are properties of the *target* filter, which are not the same as the actual filter.
    
    Outputs:
    NL is the number of Laplacian steps
    sL is s_i for the Laplacian steps; units of sL are one over the units of dxMin and Lf, squared
    NB is the number of Biharmonic steps
    sB is s_i for the Biharmonic steps; units of sB are one over the units of dxMin and Lf, squared
    """
    # Code only works for N>2
    if N <= 2:
        print("Code requires N>2")
        return 
    # First set up the mass matrix for the Galerkin basis from Shen (SISC95)
    M = (np.pi/2)*(2*np.eye(N-1) - np.diag(np.ones(N-3),2) - np.diag(np.ones(N-3),-2))
    M[0,0] = 3*np.pi/2
    # The range of wavenumbers is 0<=|k|<=sqrt(2)*pi/dxMin. Nyquist here is for a 2D grid. 
    # Per the notes, define s=k^2.
    # Need to rescale to t in [-1,1]: t = (2/sMax)*s -1; s = sMax*(t+1)/2
    sMax = 2*(np.pi/dxMin)**2
    # Set up target filter
    if shape == "Gaussian":
        F = lambda t: np.exp(-(sMax*(t+1)/2)*(Lf/2)**2)
    elif shape == "Taper":
        F = interpolate.PchipInterpolator(np.array([-1,(2/sMax)*(np.pi/(X*Lf))**2 -1,(2/sMax)*(np.pi/Lf)**2 -1,2]),np.array([1,1,0,0]))
    else:
        print("Please input a valid shape")
        return
    # Compute inner products of Galerkin basis with target
    b = np.zeros(N-1)
    points, weights = np.polynomial.chebyshev.chebgauss(N+1)
    for i in range(N-1):
        tmp = np.zeros(N+1)
        tmp[i] = 1
        tmp[i+2] = -1
        phi = np.polynomial.chebyshev.chebval(points,tmp)
        b[i] = np.sum(weights*phi*(F(points)-((1-points)/2 + F(1)*(points+1)/2)))
    # Get polynomial coefficients in Galerkin basis
    cHat = np.linalg.solve(M,b)
    # Convert back to Chebyshev basis coefficients
    p = np.zeros(N+1)
    p[0] = cHat[0] + (1+F(1))/2
    p[1] = cHat[1] - (1-F(1))/2
    for i in range(2,N-1):
        p[i] = cHat[i] - cHat[i-2]
    p[N-1] = -cHat[N-3]
    p[N] = -cHat[N-2]
    # Now plot the target filter and the approximate filter
    #x = np.linspace(-1,1,251)
    x = np.linspace(-1,1,10000)
    k = np.sqrt((sMax/2)*(x+1))
    #params = {'legend.fontsize': 'x-large',
    #     'axes.labelsize': 'x-large',
    #     'axes.titlesize':'x-large',
    #     'xtick.labelsize':'x-large',
    #     'ytick.labelsize':'x-large'}
    #pylab.rcParams.update(params)
    plt.plot(k,F(x),'g',label='target filter',linewidth=4)
    plt.plot(k,np.polynomial.chebyshev.chebval(x,p),'m',label='approximation',linewidth=4)
    #plt.xticks(np.arange(5), ('0', r'$1/\Delta x$', r'$2/\Delta x$',r'$3/\Delta x$', r'$4/\Delta x$'))
    plt.axvline(1/Lf,color='k',linewidth=2)
    plt.axvline(np.pi/Lf,color='k',linewidth=2)
    #plt.text(1/Lf, 1.15, r'$\frac{1}{2}$',fontsize=20)
    #plt.text(np.pi/Lf, 1.15, r'$\frac{\pi}{2}$',fontsize=20)
    left, right = plt.xlim()
    plt.xlim(left=0)
    bottom,top = plt.ylim()
    plt.ylim(bottom=-0.1)
    plt.ylim(top=1.1)
    plt.xlabel('k', fontsize=18)
    plt.grid(True)
    plt.legend()
    #plt.savefig('figures/filtershape_%s%i_dxMin%i_Lf%i.png' % (shape,N,dxMin,Lf),dpi=400,bbox_inches='tight',pad_inches=0)
    
    # Get roots of the polynomial
    r = np.polynomial.chebyshev.chebroots(p)
    # convert back to s in [0,sMax]
    s = (sMax/2)*(r+1)
    # Separate out the real and complex roots
    NL = np.size(s[np.where(np.abs(np.imag(r)) < 1E-12)]) 
    sL = np.real(s[np.where(np.abs(np.imag(r)) < 1E-12)])
    NB = (N - NL)//2
    sB_re,indices = np.unique(np.real(s[np.where(np.abs(np.imag(r)) > 1E-12)]),return_index=True)
    sB_im = np.imag(s[np.where(np.abs(np.imag(r)) > 1E-12)])[indices]
    sB = sB_re + sB_im*1j
    return NL,sL,NB,sB

def applyFilter(field,landMask,dx,dy,NL,sL, NB, sB):
    """
    Filters a 2D field, applying an operator of type (*) above. 
    Assumes dy=constant, dx varies in y direction
    Inputs:
    field: 2D array (y, x) to be filtered
    landMask: 2D array, same size as field: 0 if cell is not on land, 1 if it is on land.
    dx is a 1D array, same size as 1st dimension of field
    dy is constant
    NL is number of Laplacian steps, see output of filterSpec fct above
    sL is s_i for the Laplacian steps, see output of filterSpec fct above
    NB is the number of Biharmonic steps, see output of filterSpec fct above
    sB is s_i for the Biharmonic steps, see output of filterSpec fct above
    Output:
    Filtered field.
    """
    fieldBar = field.copy() # Initalize the filtering process
    for i in range(NL):
        tempL = Laplacian2D_FV(fieldBar,landMask,dx,dy) # Compute Laplacian
        fieldBar = fieldBar + (1/sL[i])*tempL # Update filtered field
    for i in range(NB): 
        tempL = Laplacian2D_FV(fieldBar,landMask,dx,dy) # Compute Laplacian
        tempB = Laplacian2D_FV(tempL,landMask,dx,dy) # Compute Biharmonic (apply Laplacian twice)
        fieldBar = fieldBar + (2*np.real(sB[i])/np.abs(sB[i])**2)*tempL + (1/np.abs(sB[i])**2)*tempB
    return fieldBar   

def Laplacian2D_FV(field,landMask,dx,dy):
    """
    Computes a Cartesian Laplacian of field, using a finite volume discretization. 
    Assumes dy=constant, dx varies in y direction
    Inputs:
    field is a 2D array (y, x) whose Laplacian is computed; note: (y,x) is order of dims in NW2 output 
    landMask: 2D array, same size as field: 0 if cell is not on land, 1 if it is on land.
    dx is a 1D array, same size as 1st dimension of field
    dy is constant
    Output:
    Laplacian of field.
    """
    Ny = np.size(field,0)
    Nx = np.size(field,1)
    notLand = 1 - landMask
    field = np.nan_to_num(field) # set all NaN's to zero
    ## transpose all fields so that numpy broadcasting coorperates (when multiplying with dx)
    field = np.transpose(field)
    notLand = np.transpose(notLand)
    ## Approximate x derivatives on left and right cell boundaries
    fluxLeft = np.zeros((Nx,Ny))
    fluxRight = np.zeros((Nx,Ny))
    fluxRight[0:Nx-1,:] = notLand[1:Nx,:]*(field[1:Nx,:] - field[0:Nx-1,:])/dx # Set flux to zero if on land
    fluxRight[Nx-1,:] = notLand[0,:]*(field[0,:]-field[Nx-1,:])/dx # Periodic unless there's land in the way
    fluxLeft[1:Nx,:] = notLand[0:Nx-1,:]*(field[1:Nx,:] - field[0:Nx-1,:])/dx # Set flux to zero if on land
    fluxLeft[0,:] = notLand[Nx-1,:]*(field[0,:]-field[Nx-1,:])/dx # Periodic unless there's land in the way
    # multiply by length of cell boundary
    fluxLeft = fluxLeft*dy 
    fluxRight = fluxRight*dy
    OUT = fluxRight - fluxLeft
    # Approximate y derivatives on south and north cell boundaries
    fluxNorth = np.zeros((Nx,Ny))
    fluxSouth = np.zeros((Nx,Ny))
    fluxNorth[:,0:Ny-1] = notLand[:,1:Ny]*(field[:,1:Ny] - field[:,0:Ny-1])/dy # Set flux to zero if on land
    fluxNorth[:,Ny-1] = notLand[:,0]*(field[:,0]-field[:,Ny-1])/dy # Periodic unless there's land in the way
    fluxSouth[:,1:Ny] = notLand[:,0:Ny-1]*(field[:,1:Ny] - field[:,0:Ny-1])/dy # Set flux to zero if on land
    fluxSouth[:,0] = notLand[:,Ny-1]*(field[:,0]-field[:,Ny-1])/dy # Periodic unless there's land in the way
    # multiply by length of cell boundary
    # note: the following 4 lines is where this code makes a difference from above Laplacian2D_FD
    fluxNorth[:,0:Ny-1] = fluxNorth[:,0:Ny-1]*(dx[0:Ny-1]+dx[1:Ny])/2 
    fluxNorth[:,Ny-1] = fluxNorth[:,Ny-1]*(dx[Ny-1]+dx[0])/2 # Periodic unless there's land in the way
    fluxSouth[:,1:Ny] = fluxSouth[:,1:Ny]*(dx[0:Ny-1]+dx[1:Ny])/2 
    fluxSouth[:,0] = fluxSouth[:,0]*(dx[0]+dx[Nx-1])/2 # Periodic unless there's land in the way
    OUT = OUT + (fluxNorth - fluxSouth)
    # divide by cell area
    area = dx*dy
    OUT = notLand * OUT/area
    OUT = np.transpose(OUT) # transpose back
    return OUT
