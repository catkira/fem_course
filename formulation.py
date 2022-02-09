import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pyvista as pv
import vtk
import time
import sys
import pkg_resources
from scipy.sparse import *
from parameter import *
from region import Region
from field import *
from utils import *

if 'petsc4py' in pkg_resources.working_set.by_key:
    hasPetsc = True
    import petsc4py
    petsc4py.init(sys.argv)        
    from petsc4py import PETSc
else:
    print("Warning: no petsc4py found, solving will be very slow!")

from mesh import *

# unofficial python 3.10 pip wheels for vtk 
# pip install https://github.com/pyvista/pyvista-wheels/raw/main/vtk-9.1.0.dev0-cp310-cp310-win_amd64.whl
# pip install https://github.com/pyvista/pyvista-wheels/raw/main/vtk-9.1.0.dev0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

def globalCoordinate(G, t, xi):
    B, x1 = transformationJacobian(G, t)
    return x1 + B @ xi    

def localCoordinate(G, t, x):
    B, x1 = transformationJacobian(G, t)
    xi,_,_,_ = np.linalg.lstsq(B, x-x1, rcond=None)
    return xi


# integral curl(u) * sigma * curl(tf(u)) 
def stiffnessMatrixCurl(field, sigmas, region=[]):
    Curls = field.shapeFunctionCurls()
    if mesh()['problemDimension'] == 2:    
        computeEdges2d()
        dim = 2
        nBasis = 3
        elementMatrixSize = nBasis**2
        elementArea = 1/2
    if mesh()['problemDimension'] == 3:
        computeEdges3d()
        dim = 3
        nBasis = 6
        elementMatrixSize = nBasis**2
        elementArea = 1/6
    if region == []:
        elements = mesh()['ett']
    else:
        elements = region.getElements(edges=True)
        sigmas = sigmas.getValues(region)
    computeSigns()
    m = len(elements)
    rows = np.zeros(m*elementMatrixSize)
    cols = np.zeros(m*elementMatrixSize)
    data = np.zeros(m*elementMatrixSize)    
    jacs = transformationJacobians(elementDim=dim)
    detJacs = np.abs(np.linalg.det(jacs))    
    invJacs = np.linalg.inv(jacs)      

    if mesh()['problemDimension'] == 3:
        B = np.zeros((dim,dim,nBasis,nBasis))
        for i in range(3):
            for k in range(3):
                B[i,k] = elementArea * np.matrix(Curls[:,i]).T * np.matrix(Curls[:,k]) 

        sigmasDuplicated = np.repeat(sigmas, dim**2).reshape((len(elements), dim, dim))
        detJacsDuplicated = np.repeat(detJacs, dim**2).reshape((len(elements), dim, dim))
        gammas = sigmasDuplicated * invJacs @ np.swapaxes(invJacs,1,2) * detJacsDuplicated
        rows = np.tile(elements, nBasis).astype(np.int64).ravel()
        cols = np.repeat(elements,nBasis).astype(np.int64).ravel()  
        for elementIndex, element in enumerate(elements):
            indexRange = np.arange(start=elementIndex*elementMatrixSize, stop=elementIndex*elementMatrixSize+elementMatrixSize)            
            signs = np.matrix(mesh()['signs3d'][elementIndex]).T @ np.matrix(mesh()['signs3d'][elementIndex])
            data[indexRange] = np.multiply(np.einsum('jk,jk...',gammas[elementIndex],B),signs).ravel() # this is generic and faster than explicit summation like below
    
    if mesh()['problemDimension'] == 2:
        pass
        # TODO
    
    n = numberOfEdges()      
    K = csr_matrix((data, (rows, cols)), shape=[n,n]) 
    return K    

# integral grad(u) * sigma * grad(tf(u)) 
def stiffnessMatrix(field, sigmas, region=[], vectorized=True, legacy=False):
    Grads = field.shapeFunctionGradients()
    if region == []:
        elements = mesh()['pt']
    else:
        elements = region.getElements()
        sigmas = sigmas.getValues(region)
    if mesh()['problemDimension'] == 2:
        dim = 2
        nBasis = 3
        area = 1/2
    elif mesh()['problemDimension'] == 3:
        dim = 3
        nBasis = 4
        area = 1/6        
    m = len(elements)
    elementMatrixSize = (mesh()['problemDimension']+1)**2
    rows = np.zeros(m*elementMatrixSize)
    cols = np.zeros(m*elementMatrixSize)
    data = np.zeros(m*elementMatrixSize)    
    data2 = np.zeros(0)    
    jacs = transformationJacobians(elementDim=dim)
    detJacs = np.abs(np.linalg.det(jacs))    
    invJacs = np.linalg.inv(jacs)       

    if not legacy:  
        # precalculate mesh independant parts of the integral
        B = np.zeros((dim, dim, nBasis, nBasis))
        for i in range(dim):
            for k in range(dim):
                B[i,k] = area * np.matrix(Grads[:,i]).T * np.matrix(Grads[:,k]) 
        
        if len(sigmas.shape) == 1:
            if False:
                # this looks ugly, bit is a tiny bit faster than einsum!
                detJacsDuplicated = np.repeat(detJacs, dim**2).reshape((len(elements), dim, dim))
                sigmasDuplicated = np.repeat(sigmas, dim**2).reshape((len(elements), dim, dim))
                gammas = sigmasDuplicated * invJacs @ np.swapaxes(invJacs,1,2) * detJacsDuplicated
            else:
                gammas = np.einsum('i,i,ijk,ilk->ijl',sigmas,detJacs,invJacs,invJacs)  # np.einsum is epic!
        else:
            detJacsDuplicated = np.repeat(detJacs, dim**2).reshape((len(elements), dim, dim))
            gammas = np.zeros((len(elements),dim,dim))
            for elementIndex in range(len(elements)):        # TODO: vectorize this        
                gammas[elementIndex] = sigmas[elementIndex] @ invJacs[elementIndex] @ np.swapaxes(invJacs,1,2)[elementIndex] * detJacsDuplicated[elementIndex]
        rows = np.tile(elements, nBasis).astype(np.int64).ravel()
        cols = np.repeat(elements,nBasis).astype(np.int64).ravel()
        if vectorized:
            data = np.einsum('ijk,jklm', gammas, B).swapaxes(0,2).ravel(order='F')
        else:
            for elementIndex, element in enumerate(elements):
                indexRange = np.arange(start=elementIndex*elementMatrixSize, stop=elementIndex*elementMatrixSize+elementMatrixSize)            
                data[indexRange] = np.einsum('jk,jk...', gammas[elementIndex],B).ravel() # this is generic and faster than explicit summation like below
    
    else:  # LEGACY CODE       
        if mesh()['problemDimension'] == 3:
            # precalculate mesh independant parts of the integral
            B = np.zeros((dim, dim, nBasis, nBasis))
            for i in range(dim):
                for k in range(dim):
                    B[i,k] = area * np.matrix(Grads[:,i]).T * np.matrix(Grads[:,k]) 
            
            sigmasDuplicated = np.repeat(sigmas, dim**2).reshape((len(elements), dim, dim))
            detJacsDuplicated = np.repeat(detJacs, dim**2).reshape((len(elements), dim, dim))
            gammas = sigmasDuplicated * invJacs @ np.swapaxes(invJacs,1,2) * detJacsDuplicated
            rows = np.tile(elements, nBasis).astype(np.int64).ravel()
            cols = np.repeat(elements,nBasis).astype(np.int64).ravel()      
            for elementIndex, element in enumerate(elements):
                indexRange = np.arange(start=elementIndex*elementMatrixSize, stop=elementIndex*elementMatrixSize+elementMatrixSize)            
                data[indexRange] = np.einsum('jk,jk...', gammas[elementIndex],B).ravel() # this is generic and faster than explicit summation like below

        if mesh()['problemDimension'] == 2:
            # area of ref triangle is 0.5, integrands are constant within integral
            B_11 = 0.5 * np.matrix(Grads[:,0]).T * np.matrix(Grads[:,0]) 
            B_12 = 0.5 * np.matrix(Grads[:,0]).T * np.matrix(Grads[:,1])
            B_21 = 0.5 * np.matrix(Grads[:,1]).T * np.matrix(Grads[:,0])
            B_22 = 0.5 * np.matrix(Grads[:,1]).T * np.matrix(Grads[:,1])
            #K_Ts = np.zeros([m,3,3])
            for elementIndex, element in enumerate(elements):
                jac,_ = transformationJacobian(elementIndex)
                detJac = np.abs(np.linalg.det(jac))
                if len(sigmas.shape) == 1:
                    gamma11 = sigmas[elementIndex]*1/detJac*np.dot(jac[:,1],jac[:,1])
                    gamma12 = -sigmas[elementIndex]*1/detJac*np.dot(jac[:,0],jac[:,1])
                    gamma21 = -sigmas[elementIndex]*1/detJac*np.dot(jac[:,1],jac[:,0])
                    gamma22 = sigmas[elementIndex]*1/detJac*np.dot(jac[:,0],jac[:,0])
                else:
                    invJac = np.linalg.inv(jac)
                    sigma_dash = invJac @ sigmas[elementIndex] @ invJac.T * detJac
                    gamma11 = sigma_dash[0,0] 
                    gamma12 = sigma_dash[1,0]
                    gamma21 = sigma_dash[0,1]
                    gamma22 = sigma_dash[1,1]
                indexRange = np.arange(start=elementIndex*elementMatrixSize, stop=elementIndex*elementMatrixSize+elementMatrixSize)            
                rows[indexRange] = np.tile(element[:],3).astype(np.int64)
                cols[indexRange] = np.repeat(element[:],3).astype(np.int64)
                data[indexRange] = (gamma11*B_11 + gamma12*B_12 + gamma21*B_21 + gamma22*B_22).ravel()
                #K_Ts[triangleIndex] = gamma1*B_11 + gamma2*B_12 + gamma3*B_21 + gamma4*B_22
                #K[np.ix_(triangle[:],triangle[:])] = K[np.ix_(triangle[:],triangle[:])] + K_T      
    n = numberOfVertices()
    K = csr_matrix((data, (rows, cols)), shape=[n,n]) 
    return K

# integral br * rot(tf(u))
def fluxRhsCurl(field, br, region=[], vectorized=True):
    if region == []:
        elements = mesh()['pt']
    else:
        elements = region.getElements(edges=True)
        br = br.getValues(region)
    n = numberOfEdges()
    rhs = np.zeros(n)
    if mesh()['problemDimension'] == 2:
        zero = [0, 0]
        area = 1/2
        Curls = field.shapeFunctionCurls()[:,0:2]  # discard z coordinates  
        nCoords = 2
        nBasis = 3
        dim = 2
    else:
        zero = [0, 0, 0]
        area = 1/6
        Curls = field.shapeFunctionCurls()    
        nCoords = 3
        nBasis = 6
        dim = 3
    jacs = transformationJacobians(elementDim=dim)
    detJacs = np.abs(np.linalg.det(jacs))    
    invJacs = np.linalg.inv(jacs)         
    if vectorized:
        temp2 = area* np.repeat(detJacs,nCoords*nBasis).reshape((len(detJacs),nCoords,nBasis))* np.einsum('ijk,kl',invJacs.swapaxes(1,2), Curls.T)
        rows = elements.ravel(order='F')
        rhs2 = np.zeros((len(elements),nBasis))
        for basis in range(nBasis):
            rhs2[:,basis] = np.einsum('ij,ij->i',br,temp2[:,:,basis])
        rhs = csr_matrix((rhs2.ravel(order='F'), (rows,np.zeros(len(rows)))), shape=[n,1]).toarray().ravel()
    else:
        for elementIndex, element in enumerate(elements):
            if np.array_equal(br[elementIndex], zero): # just for speedup
                continue
            temp = area * invJacs[elementIndex].T @ Curls.T * detJacs[elementIndex]
            for basis in range(nBasis):
                rhs[element[basis]] = rhs[element[basis]] + np.dot(br[elementIndex], temp.T[basis])
    return rhs

# integral br * grad(tf(u))
def fluxRhs(field, br, region=[], vectorized=True):
    if region == []:
        elements = mesh()['pt']
    else:
        elements = region.getElements()
        br = br.getValues(region)
    n = numberOfVertices()
    rhs = np.zeros(n)
    if mesh()['problemDimension'] == 2:
        zero = [0, 0]
        area = 1/2
        Grads = field.shapeFunctionGradients()[:,0:2]  # discard z coordinates  
        nCoords = 2
        nBasis = 3
        dim = 2
    else:
        zero = [0, 0, 0]
        area = 1/6
        Grads = field.shapeFunctionGradients()    
        nCoords = 3
        nBasis = 4
        dim = 3
    jacs = transformationJacobians(elements, elementDim = dim)
    detJacs = np.abs(np.linalg.det(jacs))    
    invJacs = np.linalg.inv(jacs)         
    if vectorized:
        temp2 = area* np.repeat(detJacs,nCoords*nBasis).reshape((len(detJacs),nCoords,nBasis))* np.einsum('ijk,kl',invJacs.swapaxes(1,2), Grads.T)
        rows = elements.ravel(order='F')
        rhs2 = np.zeros((len(elements),nBasis))
        for basis in range(nBasis):
            rhs2[:,basis] = np.einsum('ij,ij->i',br,temp2[:,:,basis])
        rhs = csr_matrix((rhs2.ravel(order='F'), (rows,np.zeros(len(rows)))), shape=[n,1]).toarray().ravel()
    else:
        for elementIndex, element in enumerate(elements):
            if np.array_equal(br[elementIndex], zero): # just for speedup
                continue
            temp = area * invJacs[elementIndex].T @ Grads.T * detJacs[elementIndex]
            for basis in range(nBasis):
                rhs[element[basis]] = rhs[element[basis]] + np.dot(br[elementIndex], temp.T[basis])
    return rhs

# integral rho * u * tf(u)
def massMatrixCurl(field, rhos, region=[], dim=2):
    if isinstance(region, list) or type(region) is np.ndarray:
        elements = region
    elif isinstance(region, Region):
        elements = region.getElements(edges=True)
        rhos = rhos.getValues(region)
        dim = region.regionDimension
    else:
        print("Error: unsupported paramter!")
        sys.exit()
    if dim == 1:
        #TODO
        sys.exit()
    elif dim == 2:
        order = 2
        nBasis = 3
        if order == 1:
            gfs = np.array([1/2])
            gps = np.array([[1/3, 1/3, 0]])
        elif order == 2:
            gfs = np.array([1/6, 1/6, 1/6])
            gps = np.array([[1/3, 1/6, 0],
                            [2/3, 1/6, 0],
                            [1/3, 2/3, 0]])
        Mm = np.zeros((3,3,nBasis,nBasis))  # TODO: is 3 correct here?
        for i in range(3):
            for k in range(3):
                for j in range(len(gfs)):
                    # Mm[i,k] = Mm[i,k] + gfs[j] * np.dot(field.shapeFunctionValues(gps[j])[i],field.shapeFunctionValues(gps[j])[k])
                    # np.matrix(Curls[:,i]).T * np.matrix(Curls[:,k]) 
                    Mm[i,k] = Mm[i,k] + gfs[j] * np.matrix(field.shapeFunctionValues(gps[j])).T[i,:] * np.matrix(field.shapeFunctionValues(gps[j]))[:,k]
    else:
        print("Error: this dimension is not implemented!")
        sys.exit()
    elementMatrixSize = nBasis**2     
    m = len(elements)
    rows = np.tile(elements, nBasis).astype(np.int64).ravel()
    cols = np.repeat(elements,nBasis).astype(np.int64).ravel()
    data = np.zeros(m*elementMatrixSize)    
    if dim == 1: # TODO: why can det(..) not be used here?
        sys.exit()
    elif dim == 2:
        jacs = transformationJacobians(elements, elementDim=2)
        detJacs = np.abs(np.linalg.det(jacs))    
        invJacs = np.linalg.inv(jacs)       
    for elementIndex, element in enumerate(elements):
        indexRange = np.arange(start=elementIndex*elementMatrixSize, stop=elementIndex*elementMatrixSize+elementMatrixSize)    
        signs = np.matrix(mesh()['signs3d'][elementIndex]).T @ np.matrix(mesh()['signs3d'][elementIndex])        
        data[indexRange] = (rhos[elementIndex]* invJacs[elementIndex] @ invJacs[elementIndex].T * detJacs[elementIndex] @ Mm).ravel()
    n = numberOfEdges()           
    M = csr_matrix((data, (rows, cols)), shape=[n,n]) 
    return M    

# integral rho * u * tf(u)
def massMatrix(field, rhos, region=[], dim=2, vectorized=True):
    Grads = field.shapeFunctionGradients()
    n = numberOfVertices()
    if isinstance(region, list) or type(region) is np.ndarray:
        elements = np.array(region)
    elif isinstance(region, Region):
        elements = region.getElements()
        rhos = rhos.getValues(region)
        dim = region.regionDimension
    else:
        print("Error: unsupported paramter!")
        sys.exit()
    if dim == 1:
        Mm = 1/6 * np.array([[2,1],
                            [1,2]])
        nBasis = 2
    elif dim == 2:
        nBasis = 3
        if False:
            Mm = 1/24 * np.array([[2,1,1],
                                [1,2,1],
                                [1,1,2]])
        else: 
            # precalculate mesh independant parts of the integral
            Mm = np.zeros((nBasis,nBasis))
            gfs = np.array([1/6, 1/6, 1/6])
            gps = np.array([[1/6, 1/6, 0],
                            [2/3, 1/6, 0],
                            [1/6, 2/3, 0]])
            for i,gp in enumerate(gps):
                for j in range(nBasis):
                    for k in range(nBasis):
                        Mm[j,k] = Mm[j,k] + gfs[i] * field.shapeFunctionValues(gp)[j] * field.shapeFunctionValues(gp)[k] 
    else:
        print("Error: this dimension is not implemented!")
        sys.exit()
    elementMatrixSize = nBasis**2        
    data = np.zeros(len(elements)*nBasis**2)    
    if dim == 1: # TODO: why can det(..) not be used here?
        detJacs = np.abs(np.linalg.norm( mesh()['xp'][elements[:,0]] - mesh()['xp'][elements[:,1]], axis=1))
    else:
        jacs = transformationJacobians(elements, elementDim=dim)
        detJacs = np.abs(np.linalg.det(jacs))        
    rows = np.tile(elements, nBasis).astype(np.int64).ravel()
    cols = np.repeat(elements,nBasis).astype(np.int64).ravel()
    if vectorized:
        data = np.einsum('i,jk',rhos * detJacs, Mm).ravel()
    else:
        for elementIndex, element in enumerate(elements):
            rangeIndex = np.arange(start=elementIndex*elementMatrixSize, stop=elementIndex*elementMatrixSize+elementMatrixSize)
            data[rangeIndex] = (rhos[elementIndex]*detJacs[elementIndex]*Mm).ravel()
            #M_T = rhos[triangleIndex]*detJac*Mm
            #M[np.ix_(triangle[:],triangle[:])] = M[np.ix_(triangle[:],triangle[:])] + M_T
    M = csr_matrix((data, (rows, cols)), shape=[n,n]) 
    return M    

# this function is only here for legacy
# it can be used to simulate the most simple poisson equation
def boundaryMassMatrix(field, alphas, region=[]):
    Grads = field.shapeFunctionGradients()
    Bb = 1/6 * np.array([[2,1],
                        [1,2]])
    r = numberOfBoundaryEdges()
    n = numberOfVertices()
    #B_Ts = np.zeros([r,2,2])
    rows = np.zeros(r*4)
    cols = np.zeros(r*4)
    data = np.zeros(r*4)
    if region == []:
        elements = mesh()['pe'][mesh()['eb']]
    else:
        elements = region.getElements()
        alphas = alphas.getValues(region)
    for elementIndex, element in enumerate(elements):
        detJac = np.abs(np.linalg.norm(mesh()['xp'][element[0]] - mesh()['xp'][element[1]]))
        range = np.arange(start=elementIndex*4, stop=elementIndex*4+4)        
        rows[range] = np.tile(element[:],2).astype(np.int64)
        cols[range] = np.repeat(element[:],2).astype(np.int64)
        data[range] = (alphas[elementIndex]*detJac*Bb).ravel()
        #B_Ts[edgeCount] = alphas[edgeCount]*detJac*Bb
        #B[np.ix_(ps[:],ps[:])] = B[np.ix_(ps[:],ps[:])] + B_T
    B = csr_matrix((data, (rows, cols)), shape=[n,n]) 
    return B

def solve(A, b, method='np'):
    start = time.time()
    if method == 'sparse':
        from scipy.sparse.linalg import inv    
        A = csc_matrix(A)
        u = inv(A) @ b
    elif method == 'petsc':
        if not hasPetsc:
            print("petsc is not available on this system")
            sys.exit()            
        n = len(b)   
        csr_mat=csr_matrix(A)
        Ap = PETSc.Mat().createAIJ(size=(n, n),  csr=(csr_mat.indptr, csr_mat.indices, csr_mat.data))        
        Ap.setUp()
        Ap.assemblyBegin()
        Ap.assemblyEnd()
        bp = PETSc.Vec().createSeq(n) 
        bp.setValues(range(n),b)        
        up = PETSc.Vec().createSeq(n)        
        ksp = PETSc.KSP().create()
        ksp.setOperators(Ap)        
        ksp.setFromOptions()
        print(f'Solving with: {ksp.getType():s}')
        ksp.solve(bp, up)
        print(f"Converged in {ksp.getIterationNumber():d} iterations.")
        u = np.array(up)
    elif method == 'np':
        u = np.linalg.inv(A.toarray()) @ b
    else:
        print("unknown method")
        sys.exit()
    stop = time.time()
    print(f"{bcolors.OKGREEN}solved in {stop - start:.2f} s{bcolors.ENDC}")    
    return u

def bookExample1():
    # example from book page 33
    n = numberOfVertices()
    m = numberOfTriangles()
    r = numberOfBoundaryEdges()
    sigmas = np.ones(m)
    rhos = np.ones(m)
    alphas = 1e9*np.ones(r)  # dirichlet BC
    f = np.ones(n)

    field = FieldH1()
    K = stiffnessMatrix(field, sigmas)
    M = massMatrix(field, rhos, region=mesh()['pt'])
    #B = boundaryMassMatrix(alphas) # this function is only here to illustrate the most simple way to do it
    B = massMatrix(field, alphas, region=mesh()['pe'][mesh()['eb']], dim=1)
    b = M @ f
    A = K+B
    u = solve(A,b)
    print(f'u_max = {max(u):.4f}')
    assert(abs(max(u) - 0.0732) < 1e-3)

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.plot_trisurf(G['xp'][:,0], G['xp'][:,1], u)
    #plt.show()

    storeInVTK(u,"example1.vtk", writePointData=True)

def bookExample2(scalarSigma, anisotropicInclusion=False, method='petsc'):
    # example from book page 34
    n = numberOfVertices()
    m = numberOfTriangles()
    r = numberOfBoundaryEdges()
    start = time.time()    
    if scalarSigma:
        sigmas = np.zeros(m)
        sigmaTensor1 = 1e-3
        sigmaTensor2 = 1
    else:
        sigmas = np.zeros((m,2,2))
        sigmaTensor2 = np.eye(2)        
        if anisotropicInclusion:
            sigmaTensor1 = np.array([[1.0001, 0.9999],
                                    [0.9999, 1.0001]])
        else:
            sigmaTensor1 = 1e-3*np.eye(2)
    for t in range(m):
        cog = np.sum(mesh()['xp'][mesh()['pt'][t,:],:],0)/3
        if 1<cog[0] and cog[0]<2 and 1<cog[1] and cog[1]<2:
            sigmas[t] = sigmaTensor1
        elif 3<cog[0] and cog[0]<4 and 2<cog[1] and cog[1]<3:
            sigmas[t] = sigmaTensor1
        else:
            sigmas[t] = sigmaTensor2
    alphas = np.zeros(r) 
    for e in range(r):
        cog = np.sum(mesh()['xp'][mesh()['pe'][mesh()['eb'][e],:],:],0)/2
        if abs(cog[0]-5) < 1e-6: 
            alphas[e] = 1e9 # Dirichlet BC
        elif abs(cog[0]) < 1e-6:
            alphas[e] = 1e-9 # Neumann BC
        else:
            alphas[e] = 0 # natural Neumann BC
    pd = np.zeros(n)
    for i in range(n):
        x = mesh()['xp'][i,:]
        if (abs(x[0]-5) < 1e-6):
            pd[i] = 4-x[1] # Dirichlet BC
        elif abs(x[0] < 1e-6):
            pd[i] = -1e9 # Neumann BC
    stop = time.time()    
    print(f'parameters prepared in {stop - start:.2f} s')        
    start = time.time()
    field = FieldH1()
    K = stiffnessMatrix(field, sigmas)
    B = boundaryMassMatrix(field, alphas)
    b = B @ pd
    A = K+B
    stop = time.time()    
    print(f'assembled in {stop - start:.2f} s')        
    u = solve(A, b, method)
    print(f'u_max = {max(u):.4f}')    
    assert(abs(max(u) - 4) < 1e-3)
    if anisotropicInclusion:
        #storeFluxInVTK(u,sigmas,"example2_anisotropicInclusions.vtk")
        pass
    else:
        if scalarSigma:
            storeInVTK(u,"example2_scalar_isotropicInclusions.vtk", writePointData=True)
        else:
            storeInVTK(u,"example2_tensor_isotropicInclusions.vtk", writePointData=True)
  

def bookExample2Parameter(scalarSigma, anisotropicInclusion=False, method='petsc'):
    # example from book page 34
    n = numberOfVertices()
    # regions
    incl1 = 0
    incl2 = 1
    air = 2
    infBottom = 3
    infLeft = 4
    infRight = 5
    infTop = 6

    start = time.time()    
    sigma = Parameter()
    sigma.set([incl1, incl2], 1e-3)
    sigma.set(air, 1)

    alpha = Parameter()
    alpha.set([infBottom, infTop], 0)
    alpha.set(infLeft, 1e-9) # Neumann BC
    alpha.set(infRight, 1e9) # Dirichlet BC

    pd = np.zeros(n)
    for i in range(n):
        x = mesh()['xp'][i,:]
        if (abs(x[0]-5) < 1e-6):
            pd[i] = 4-x[1] # Dirichlet BC
        elif abs(x[0] < 1e-6):
            pd[i] = -1e9 # Neumann BC
    stop = time.time()    
    print(f'parameters prepared in {stop - start:.2f} s')        
    start = time.time()

    surfaceRegion = Region()
    surfaceRegion.append([incl1, incl2, air])

    boundaryRegion = Region()
    boundaryRegion.append([infBottom, infTop, infLeft, infRight])

    field = FieldH1()
    K = stiffnessMatrix(field, sigma, surfaceRegion)    
    B = massMatrix(field, alpha, boundaryRegion)
    b = B @ pd
    A = K+B
    stop = time.time()    
    print(f'assembled in {stop - start:.2f} s')        
    u = solve(A, b, method)
    if anisotropicInclusion:
        #storeFluxInVTK(u,sigma.triangleValues,"example2_anisotropicInclusions_p.vtk")
        pass
    else:
        if scalarSigma:
            storeInVTK(u,"example2_scalar_isotropicInclusions_p.vtk", writePointData=True)
        else:
            storeInVTK(u,"example2_tensor_isotropicInclusions_p.vtk", writePointData=True)
    print(f'u_max = {max(u):.4f}')    
    assert(abs(max(u) - 4) < 1e-3)

def exampleMagnetInRoom():
    loadMesh("examples/magnet_in_room.msh")
    mu0 = 4*np.pi*1e-7
    mur_wall = 1000
    b_r_magnet = 1.5    
    # regions
    magnet = 1
    insideAir = 2
    wall = 3
    outsideAir = 4
    inf = 5

    start = time.time()
    mu = Parameter()
    mu.set(wall, mu0*mur_wall)
    mu.set([magnet, insideAir, outsideAir], mu0)
    #storeInVTK(mu, "mu.vtk")
    
    br = Parameter(2)
    br.set(magnet, [b_r_magnet, 0])
    br.set([wall, insideAir, outsideAir], [0, 0])
    #storeInVTK(br, "br.vtk")

    alpha = Parameter()
    alpha.set(inf, 1e9) # Dirichlet BC

    surfaceRegion = Region()
    surfaceRegion.append([wall, magnet, insideAir, outsideAir])

    boundaryRegion = Region()
    boundaryRegion.append(inf)

    field = FieldH1()
    K = stiffnessMatrix(field, mu, surfaceRegion)
    B = massMatrix(field, alpha, boundaryRegion)
    rhs = fluxRhs(field, br, surfaceRegion)
    b = rhs
    A = K+B
    stop = time.time()    
    print(f"{bcolors.OKGREEN}assembled in {stop - start:.2f} s{bcolors.ENDC}")        
    u = solve(A, b, 'petsc')
    storeInVTK(u,"magnet_in_room_phi.vtk", writePointData=True)
    m = numberOfTriangles()   
    h = -field.grad(u)
    storeInVTK(h,"magnet_in_room_h.vtk")
    mus = mu.getValues()  
    brs = np.column_stack([br.getValues(), np.zeros(m)])
    b = np.column_stack([mus,mus,mus])*h + brs  # this is a bit ugly
    storeInVTK(b,"magnet_in_room_b.vtk")
    print(f'b_max = {max(np.linalg.norm(b,axis=1)):.4f}')    
    assert(abs(max(np.linalg.norm(b,axis=1)) - 1.6054) < 1e-3)

def exampleHMagnetCurl():
    loadMesh("examples/h_magnet.msh")
    mu0 = 4*np.pi*1e-7
    mur_frame = 1000
    b_r_magnet = 1.5    
    # regions
    magnet = 0
    frame = 1
    air = 2
    inf = 3

    start = time.time()
    mu = Parameter()
    mu.set(frame, mu0*mur_frame)
    mu.set([magnet, air], mu0)
    #storeInVTK(mu, "mu.vtk")
    
    br = Parameter(3)
    br.set(magnet, [0, 0, b_r_magnet])
    br.set([frame, air], [0, 0, 0])
    #storeInVTK(br, "br.vtk")    

    alpha = Parameter()
    alpha.set(inf, 1e9) # Dirichlet BC

    volumeRegion = Region()
    volumeRegion.append([magnet, frame, air])

    boundaryRegion = Region()
    boundaryRegion.append(inf)

    field = FieldHCurl()
    K = stiffnessMatrixCurl(field, mu, volumeRegion)
    B = massMatrixCurl(field, alpha, boundaryRegion)
    rhs = fluxRhsCurl(field, br, volumeRegion)    
    b = rhs
    A = K+B    
    stop = time.time()
    print(f"{bcolors.OKGREEN}assembled in {stop - start:.2f} s{bcolors.ENDC}")       
    u = solve(A, b, 'petsc')    
    storeInVTK(u, "h_magnetCurl_u.vtk", writePointData=True)    
    h = -field.grad(u, dim=3)
    storeInVTK(h, "h_magnetCurl_h.vtk")
    mus = mu.getValues()  
    m = numberOfTetraeders()       
    brs = br.getValues()
    b = np.column_stack([mus,mus,mus])*h + brs  # this is a bit ugly
    storeInVTK(b,"h_magnetCurl_b.vtk")    
    print(f'b_max = {max(np.linalg.norm(b,axis=1)):.4f}')    
    assert(abs(max(np.linalg.norm(b,axis=1)) - 3.2898) < 1e-3)

def exampleHMagnet(vectorized=True, legacy=False):
    loadMesh("examples/h_magnet.msh")
    mu0 = 4*np.pi*1e-7
    mur_frame = 1000
    b_r_magnet = 1.5    
    # regions
    magnet = 0
    frame = 1
    air = 2
    inf = 3

    start = time.time()
    mu = Parameter()
    mu.set(frame, mu0*mur_frame)
    mu.set([magnet, air], mu0)
    #storeInVTK(mu, "mu.vtk")
    
    br = Parameter(3)
    br.set(magnet, [0, 0, b_r_magnet])
    br.set([frame, air], [0, 0, 0])
    #storeInVTK(br, "br.vtk")    

    alpha = Parameter()
    alpha.set(inf, 1e9) # Dirichlet BC

    volumeRegion = Region()
    volumeRegion.append([magnet, frame, air])

    boundaryRegion = Region()
    boundaryRegion.append(inf)

    field = FieldH1()
    K = stiffnessMatrix(field, mu, volumeRegion, vectorized=vectorized, legacy=legacy)
    B = massMatrix(field, alpha, boundaryRegion, vectorized=vectorized)
    rhs = fluxRhs(field, br, volumeRegion, vectorized=vectorized)    
    b = rhs
    A = K+B    
    stop = time.time()
    print(f"{bcolors.OKGREEN}assembled in {stop - start:.2f} s{bcolors.ENDC}")       
    u = solve(A, b, 'petsc')    
    storeInVTK(u, "h_magnet_u.vtk", writePointData=True)    
    h = -field.grad(u, dim=3)
    storeInVTK(h, "h_magnet_h.vtk")
    mus = mu.getValues()  
    m = numberOfTetraeders()       
    brs = br.getValues()
    b = np.column_stack([mus,mus,mus])*h + brs  # this is a bit ugly
    storeInVTK(b,"h_magnet_b.vtk")    
    print(f'b_max = {max(np.linalg.norm(b,axis=1)):.4f}')    
    assert(abs(max(np.linalg.norm(b,axis=1)) - 2.863) < 1e-3)

def exampleHMagnetOctant(vectorized=True, legacy=False):
    loadMesh("examples/h_magnet_octant.msh")
    mu0 = 4*np.pi*1e-7
    mur_frame = 1000
    b_r_magnet = 1.5    
    # regions
    magnet = 1
    frame = 2
    air = 3
    inf = 4
    innerXZBoundary = 5
    innerYZBoundary = 6
    innerXYBoundary = 7
    magnetXYBoundary = 8

    start = time.time()
    mu = Parameter()
    mu.set(frame, mu0*mur_frame)
    mu.set([magnet, air], mu0)
    #storeInVTK(mu, "mu.vtk")
    
    br = Parameter(3)
    br.set(magnet, [0, 0, b_r_magnet])
    br.set([frame, air], [0, 0, 0])
    #storeInVTK(br, "br.vtk")    

    alpha = Parameter()
    alpha.set([inf, innerXYBoundary, magnetXYBoundary], 1e9) # Dirichlet BC

    volumeRegion = Region()
    volumeRegion.append([magnet, frame, air])

    boundaryRegion = Region()
    boundaryRegion.append([inf, innerXYBoundary, magnetXYBoundary])

    field = FieldH1()
    K = stiffnessMatrix(field, mu, volumeRegion, vectorized=vectorized, legacy=legacy)
    B = massMatrix(field, alpha, boundaryRegion, vectorized=vectorized)
    rhs = fluxRhs(field, br, volumeRegion, vectorized=vectorized)    
    b = rhs
    A = K+B    
    stop = time.time()
    print(f"{bcolors.OKGREEN}assembled in {stop - start:.2f} s{bcolors.ENDC}")       
    u = solve(A, b, 'petsc')    
    storeInVTK(u, "h_magnet_octant_u.vtk", writePointData=True)    
    h = -field.grad(u, dim=3)
    storeInVTK(h, "h_magnet_octant_h.vtk")
    mus = mu.getValues()  
    m = numberOfTetraeders()       
    brs = br.getValues()
    b = np.column_stack([mus,mus,mus])*h + brs  # this is a bit ugly
    storeInVTK(b,"h_magnet_octant_b.vtk")        
    print(f'b_max = {max(np.linalg.norm(b,axis=1)):.4f}')    
    assert(abs(max(np.linalg.norm(b,axis=1)) - 2.675) < 1e-3)

def runAll():
    loadMesh("examples/air_box_2d.msh")
    computeEdges2d()
    computeBoundary()    
    bookExample1()

    rectangularCriss(50,50)
    computeEdges2d()
    computeBoundary()    
    bookExample1()

    loadMesh("examples/example2.msh")
    bookExample2Parameter(True, anisotropicInclusion=False, method='petsc')
    computeEdges2d()
    computeBoundary()       
    bookExample2(False, anisotropicInclusion=True, method='petsc')    

    rectangularCriss(50,50)
    computeEdges2d()
    computeBoundary()       
    mesh()['xp'][:,0] = mesh()['xp'][:,0]*5
    mesh()['xp'][:,1] = mesh()['xp'][:,1]*4    
    bookExample2(False, anisotropicInclusion=True, method='petsc')    

    exampleHMagnetOctant()
    exampleHMagnetOctant(vectorized=False)
    exampleHMagnetOctant(vectorized=False, legacy=True)
    exampleHMagnet()
    exampleHMagnet(vectorized=False)
    exampleHMagnet(vectorized=False, legacy=True)
    
    exampleMagnetInRoom()    

def main():
    if False:
        G = {}
        # store point coordinates 'xp'
        G['xp'] = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]])
        # store the points which make up an element in 'pt'
        G['pt'] = np.array([
            [0, 1, 2],
            [0, 2, 3]])        
        print("number of Vertices(G) = " + str(numberOfVertices(G)))
        print("number of Triangles(G) = " + str(numberOfTriangles(G)))
        xi = np.array([0, 1])
        p = globalCoordinate(G, 0, xi)
        print(f'point ({xi[0]:d}, {xi[1]:d}) of ref triangle transformed to global triangle = ({p[0]:d}, {p[1]:d})')
        xi = localCoordinate(G, 0, p)
        print(f'point ({p[0]:d}, {p[1]:d}) of global triangle transformed to ref triangle = ({xi[0]:f}, {xi[1]:f})')
    
    #plotShapeFunctions()
    #loadMesh("examples/air_box_2d.msh")
    # rectangularCriss(50,50)
    # plotMesh(G)

    if False:
        runAll()
    else:
        exampleHMagnetCurl()  # WIP
        exampleHMagnetOctant()
        exampleHMagnet()    
        exampleMagnetInRoom()  

    print('finished')

if __name__ == "__main__":
    main()