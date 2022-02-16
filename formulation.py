import mmap
from os import killpg
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
import spanningtree as st

if 'petsc4py' in pkg_resources.working_set.by_key:
    hasPetsc = True
    import petsc4py
    petsc4py.init(sys.argv)        
    from petsc4py import PETSc
else:
    print("Warning: no petsc4py found, solving will be very slow!")
np.set_printoptions(linewidth=400)    

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
def stiffnessMatrixCurl(field, sigmas, region=[], vectorized=True):
    if region == []:
        elements = mesh()['ett']
        elementDim = mesh()['problemDimension'] 
    else:
        elements = region.getElements(edges=True)
        sigmas = sigmas.getValues(region)
        elementDim = region.regionDimension
    if elementDim == 2:    
        elementDim = 2
        nBasis = 3
        elementArea = 1/2
    elif elementDim == 3:
        elementDim = 3
        nBasis = 6
        elementArea = 1/6        
    curls = field.shapeFunctionCurls(elementDim)
    m = len(elements)
    elementMatrixSize = nBasis**2
    data = np.zeros(m*elementMatrixSize)    
    jacs = transformationJacobians(elementDim=elementDim)
    detJacs = np.abs(np.linalg.det(jacs))
    n = numberOfEdges()      

    if elementDim == 2:
        print("Error: hcurl elements are not possible in 2d!")
        sys.exit()
    elif elementDim == 3:
        rows = np.tile(elements, nBasis).astype(np.int64).ravel()
        cols = np.repeat(elements, nBasis).astype(np.int64).ravel()  
        signs = np.einsum('ij,ik->ijk',mesh()['signs3d'],mesh()['signs3d']) 
        if True:
            # this formulation might be a bit faster but only supports order 1!
            B = np.zeros((elementDim, elementDim, nBasis, nBasis))
            for i in range(3):
                for k in range(3):
                    B[i,k] = elementArea * np.matrix(curls[:,i]).T * np.matrix(curls[:,k]) 
            gammas = np.einsum('i,i,ikj,ikl->ijl', sigmas, 1/detJacs, jacs, jacs)
            if vectorized:
                data = np.einsum('ilm,ijk,jklm->ilm', signs, gammas, B).ravel(order='C')
        else:
            # this formulation is more generic, because it supports higher orders
            data2 = np.zeros((len(elements), nBasis, nBasis))
            signs = mesh()['signs3d']
            for i in range(1):
                for m in range(nBasis):
                    for k in range(nBasis):
                        factor1 = np.einsum('i,i,i,ijk,k->ij', signs[:,m], sigmas, elementArea * 1/detJacs, jacs, curls[m,:])
                        factor2 = np.einsum('i,ijk,k->ij', signs[:,k], jacs, curls[k,:])
                        data2[:,m,k] = np.einsum('ij,ij->i', factor1, factor2)    
            data = data2.ravel(order='C')
    K = csr_matrix((data, (rows, cols)), shape=[n,n]) 
    return K

# integral grad(u) * sigma * grad(tf(u)) 
def stiffnessMatrix(field, sigmas, region=[], vectorized=True, legacy=False):
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
    Grads = field.shapeFunctionGradients(dim)
    elementMatrixSize = (mesh()['problemDimension']+1)**2
    rows = np.zeros(m*elementMatrixSize)
    cols = np.zeros(m*elementMatrixSize)
    data = np.zeros(m*elementMatrixSize)    
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
            data = np.einsum('ijk,jklm', gammas, B).ravel(order='C')
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
    if mesh()['problemDimension'] == 2:
        area = 1/2
        nBasis = 3
        elementDim = 2
        signs = mesh()['signs2d']
    else:
        area = 1/6
        nBasis = 6
        elementDim = 3
        signs = mesh()['signs3d']
    curls = field.shapeFunctionCurls(elementDim)    
    jacs = transformationJacobians([], elementDim)
    invJacs = transformationJacobians([], elementDim)
    detJacs = np.abs(np.linalg.det(jacs))

    #temp2 = area* np.einsum('i,i...->i...', detJacs, np.einsum('ikj,lk', invJacs, curls))    
    # the line below seems to be right
    temp = area* np.einsum('ik,ijk->ijk', signs, np.einsum('ijk,lk', jacs, curls))  # TODO is detJac needed here??
    rows = elements.ravel(order='C')
    rhs2 = np.zeros((len(elements),nBasis))
    for basis in range(nBasis):
        rhs2[:,basis] = np.einsum('ij,ij->i', br, temp[:,:,basis])
    n = numberOfEdges()
    rhs = csr_matrix((rhs2.ravel(order='C'), (rows,np.zeros(len(rows)))), shape=[n,1]).toarray().ravel()

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
        #Grads = field.shapeFunctionGradients()[:,0:2]  # discard z coordinates  
        nCoords = 2
        nBasis = 3
        dim = 2
    else:
        zero = [0, 0, 0]
        area = 1/6
        nCoords = 3
        nBasis = 4
        dim = 3
    Grads = field.shapeFunctionGradients(dim)    
    jacs = transformationJacobians(elements, elementDim = dim)
    detJacs = np.abs(np.linalg.det(jacs))    
    invJacs = np.linalg.inv(jacs)         
    if vectorized:
        # this line has convergence issues with example magnet_in_room
        # temp2 = area * np.einsum('i,ikj,lk->ijl', detJacs, invJacs, Grads)
        # this line has no convergence issues - strange!
        # temp2 = area* np.repeat(detJacs,nCoords*nBasis).reshape((len(detJacs),nCoords,nBasis))* np.einsum('ikj,lk', invJacs, Grads)        
        # this line is also ok, the order of multiplication is probably favorable with regard to rounding errors
        temp2 = area* np.einsum('i,i...->i...', detJacs, np.einsum('ikj,lk', invJacs, Grads))
        rows = elements.ravel(order='C')
        rhs2 = np.zeros((len(elements),nBasis))
        for basis in range(nBasis):
            rhs2[:,basis] = np.einsum('ij,ij->i', br, temp2[:,:,basis])
        rhs = csr_matrix((rhs2.ravel(order='C'), (rows,np.zeros(len(rows)))), shape=[n,1]).toarray().ravel()
    else:
        for elementIndex, element in enumerate(elements):
            if np.array_equal(br[elementIndex], zero): # just for speedup
                continue
            temp = area * invJacs[elementIndex].T @ Grads.T * detJacs[elementIndex]
            for basis in range(nBasis):
                rhs[element[basis]] = rhs[element[basis]] + np.dot(br[elementIndex], temp.T[basis])
    return rhs

# integral rho * u * tf(u)
def massMatrixCurl(field, rhos, region=[], elementDim=2, verify=False):
    if isinstance(region, list) or type(region) is np.ndarray:
        elements = region
    elif isinstance(region, Region):
        elements = region.getElements(edges=True)
        rhos = rhos.getValues(region)
        elementDim = region.regionDimension
    else:
        print("Error: unsupported paramter!")
        sys.exit()
    if elementDim == 2:
        integrationOrder = 2 # should be elementOrder*2
        nBasis = 3
        gfs,gps = gaussData(integrationOrder, elementDim)
    else:
        print("Error: this dimension is not implemented!")
        sys.exit()
    elementMatrixSize = nBasis**2     
    m = len(elements)
    rows = np.tile(elements, nBasis).astype(np.int64).ravel()
    cols = np.repeat(elements,nBasis).astype(np.int64).ravel()
    data = np.zeros(m*elementMatrixSize)    
    n = numberOfEdges()         
    if elementDim == 2:
        jacs = transformationJacobians(region.getElements(edges=False), elementDim=elementDim)
        detJacs = np.abs(np.linalg.det(jacs))    
        invJacs = np.linalg.inv(jacs)       
        signs = mesh()['signs2d'] 
        calcMethod = 1
        if calcMethod == 1 or verify:
            nCoords = 3
            Mm = np.zeros((nCoords, nCoords, nBasis, nBasis))
            for i,gp in enumerate(gps):
                for m in range(nBasis):
                    for k in range(nBasis):
                        Mm[m,k] = Mm[m,k] + gfs[i] * np.einsum('i,j->ij',
                            field.shapeFunctionValues(gp, elementDim)[:,m], field.shapeFunctionValues(gp, elementDim)[:,k])
            gammas = np.einsum('i,i,ikj,ikl->ijl', rhos, detJacs, invJacs, invJacs)
            signsMultiplied = np.einsum('ij,ik->ijk', signs, signs)
            data = np.einsum('ilm,ijk,jklm->iml', signsMultiplied, gammas, Mm).ravel()
            M = csr_matrix((data, (rows, cols)), shape=[n,n]) 

        if calcMethod == 2 or verify:
            data2 = np.zeros((len(elements), nBasis, nBasis))
            for i,gp in enumerate(gps):
                for m in range(nBasis):
                    for k in range(nBasis):
                        factor1 = np.einsum('i,i,i,ijk,k->ij', signs[:,m], rhos, detJacs, invJacs, field.shapeFunctionValues(gp, elementDim)[m,:])
                        factor2 = np.einsum('i,ijk,k->ij', signs[:,k], invJacs, field.shapeFunctionValues(gp, elementDim)[k,:])                   
                        data2[:,m,k] = data2[:,m,k] + gfs[i] * np.einsum('ij,ij->i', factor1, factor2)
            data = data2.ravel()
            M2 = csr_matrix((data, (rows, cols)), shape=[n,n])

    if verify:
        if not np.all(np.round((M[342:350,342:350].toarray()),3) == np.round((M2[342:350,342:350].toarray()),3)):
            print("Error: methods give different results!")
            print(M[342:350,342:350].toarray())
            print("\n")
            print(M2[342:350,342:350].toarray())
            sys.exit()       
    return M if calcMethod == 1 else M2  

# integral rho * u * tf(u)
def massMatrix(field, rhos, region=[], elementDim=2, vectorized=True):
    Grads = field.shapeFunctionGradients()
    n = numberOfVertices()
    if isinstance(region, list) or type(region) is np.ndarray:
        elements = np.array(region)
    elif isinstance(region, Region):
        elements = region.getElements()
        rhos = rhos.getValues(region)
        elementDim = region.regionDimension
    else:
        print("Error: unsupported paramter!")
        sys.exit()
    if elementDim == 1:
        Mm = 1/6 * np.array([[2,1],
                            [1,2]])
        nBasis = 2
    elif elementDim == 2:
        nBasis = 3
        if False:
            Mm = 1/24 * np.array([[2,1,1],
                                [1,2,1],
                                [1,1,2]])
        else: 
            # precalculate mesh independant parts of the integral
            Mm = np.zeros((nBasis,nBasis))
            gfs,gps = gaussData(2, elementDim)
            for i,gp in enumerate(gps):
                for j in range(nBasis):
                    for k in range(nBasis):
                        Mm[j,k] = Mm[j,k] + gfs[i] * field.shapeFunctionValues(gp, elementDim)[j] * field.shapeFunctionValues(gp, elementDim)[k] 
    else:
        print("Error: this dimension is not implemented!")
        sys.exit()
    elementMatrixSize = nBasis**2        
    data = np.zeros(len(elements)*nBasis**2)    
    if elementDim == 1: # TODO: why can det(..) not be used here?
        detJacs = np.abs(np.linalg.norm( mesh()['xp'][elements[:,0]] - mesh()['xp'][elements[:,1]], axis=1))
    else:
        jacs = transformationJacobians(elements, elementDim=elementDim)
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
        opts = PETSc.Options()
        #opts.setValue("st_pc_factor_shift_type", "NONZERO")        
        n = len(b)   
        Ap = PETSc.Mat().createAIJ(size=(n, n),  csr=(A.indptr, A.indices, A.data))        
        Ap.setUp()
        Ap.assemblyBegin()
        Ap.assemblyEnd()
        bp = PETSc.Vec().createSeq(n) 
        bp.setValues(range(n),b)        
        up = PETSc.Vec().createSeq(n)        
        if True:
            ksp = PETSc.KSP().create()
            ksp.setOperators(Ap)        
            ksp.setFromOptions()
            #ksp.getPC().setFactorSolverType('mumps')
            ksp.getPC().setFactorSolverType('petsc')
            #ksp.setType('cg')  # conjugate gradient
            #ksp.setType('gmres')
            #ksp.getPC().setType('lu')
            # ksp.getPC().setType('cholesky') # cholesky
            #ksp.getPC().setType('icc') # incomplete cholesky
            print(f'Solving with: {ksp.getType():s}')
            ksp.solve(bp, up)
        else:
            snes = PETSc.SNES() 
            snes.create(PETSc.COMM_SELF)
            snes.setUseMF(True)
            snes.getKSP().setType('cg')
            snes.setFromOptions()
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
    assert(abs(max(np.linalg.norm(b,axis=1)) - 1.6104) < 1e-3)

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
    assert(abs(max(np.linalg.norm(b,axis=1)) - 3.3684) < 1e-3)

def runAll():
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
        exampleHMagnetOctant()
        exampleHMagnet()    
        exampleMagnetInRoom()  

    print('finished')

if __name__ == "__main__":
    main()