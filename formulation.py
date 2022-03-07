import numpy as np
import time
import sys
import pkg_resources
from scipy.sparse import *
from dofManager import countFreeDofs

from parameter import *
from region import Region
from field import *
from utils import *
from dofManager import *
from ioHelper import *

if 'petsc4py' in pkg_resources.working_set.by_key:
    hasPetsc = True
    import petsc4py
    petsc4py.init(sys.argv)        
    from petsc4py import PETSc
else:
    print("Warning: no petsc4py found, solving will be very slow!")
np.set_printoptions(linewidth=400)    

from mesh import getMesh
import spanningtree as st

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

# integral sigma * grad(dof(v)) * tf(a)
def matrix_gradDofV_tfA(fieldV, fieldA, sigmas, region):
    elementsV = fieldV.getElements(region = region)
    elementsA = fieldA.getElements(region = region)
    sigmas = sigmas.getValues(region)
    elementDim = region.regionDimension
    jacs = transformationJacobians(region, elementDim=elementDim)    
    detJacs = np.abs(np.linalg.det(jacs))
    invJacs = np.linalg.inv(jacs)
    signs = getSigns(region)
    if elementDim == 2:
        print("Error: hcurl elements are not possible in 2d!")
        sys.exit()
    elif elementDim == 3:
        nBasisV = 4
        nBasisA = 6
        rows = np.tile(elementsA, nBasisV).astype(np.int64).ravel()
        cols = np.repeat(elementsV, nBasisA).astype(np.int64).ravel()  
        data2 = np.zeros((len(elementsV), nBasisV, nBasisA))
        grads = fieldV.shapeFunctionGradients(elementDim)        
        integrationOrder = 2
        gfs, gps = gaussData(integrationOrder, elementDim)        
        for i in range(len(gfs)):
            values = fieldA.shapeFunctionValues(xi = gps[i], elementDim=elementDim)
            for m in range(nBasisV):
                for k in range(nBasisA):
                    factor1 = np.einsum('i,i,ikj,k->ij', sigmas, detJacs, invJacs, grads[m])
                    factor2 = np.einsum('i,ikj,k->ij', signs[:,k], invJacs, values[k,:])
                    data2[:,m,k] += gfs[i] * np.einsum('ij,ij->i', factor1, factor2)    
        data = data2.ravel(order='C')
    # delete all rows and cols with index -1
    idx = np.append(np.where(rows == -1)[0], np.where(cols == -1)[0])
    data = np.delete(data, idx)
    rows = np.delete(rows, idx)
    cols = np.delete(cols, idx)
    #
    numFreeDofs = countAllFreeDofs()
    K = csr_matrix((data, (rows, cols)), shape=[numFreeDofs, numFreeDofs]) 
    return K        

# integral sigma * dt(dof(a)) * tf(a)
def matrix_dtDofA_tfA(field1, field2, sigmas, region, frequency):
    elements1 = field1.getElements(region = region)
    elements2 = field2.getElements(region = region)
    sigmas = sigmas.getValues(region)
    elementDim = region.regionDimension
    jacs = transformationJacobians(region, elementDim=elementDim)    
    detJacs = np.abs(np.linalg.det(jacs))
    invJacs = np.linalg.inv(jacs)
    signs = getSigns(region)
    if elementDim == 2:
        print("Error: hcurl elements are not possible in 2d!")
        sys.exit()
    elif elementDim == 3:
        nBasis = 6
        rows = np.tile(elements2, nBasis).astype(np.int64).ravel()
        cols = np.repeat(elements1, nBasis).astype(np.int64).ravel()  
        data2 = np.zeros((len(elements1), nBasis, nBasis))
        integrationOrder = 2
        gfs, gps = gaussData(integrationOrder, elementDim)
        dt = 2*np.pi*frequency              
        for i in range(len(gfs)):
            values = field1.shapeFunctionValues(xi = gps[i], elementDim=elementDim)
            for m in range(nBasis):
                for k in range(nBasis):
                    factor1 = np.einsum('i,i,i,ikj,k->ij', signs[:,m], sigmas, detJacs, invJacs, values[m,:])
                    factor2 = np.einsum('i,ikj,k->ij', signs[:,k], invJacs, values[k,:])
                    data2[:,m,k] += gfs[i] * np.einsum('ij,ij->i', factor1, factor2)    
        data = dt * data2.ravel(order='C')
    # delete all rows and cols with index -1
    idx = np.append(np.where(rows == -1)[0], np.where(cols == -1)[0])
    data = np.delete(data, idx)
    rows = np.delete(rows, idx)
    cols = np.delete(cols, idx)
    #
    numFreeDofs = countAllFreeDofs()
    K = csr_matrix((data, (rows, cols)), shape=[numFreeDofs, numFreeDofs]) 
    return K

# integral sigma * dt(dof(a)) * grad(tf(v))
def matrix_dtDofA_gradTfV(fieldA, fieldV, sigmas, region, frequency):
    elementsV = fieldV.getElements(region = region)
    elementsA = fieldA.getElements(region = region)
    sigmas = sigmas.getValues(region)
    elementDim = region.regionDimension
    jacs = transformationJacobians(region, elementDim=elementDim)    
    detJacs = np.abs(np.linalg.det(jacs))
    invJacs = np.linalg.inv(jacs)
    signs = getSigns(region)
    if elementDim == 2:
        print("Error: hcurl elements are not possible in 2d!")
        sys.exit()
    elif elementDim == 3:
        nBasisV = 4
        nBasisA = 6
        rows = np.tile(elementsV, nBasisA).astype(np.int64).ravel()
        cols = np.repeat(elementsA, nBasisV).astype(np.int64).ravel()  
        data2 = np.zeros((len(elementsV), nBasisA, nBasisV))
        integrationOrder = 2
        grads = fieldV.shapeFunctionGradients(elementDim)        
        gfs, gps = gaussData(integrationOrder, elementDim)
        dt = 2*np.pi*frequency              
        for i in range(len(gfs)):
            values = fieldA.shapeFunctionValues(xi = gps[i], elementDim=elementDim)
            for m in range(nBasisA):
                for k in range(nBasisV):
                    factor1 = np.einsum('i,i,i,ikj,k->ij', signs[:,m], sigmas, detJacs, invJacs, values[m,:])
                    factor2 = np.einsum('ikj,k->ij', invJacs, grads[k])
                    data2[:,m,k] += dt * gfs[i] * np.einsum('ij,ij->i', factor1, factor2)    
        data = data2.ravel(order='C')
    # delete all rows and cols with index -1
    idx = np.append(np.where(rows == -1)[0], np.where(cols == -1)[0])
    data = np.delete(data, idx)
    rows = np.delete(rows, idx)
    cols = np.delete(cols, idx)
    #
    numFreeDofs = countAllFreeDofs()
    K = csr_matrix((data, (rows, cols)), shape=[numFreeDofs, numFreeDofs]) 
    return K            

# integral curl(u) * sigma * curl(tf(u)) 
def stiffnessMatrixCurl(field, sigmas, region=[], legacy=False):
    if region == []:
        elementDim = getMesh()['problemDimension'] 
        elements = field.getElements(dim = elementDim)
        jacs = transformationJacobians(elementDim=elementDim)
        signs = getMesh()['signs3d']
    else:
        elements = field.getElements(region = region)
        sigmas = sigmas.getValues(region)
        elementDim = region.regionDimension
        jacs = transformationJacobians(region, elementDim=elementDim)
        signs = getSigns(region)
    if elementDim == 2:    
        nBasis = 3
        elementArea = 1/2
    elif elementDim == 3:
        nBasis = 6
        elementArea = 1/6        
    detJacs = np.abs(np.linalg.det(jacs))
    if elementDim == 2:
        print("Error: hcurl elements are not possible in 2d!")
        sys.exit()
    elif elementDim == 3:
        rows = np.tile(elements, nBasis).astype(np.int64).ravel()
        cols = np.repeat(elements, nBasis).astype(np.int64).ravel()  
        if legacy:
            # this formulation might be a bit faster but only supports order 1!
            signsMultiplied = np.einsum('ij,ik->ijk', signs, signs) 
            curls = field.shapeFunctionCurls(elementDim)
            B = np.zeros((elementDim, elementDim, nBasis, nBasis))
            for i in range(3):
                for k in range(3):
                    B[i,k] = elementArea * np.matrix(curls[:,i]).T * np.matrix(curls[:,k]) 
            gammas = np.einsum('i,i,ikj,ikl->ijl', sigmas, 1/detJacs, jacs, jacs)
            data = np.einsum('ilm,ijk,jklm->ilm', signsMultiplied, gammas, B).ravel(order='C')
        else:
            # this formulation is more generic, because it supports higher orders
            data2 = np.zeros((len(elements), nBasis, nBasis))
            integrationOrder = 2
            gfs,gps = gaussData(integrationOrder, elementDim)              
            for i in range(len(gfs)):
                curls = field.shapeFunctionCurls(elementDim)
                for m in range(nBasis):
                    for k in range(nBasis):
                        factor1 = np.einsum('i,i,i,ijk,k->ij', signs[:,m], sigmas, 1/detJacs, jacs, curls[m,:])
                        factor2 = np.einsum('i,ijk,k->ij', signs[:,k], jacs, curls[k,:])
                        data2[:,m,k] += gfs[i] * np.einsum('ij,ij->i', factor1, factor2)    
            data = data2.ravel(order='C')
    # delete all rows and cols with index -1
    idx = np.append(np.where(rows == -1)[0], np.where(cols == -1)[0])
    data = np.delete(data, idx)
    rows = np.delete(rows, idx)
    cols = np.delete(cols, idx)
    #
    numFreeDofs = countAllFreeDofs()
    K = csr_matrix((data, (rows, cols)), shape=[numFreeDofs,numFreeDofs]) 
    return K

# integral grad(u) * sigma * grad(tf(u)) 
def stiffnessMatrix(field, sigmas, region=[], vectorized=True, legacy=False):
    if region == []:
        elementDim = 2
        elements = field.getElements(dim = elementDim)
        jacs = transformationJacobians(elementDim=elementDim)
    else:
        elements = field.getElements(region = region)
        elementDim = region.regionDimension
        sigmas = sigmas.getValues(region)
        jacs = transformationJacobians(region, elementDim=elementDim)
    if elementDim == 2:
        nBasis = 3
        area = 1/2
    elif elementDim == 3:
        nBasis = 4
        area = 1/6        
    numElements = len(elements)
    Grads = field.shapeFunctionGradients(elementDim)
    elementMatrixSize = (getMesh()['problemDimension']+1)**2

    data = np.zeros(numElements*elementMatrixSize)    
    rows = np.tile(elements, nBasis).astype(np.int64).ravel()
    cols = np.repeat(elements,nBasis).astype(np.int64).ravel()       
    detJacs = np.abs(np.linalg.det(jacs))    
    invJacs = np.linalg.inv(jacs)       

    if not legacy:  
        # precalculate mesh independant parts of the integral
        B = np.zeros((elementDim, elementDim, nBasis, nBasis))
        for i in range(elementDim):
            for k in range(elementDim):
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
            detJacsDuplicated = np.repeat(detJacs, elementDim**2).reshape((numElements, elementDim, elementDim))
            gammas = np.zeros((numElements,elementDim,elementDim))
            for elementIndex in range(numElements):        # TODO: vectorize this        
                gammas[elementIndex] = sigmas[elementIndex] @ invJacs[elementIndex] @ np.swapaxes(invJacs,1,2)[elementIndex] * detJacsDuplicated[elementIndex]
        if vectorized:
            data = np.einsum('ijk,jklm', gammas, B).ravel(order='C')
        else:
            for elementIndex in range(numElements):
                indexRange = np.arange(start=elementIndex*elementMatrixSize, stop=elementIndex*elementMatrixSize+elementMatrixSize)            
                data[indexRange] = np.einsum('jk,jk...', gammas[elementIndex],B).ravel()
    
    else:  # LEGACY CODE       
        if getMesh()['problemDimension'] == 3:
            # precalculate mesh independant parts of the integral
            B = np.zeros((elementDim, elementDim, nBasis, nBasis))
            for i in range(elementDim):
                for k in range(elementDim):
                    B[i,k] = area * np.matrix(Grads[:,i]).T * np.matrix(Grads[:,k]) 
            
            sigmasDuplicated = np.repeat(sigmas, elementDim**2).reshape((numElements, elementDim, elementDim))
            detJacsDuplicated = np.repeat(detJacs, elementDim**2).reshape((numElements, elementDim, elementDim))
            gammas = sigmasDuplicated * invJacs @ np.swapaxes(invJacs,1,2) * detJacsDuplicated   
            for elementIndex, element in enumerate(elements):
                indexRange = np.arange(start=elementIndex*elementMatrixSize, stop=elementIndex*elementMatrixSize+elementMatrixSize)            
                data[indexRange] = np.einsum('jk,jk...', gammas[elementIndex],B).ravel() # this is generic and faster than explicit summation like below

        if getMesh()['problemDimension'] == 2:      
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
                data[indexRange] = (gamma11*B_11 + gamma12*B_12 + gamma21*B_21 + gamma22*B_22).ravel()
                #K_Ts[triangleIndex] = gamma1*B_11 + gamma2*B_12 + gamma3*B_21 + gamma4*B_22
                #K[np.ix_(triangle[:],triangle[:])] = K[np.ix_(triangle[:],triangle[:])] + K_T      
    n = countAllFreeDofs()
    # delete all rows and cols with index -1
    idx = np.append(np.where(rows == -1)[0], np.where(cols == -1)[0])
    data = np.delete(data, idx)
    rows = np.delete(rows, idx)
    cols = np.delete(cols, idx)
    #    
    K = csr_matrix((data, (rows, cols)), shape=[n,n]) 
    return K

# integral br * rot(tf(u))
def fluxRhsCurl(field, br, region=[], vectorized=True):
    if region == []:
        elements = getMesh()['pt']
        elementDim = getMesh()['problemDimension']         
        elements = field.getElements(dim = elementDim)        
        jacs = transformationJacobians([], elementDim)
    else:
        elements = field.getElements(region = region)        
        elementDim = region.regionDimension        
        jacs = transformationJacobians(region, elementDim)
        br = br.getValues(region)
    if elementDim == 2:
        area = 1/2
        nBasis = 3
        signs = getMesh()['signs2d']
    else:
        area = 1/6
        nBasis = 6
        signs = getMesh()['signs3d']
    curls = field.shapeFunctionCurls(elementDim)    

    numAllDofs = len(elements)
    temp = area* np.einsum('ik,ijk->ijk', signs, np.einsum('ijk,lk', jacs, curls))  # TODO is detJac needed here??
    rows = elements.ravel(order='C')
    data = np.zeros((numAllDofs,nBasis))
    for basis in range(nBasis):
        data[:,basis] = np.einsum('ij,ij->i', br, temp[:,:,basis])
    data = data.ravel(order='C')        
    n = countAllFreeDofs()
    # delete all rows and cols with index -1
    idx = np.where(rows == -1)[0]
    rows = np.delete(rows, idx)
    data = np.delete(data, idx)
    #
    rhs = csr_matrix((data, (rows,np.zeros(len(rows)))), shape=[n,1]).toarray().ravel()
    return rhs

# integral j * tf(u)
# j contains a constant value for each element
def loadRhs(field, j, region=[], vectorized=True):
    if region == []:
        elementDim = getMesh()['problemDimension']             
        elements = field.getElements(dim = elementDim)        
        jacs = transformationJacobians([], elementDim)
    else:
        elements = field.getElements(region = region)        
        elementDim = region.regionDimension           
        jacs = transformationJacobians(region, elementDim)
        j = j.getValues(region)
    if elementDim == 2:
        nBasis = 3
        if field.isEdgeField():
            signs = getMesh()['signs2d']
    else:
        nBasis = 6
        if field.isEdgeField():
            signs = getMesh()['signs3d']
    integrationOrder = 2
    gfs,gps = gaussData(integrationOrder, elementDim)        
    invJacs = np.linalg.inv(jacs)     
    detJacs = np.abs(np.linalg.det(jacs))

    data = np.zeros((len(elements),nBasis))
    rows = elements.ravel(order='C')
    
    # here only the rotation part of the affine transformation need to be considered for j
    # because j is already given for every element
    jTransformed = np.einsum('ijk,ik->ij', invJacs, j)

    for i, gp in enumerate(gps):
        for m in range(nBasis):
            if field.isEdgeField():
                data[:,m] += gfs[i] * np.einsum('i,ij,i,j->i', detJacs, jTransformed, signs[:,m], field.shapeFunctionValues(gp, elementDim)[m,:])
            else:
                data[:,m] += gfs[i] * np.einsum('i,ij,j->i', detJacs, jTransformed, field.shapeFunctionValues(gp, elementDim)[m,:])
    data = data.ravel(order='C')
    n = countAllFreeDofs()
    # delete all rows and cols with index -1
    idx = np.where(rows == -1)[0]
    rows = np.delete(rows, idx)
    data = np.delete(data, idx)
    #
    rhs = csr_matrix((data, (rows,np.zeros(len(rows)))), shape=[n,1]).toarray().ravel()
    return rhs    

# integral br * grad(tf(u))
def fluxRhs(field, br, region=[], vectorized=True):
    if region == []:
        elementDim = 2
        elements = field.getElements(dim = elementDim)
    else:
        elements = field.getElements(region = region)
        elementDim = region.regionDimension                 
        br = br.getValues(region)
    if elementDim == 2:
        zero = [0, 0]
        area = 1/2
        nBasis = 3
        dim = 2
    else:
        zero = [0, 0, 0]
        area = 1/6
        nBasis = 4
        dim = 3
    Grads = field.shapeFunctionGradients(dim)    
    jacs = transformationJacobians(region, elementDim = dim)
    detJacs = np.abs(np.linalg.det(jacs))    
    invJacs = np.linalg.inv(jacs) 
    n = countAllFreeDofs()    
    if vectorized:
        # this line has convergence issues with example magnet_in_room
        # temp2 = area * np.einsum('i,ikj,lk->ijl', detJacs, invJacs, Grads)
        # this line has no convergence issues - strange!
        # temp2 = area* np.repeat(detJacs,nCoords*nBasis).reshape((len(detJacs),nCoords,nBasis))* np.einsum('ikj,lk', invJacs, Grads)        
        # this line is also ok, the order of multiplication is probably favorable with regard to rounding errors
        temp2 = area* np.einsum('i,i...->i...', detJacs, np.einsum('ikj,lk', invJacs, Grads))
        rows = elements.ravel(order='C')
        data = np.zeros((len(elements),nBasis))
        for basis in range(nBasis):
            data[:,basis] = np.einsum('ij,ij->i', br, temp2[:,:,basis])
        data = data.ravel(order='C')
        # delete all rows and cols with index -1
        idx = np.where(rows == -1)[0]
        rows = np.delete(rows, idx)
        data = np.delete(data, idx)
        #            
        rhs = csr_matrix((data, (rows,np.zeros(len(rows)))), shape=[n,1]).toarray().ravel()
    else:
        rhs = np.zeros(n)
        for elementIndex, element in enumerate(elements):
            if np.array_equal(br[elementIndex], zero): # just for speedup
                continue
            temp = area * invJacs[elementIndex].T @ Grads.T * detJacs[elementIndex]
            for basis in range(nBasis):
                if element[basis] != -1:
                    rhs[element[basis]] = rhs[element[basis]] + np.dot(br[elementIndex], temp.T[basis])
    return rhs

# integral rho * u * tf(u)
def massMatrixCurl(field, rhos, region=[], elementDim=2, verify=False):
    if isinstance(region, list) or type(region) is np.ndarray:
        elements = region  # TODO: Dangerous! this does not work with constraints or multiple fields!
        elementDim = getMesh['problemDimension']
        jacs = transformationJacobians(elementDim=elementDim)
    elif isinstance(region, Region):
        elements = field.getElements(region = region)
        rhos = rhos.getValues(region)
        elementDim = region.regionDimension
        jacs = transformationJacobians(region, elementDim=elementDim)
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
    n = countAllFreeDofs()        
    if elementDim == 2:
        detJacs = np.abs(np.linalg.det(jacs))    
        invJacs = np.linalg.inv(jacs)       
        signs = getMesh()['signs2d'] 
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
            # delete all rows and cols with index -1
            idx = np.where(rows == -1)[0]
            rows = np.delete(rows, idx)
            data = np.delete(data, idx)
            #            
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
            # delete all rows and cols with index -1
            idx = np.where(rows == -1)[0]
            rows = np.delete(rows, idx)
            data = np.delete(data, idx)
            #
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
    n = countAllFreeDofs()
    if isinstance(region, list) or type(region) is np.ndarray:
        elements = np.array(region) # TODO: Dangerous! this does not work with constraints or multiple fields!
        jacs = transformationJacobians(elements, elementDim=elementDim)        
    elif isinstance(region, Region):
        elements = field.getElements(region = region)
        rhos = rhos.getValues(region)
        elementDim = region.regionDimension
        jacs = transformationJacobians(region, elementDim=elementDim)        
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
                        Mm[j,k] += gfs[i] * field.shapeFunctionValues(gp, elementDim)[j] * field.shapeFunctionValues(gp, elementDim)[k] 
    else:
        print("Error: this dimension is not implemented!")
        sys.exit()
    elementMatrixSize = nBasis**2        
    data = np.zeros(len(elements)*nBasis**2)    
    detJacs = np.abs(np.linalg.det(jacs))        
    rows = np.tile(elements, nBasis).astype(np.int64).ravel()
    cols = np.repeat(elements,nBasis).astype(np.int64).ravel()
    if vectorized:
        data = np.einsum('i,jk', rhos*detJacs, Mm).ravel()
    else:
        for elementIndex, element in enumerate(elements):
            rangeIndex = np.arange(start=elementIndex*elementMatrixSize, stop=elementIndex*elementMatrixSize+elementMatrixSize)
            data[rangeIndex] = (rhos[elementIndex]*detJacs[elementIndex]*Mm).ravel()
            #M_T = rhos[triangleIndex]*detJac*Mm
            #M[np.ix_(triangle[:],triangle[:])] = M[np.ix_(triangle[:],triangle[:])] + M_T
    # delete all rows and cols with index -1
    idx = np.where(rows == -1)
    rows = np.delete(rows, idx)
    data = np.delete(data, idx)
    #            
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
        elements = getMesh()['pe'][getMesh()['eb']]
    else:
        elements = field.getElements(region = region)
        alphas = alphas.getValues(region)
    for elementIndex, element in enumerate(elements):
        detJac = np.abs(np.linalg.norm(getMesh()['xp'][element[0]] - getMesh()['xp'][element[1]]))
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
            #ksp.getPC().setType('cholesky') # cholesky
            #ksp.getPC().setType('icc') # incomplete cholesky
            print(f'Solving {n} dofs with: {ksp.getType():s}')
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
    numDofs = len(u)
    assert numDofs == countAllFreeDofs()
    putSolutionIntoFields(u)
    stop = time.time()
    print(f"{bcolors.OKGREEN}solved {numDofs} dofs in {stop - start:.2f} s{bcolors.ENDC}")    