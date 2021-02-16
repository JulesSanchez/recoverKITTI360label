import numpy as np

'''
 * @brief SparsifyData Sparsify a point cloud
 * @param M Input Point cloud
 * @param dim Dimemsion of points
 * @param min_dist Minimum distance for sparsifying
 * @param out_dist Distance for removing outliers
 * @param idx_start Start sparsifying from the specific point

 * @out idx_sparse Output binary masks indicating which point should be preserved
 * @out idx_size Output number of points to be preserved
'''

#NOT IMPLEMENTED
def sparsifyData(M, dim, min_dist, out_dist, idx_start):
    idx_sparse = []
    idx_size = 0
    return idx_sparse, idx_size

'''
 * @brief ExtractCols Extract columns of a matrix given indices
 * @param input Input matrix
 * @param idx Indices of columns to extract

 * @out output Output matrix
'''

def extractColsSparse(ipt, output, idx_sparse, idx_size):
    dim = ipt.shape[0]
    output = output.reshape(dim, idx_size)
    k = 0
    for i in range(ipt.shape[1]):
        if idx_sparse[i]:
            for c in range(dim):
                output[c, k] = ipt[c, i]
            k += 1
    return output


'''
 * @brief ExtractCols Extract columns of a matrix given indices
 * @param input Input matrix
 * @param idx Indices of columns to extract

 * @out output Output matrix
'''

def extractCols(ipt, output, idx):
    dim = ipt.shape[0]
    output = np.zeros((dim, len(idx)))
    for i in range(len(idx)):
        for c in range(dim):
            output[c, i] = ipt[c, idx[i]]
    return output

'''
 * @brief ExtractCols Extract columns of a matrix given indices
 * @param input Input matrix
 * @param idx Indices of rows to extract

 * @out output Output matrix
'''
def extractRows(ipt, output, idx):
    dim = ipt.shape[1]
    output = np.zeros((len(idx), dim))
    for i in range(len(idx)):
        for c in range(dim):
            output[i,c] = ipt[idx[i],c]
    return output

'''
 *@brief RemoveBlindSpot Remove the points within a sector
 *@param matIn Input point cloud
 *@param blind_splot_angle Sector angle

 *@out matOut Output point cloud
'''
def removeBlindSpots(matIn, blind_splot_angle):
    if blind_splot_angle <= 0:
        return matIn, np.arange(0,matIn.shape[0])

    kept_indices = []
    
    matInSub = matIn[:,0].reshape(-1)
    v = - matInSub/np.linalg.norm(matInSub)

    angle = np.cos(blind_splot_angle/2)

    boolMask = v < angle
    returnedMat = matIn[boolMask,:]
    kept_indices = list(np.arange(0,matIn.shape[0])[boolMask])

    return returnedMat, kept_indices

''' 
 *@brief CropVelodyneData Crop velodyne data given max and min distance
 *@param matIn Input point cloud
 *@param minDist Minimum distance to crop data
 *@param maxDist Maximum distance to crop data

 *@out matOut Output point cloud
''' 

def cropVelodyneData(matIn, minDist, maxDist):
    matNorm = np.linalg.norm(matIn,axis=1)
    
    k=0
    matOut = np.zeros_like(matIn)
    indices = []

    for i in range(matIn.shape[0]):    
        if matNorm[i] > minDist and matNorm[i] < maxDist:
            matOut[k] = matIn[i]
            indices.append(i)
            k += 1
    
    return matOut[:k], indices

''' 
 *@brief CropVelodyneData Crop velodyne data given max and min distance
 *@param matIn Input point cloud
 *@param minDist Minimum distance to crop data
 *@param maxDist Maximum distance to crop data

 *@out matOut Output indices of preserved points
''' 

def cropVelodyneDataIndices(matIn, minDist, maxDist):
    matNorm = np.linalg.norm(matIn,axis=1)
    
    idxOut = []

    for i in range(matIn.shape[0]):    
        if matNorm[i] > minDist and matNorm[i] < maxDist:
            idxOut.append(i)
    
    return idxOut

'''
 *@brief CurlVelodyneData Transform velodyne data given curl parameters
 *@param velo_in Input velodyne data
 *@param r Rotation matrix for curl
 *@param t Translation vector for curl

 *@out velo_out Output velodyne data
'''

def curlVelodyneData(velo_in, r, t):
    pt_num = velo_in.shape[0]
    velo_out = np.zeros_like(velo_in)
    s = np.arctan2(velo_in[:,1],velo_in[:,2])/np.pi 
    rs = s[:,None] * np.repeat(r.reshape(1,-1),len(s),axis=0)
    theta = np.sqrt(np.sum(rs**2,axis=1))
    theta_pos = theta > 1e-10
    theta_neg = theta < 1e-10
    k = rs/(theta[:,None]+1e-11)
    ct = np.cos(theta)
    st = np.sin(theta) 
    kv = np.sum(k*velo_in,axis=1)
    velo_out[theta_pos,0] = ct[theta_pos]*velo_in[theta_pos,0] + (k[theta_pos,1]*velo_in[theta_pos,2] - k[theta_pos,2]*velo_in[theta_pos,1])*st[theta_pos] + (1-ct[theta_pos])*k[theta_pos,0]*kv[theta_pos]+t[0]*theta_pos[theta_pos]
    velo_out[theta_pos,1] = ct[theta_pos]*velo_in[theta_pos,1] + (k[theta_pos,2]*velo_in[theta_pos,0] - k[theta_pos,0]*velo_in[theta_pos,2])*st[theta_pos] + (1-ct[theta_pos])*k[theta_pos,1]*kv[theta_pos]+t[1]*theta_pos[theta_pos]
    velo_out[theta_pos,2] = ct[theta_pos]*velo_in[theta_pos,2] + (k[theta_pos,0]*velo_in[theta_pos,1] - k[theta_pos,1]*velo_in[theta_pos,0])*st[theta_pos] + (1-ct[theta_pos])*k[theta_pos,2]*kv[theta_pos]+t[2]*theta_pos[theta_pos]

    velo_out[theta_neg] = velo_in[theta_neg] + theta_neg[theta_neg][:,None]*t[:3]

    return velo_out

'''
 *@brief GetHalfPoints Get either forward or backward points
 *@param matIn Input matrix
 *@param direction Bool value indicating forward (true) or backward (false) 

 *@out matOut Output matrix
'''
def getHalfPoints(matIn, direction):
    k=0 
    matOut = np.zeros_like(matIn)
    for i in range(matIn.shape[1]):
        if (direction and matIn[0,i]>0) or (not direction and matIn[0,i]<0):
            matOut[:,k] = matIn[:,i]
            k += 1
    return matOut[:,:k]

