import numpy as np

def homography2d(x1, x2):
    """
    Direct Linear Transform 
    
    Input:
    x1: 3xN set of homogeneous points
    x2: 3xN set of homogeneous points such that x1<->x2

    Returns:
    H: the 3x3 homography such that x2 = H*x1
    """
    [x1, T1] = normalise2dpts(x1)
    [x2, T2] = normalise2dpts(x2)

    Npts = x1.shape[1]

    A = np.zeros((3*Npts,9))
    
    O = np.zeros(3);

    for i in range(0, Npts):
        X = x1[:,i].T
        x = x2[0,i]
        y = x2[1,i]
        w = x2[2,i]
        A[3*i,:] = np.array([O, -w*X, y*X]).reshape(1, 9)
        A[3*i+1,:] = np.array([w*X, O, -x*X]).reshape(1, 9)
        A[3*i+2,:] = np.array([-y*X, x*X, O]).reshape(1, 9)

    [U,D,Vh] = np.linalg.svd(A)
    V = Vh.T

    H = V[:,8].reshape(3,3)

    H = np.linalg.solve(T2, H)
    H = H.dot(T1)

    return H
    
    
def normalise2dpts(pts):
    """
    Function translates and normalises a set of 2D homogeneous points 
    so that their centroid is at the origin and their mean distance from 
    the origin is sqrt(2).  This process typically improves the
    conditioning of any equations used to solve homographies, fundamental
    matrices etc.
       
       
    Inputs:
    pts: 3xN array of 2D homogeneous coordinates
   
    Returns:
    newpts: 3xN array of transformed 2D homogeneous coordinates.  The
            scaling parameter is normalised to 1 unless the point is at
            infinity. 
    T: The 3x3 transformation matrix, newpts = T*pts
    """
    if pts.shape[0] != 3:
        print("Input shoud be 3")

    finiteind = np.nonzero(abs(pts[2,:]) > np.spacing(1));
    
    if len(finiteind) != pts.shape[1]:
        print('Some points are at infinity')
    
    dist = []
    for i in finiteind:
        pts[0,i] = pts[0,i]/pts[2,i]
        pts[1,i] = pts[1,i]/pts[2,i]
        pts[2,i] = 1;

        c = np.mean(pts[0:2,i].T, axis=0).T          

        newp1 = pts[0,i]-c[0]
        newp2 = pts[1,i]-c[1]
    
    	dist.append(np.sqrt(newp1**2 + newp2**2))

    meandist = np.mean(dist[:])
    
    scale = np.sqrt(2)/meandist
    
    T = np.array([[scale, 0, -scale*c[0]], [0, scale, -scale*c[1]], [0, 0, 1]])
    
    newpts = T.dot(pts)

    return [newpts, T]
