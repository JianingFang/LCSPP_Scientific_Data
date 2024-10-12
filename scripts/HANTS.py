import numpy as np

"""
HANTS(ni,nb,nf,y,ts,HiLo,low,high,fet,dod,delta)

Modified: 
Apply suppression of high amplitudes for near-singular case by 
adding a number delta to the diagonal elements of matrix A, 
except element (1,1), because the average should not be affected

Output of reconstructed time series in array yr June 2005

Change call and input arguments to accommodate a base period length (nb)
All frequencies from 1 (base period) until nf are included

Inputs:
    ni    = nr. of images (total number of actual samples of the time 
           series)
    nb    = length of the base period, measured in virtual samples 
           (days, dekads, months, etc.)
    nf    = number of frequencies to be considered above the zero frequency
    y     = array of input sample values (e.g. NDVI values)
    ts    = array of size ni of time sample indicators 
           (indicates virtual sample number relative to the base period); 
           numbers in array ts maybe greater than nb
           If no aux file is used (no time samples), we assume ts(i)= i, 
           where i=1, ..., ni
    HiLo  = 2-character string indicating rejection of high or low outliers
    low   = valid range minimum
    high  = valid range maximum (values outside the valid range are rejeced
           right away)
    fet   = fit error tolerance (points deviating more than fet from curve 
           fit are rejected)
    dod   = degree of overdeterminedness (iteration stops if number of 
           points reaches the minimum required for curve fitting, plus 
           dod). This is a safety measure
    delta = small positive number (e.g. 0.1) to suppress high amplitudes

Outputs:

    amp   = returned array of amplitudes, first element is the average of 
         the curve
    phi   = returned array of phases, first element is zero
    yr	= array holding reconstructed time series

Original Author: Wout Verhoef
NLR, Remote Sensing Dept.
June 1998

Converted to MATLAB:
Mohammad Abouali (2011)

Converted to Python
Jianing Fang (2023), jf3423@columbia.edu

"""

def HANTS(ni,nb,nf,y,ts,HiLo,low,high,fet,dod,delta):

 
    mat = np.zeros((np.int64(min(2*nf+1,ni)),ni),dtype=np.float32)
    amp = np.zeros((np.int64(nf+1),1), dtype=np.float32)
    phi =np.zeros((np.int64(nf+1),1), dtype=np.float32)
    yr = np.zeros((np.int64(ni), 1), dtype=np.float64)

    sHiLo = 0;
    if HiLo == "Hi":
        sHiLo =-1
    if HiLo == "Lo":
        sHiLo = 1

    nr = min(2 * nf + 1, ni)
    noutmax = ni - nr - dod
    dg = 180.0 / np.pi
    mat[0,:] = 1.0

    ang = 2 * np.pi * np.arange(np.int64(nb)) / nb 
    cs = np.cos(ang)
    sn = np.sin(ang)

    i = np.arange(1, np.int64(nf+1))
    for j in np.arange(1, np.int64(ni)+1):
        index = (1 + (i * (ts[j-1]-1)) % nb).astype(np.int64)
        mat[2*i-1, j-1]=cs[index-1]
        mat[2*i, j-1]=sn[index-1]
    p = np.invert((y < low) | (y > high)).astype(np.int64)
    nout = np.sum((y < low) | (y > high))
    if nout > noutmax:
        return amp, phi, yr 

    ready = False
    nloop=0
    nloopmax=ni

    while (not ready and nloop<nloopmax):
        nloop = nloop+1
        za= mat @ (y * p)

        A = mat @ np.diag(p) @ mat.T
        A = A + np.diag(np.full(np.int64(nr), 1)) * delta
        A[0,0] = A[0, 0] - delta
        zr, _, _, _ = np.linalg.lstsq(A, za, rcond=None)

        yr = mat.T @ zr
        diffVec = sHiLo * (yr-y);
        err =p * diffVec;

        rankVec = np.argsort(err)

        maxerr=diffVec[rankVec[ni-1]]
        ready=(maxerr<=fet) or (nout==noutmax)

        if not ready:
            i = ni
            j = rankVec[i-1]
            while p[j-1] *diffVec[j-1] > maxerr * 0.5 and  nout < noutmax: 
                    p[j-1] = 0 
                    nout = nout+1 
                    i = i - 1
                    j = rank[i-1]

    amp[0] = zr[0]
    phi[0] =0.0
    zr_org = zr
    zr = np.zeros(ni)
    zr[:len(zr_org)]=zr_org
    
    i = np.arange(2, nr+1, 2)
    ifr= (i+2) / 2 
    ra=zr[np.int64(i-1)]
    rb=zr[np.int64(i)]
    amp[np.int64(ifr)-1, 0]=np.sqrt(ra*ra+rb*rb)
    phase=np.arctan2(rb,ra)*dg
    phase[phase < 0] = phase[phase<0] + 360 
    phi[np.int64(ifr)-1, 0] = phase 

    return amp, phi, yr 