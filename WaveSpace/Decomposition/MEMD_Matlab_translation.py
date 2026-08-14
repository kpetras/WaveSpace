# -*- coding: utf-8 -*-
"""
Acknowledgment: All of this code is based on the multivariate EMD code, publicly available from
                http://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm.

                the original translation into python code is by Mario de Souza e Silva. 
                original can be found here: https://github.com/mariogrune/MEMD-Python-/tree/master

                the code below is a modified version of the original translation, with some adaptations


[1]  Rehman and D. P. Mandic, "Multivariate Empirical Mode Decomposition", Proceedings of the Royal Society A, 2010
[2]  G. Rilling, P. Flandrin and P. Goncalves, "On Empirical Mode Decomposition and its Algorithms", Proc of the IEEE-EURASIP
     Workshop on Nonlinear Signal and Image Processing, NSIP-03, Grado (I), June 2003
[3]  N. E. Huang et al., "A confidence limit for the Empirical Mode Decomposition and Hilbert spectral analysis",
     Proceedings of the Royal Society A, Vol. 459, pp. 2317-2345, 2003

"""

import numpy as np
from scipy.interpolate import interp1d,CubicSpline
from math import pi,sqrt,sin,cos
import warnings
import sys
import numba
from scipy.stats.qmc import Halton

def zero_crossings(signal):
    '''find indices where the signal x crosses zero (criterion for IMF is numbre of zero crossing = number of extrema +-1)'''
    zeroCrossIndices = np.where(signal[:-1] * signal[1:] < 0)[0]

    if any(signal == 0):
        zeroIndices = np.where(signal == 0)[0]
        if any(np.diff(zeroIndices) == 1):
            zeroMask = signal == 0
            zeroDiff = np.diff([0, *zeroMask, 0])
            startIndices = np.where(zeroDiff == 1)[0]
            endIndices = np.where(zeroDiff == -1)[0] - 1
            midIndices = np.round((startIndices + endIndices) / 2)
        else:
            midIndices = zeroIndices
        zeroCrossIndices = np.sort(np.concatenate((zeroCrossIndices, midIndices)))

    return zeroCrossIndices



def boundary_conditions(minInds, maxInds, time, signal, multivariateSignal, numSymmetries):
    '''defines new extrema points to extend  interpolations at signal edges (mainly mirror symmetry)'''
    signalLength = len(signal) - 1
    maxEnd = len(maxInds) - 1
    minEnd = len(minInds) - 1
    minInds = minInds.astype(int)
    maxInds = maxInds.astype(int)
    # If tnot enough extrema, return mode = 0
    if len(minInds) + len(maxInds) < 3:
        return None, None, None, None, 0

    mode = 1 

    # left boundary
    if maxInds[0] < minInds[0]:
        if signal[0] > signal[minInds[0]]:
            leftMax = np.flipud(maxInds[1:min(maxEnd + 1, numSymmetries + 1)])
            leftMin = np.flipud(minInds[:min(minEnd + 1, numSymmetries)])
            leftSymmetry = maxInds[0]
        else:
            leftMax = np.flipud(maxInds[:min(maxEnd + 1, numSymmetries)])
            leftMin = np.concatenate((np.flipud(minInds[:min(minEnd + 1, numSymmetries - 1)]), [0]))
            leftSymmetry = 0
    else:
        if signal[0] < signal[maxInds[0]]:
            leftMax = np.flipud(maxInds[:min(maxEnd + 1, numSymmetries)])
            leftMin = np.flipud(minInds[1:min(minEnd + 1, numSymmetries + 1)])
            leftSymmetry = minInds[0]
        else:
            leftMax = np.concatenate((np.flipud(maxInds[:min(maxEnd + 1, numSymmetries - 1)]), [0]))
            leftMin = np.flipud(minInds[:min(minEnd + 1, numSymmetries)])
            leftSymmetry = 0

    #right boundary
    if maxInds[-1] < minInds[-1]:
        if signal[-1] < signal[maxInds[-1]]:
            rightMax = np.flipud(maxInds[max(maxEnd - numSymmetries + 1, 0):])
            rightMin = np.flipud(minInds[max(minEnd - numSymmetries, 0):-1])
            rightSymmetry = minInds[-1]
        else:
            rightMax = np.concatenate(([signalLength], np.flipud(maxInds[max(maxEnd - numSymmetries + 2, 0):])))
            rightMin = np.flipud(minInds[max(minEnd - numSymmetries + 1, 0):])
            rightSymmetry = signalLength
    else:
        if signal[-1] > signal[minInds[-1]]:
            rightMax = np.flipud(maxInds[max(maxEnd - numSymmetries, 0):-1])
            rightMin = np.flipud(minInds[max(minEnd - numSymmetries + 1, 0):])
            rightSymmetry = maxInds[-1]
        else:
            rightMax = np.flipud(maxInds[max(maxEnd - numSymmetries + 1, 0):])
            rightMin = np.concatenate(([signalLength], np.flipud(minInds[max(minEnd - numSymmetries + 2, 0):])))
            rightSymmetry = signalLength

    leftTimeMin = 2 * time[leftSymmetry] - time[leftMin]
    leftTimeMax = 2 * time[leftSymmetry] - time[leftMax]
    rightTimeMin = 2 * time[rightSymmetry] - time[rightMin]
    rightTimeMax = 2 * time[rightSymmetry] - time[rightMax]

    if leftTimeMin[0] > time[0] or leftTimeMax[0] > time[0]:
        if leftSymmetry == maxInds[0]:
            leftMax = np.flipud(maxInds[:min(maxEnd + 1, numSymmetries)])
        else:
            leftMin = np.flipud(minInds[:min(minEnd + 1, numSymmetries)])
        if leftSymmetry == 1:
            sys.exit('Bug detected in left boundary extension')
        leftSymmetry = 0
        leftTimeMin = 2 * time[leftSymmetry] - time[leftMin]
        leftTimeMax = 2 * time[leftSymmetry] - time[leftMax]

    if rightTimeMin[-1] < time[signalLength] or rightTimeMax[-1] < time[signalLength]:
        if rightSymmetry == maxInds[-1]:
            rightMax = np.flipud(maxInds[max(maxEnd - numSymmetries + 1, 0):])
        else:
            rightMin = np.flipud(minInds[max(minEnd - numSymmetries + 1, 0):])
        if rightSymmetry == signalLength:
            sys.exit('Bug detected in right boundary extension')
        rightSymmetry = signalLength
        rightTimeMin = 2 * time[rightSymmetry] - time[rightMin]
        rightTimeMax = 2 * time[rightSymmetry] - time[rightMax]

    leftSignalMax = multivariateSignal[leftMax, :]
    leftSignalMin = multivariateSignal[leftMin, :]
    rightSignalMax = multivariateSignal[rightMax, :]
    rightSignalMin = multivariateSignal[rightMin, :]

    extendedTimeMin = np.hstack((leftTimeMin, time[minInds], rightTimeMin))
    extendedTimeMax = np.hstack((leftTimeMax, time[maxInds], rightTimeMax))
    extendedSignalMin = np.vstack((leftSignalMin, multivariateSignal[minInds, :], rightSignalMin))
    extendedSignalMax = np.vstack((leftSignalMax, multivariateSignal[maxInds, :], rightSignalMax))

    return extendedTimeMin, extendedTimeMax, extendedSignalMin, extendedSignalMax, mode

@numba.jit(nopython=True)
def make_dir_vectors(seq, it, N_dim, dir_vec):
    # Linear normalization of hammersley sequence in the range of -1.00 - 1.00
    b = 2 * seq[it, :] - 1

    # Find angles corresponding to the normalized sequence
    tht = np.arctan2(np.sqrt(np.flipud(np.cumsum(b[:0:-1]**2))), b[:N_dim-1]).transpose()

    # Find coordinates of unit direction vectors on n-sphere.
    # Use running product to avoid repeated np.prod calls in a hot loop.
    dir_vec[0] = 1.0
    cos_prod = 1.0
    for i in range(1, N_dim):
        dir_vec[i] = np.sin(tht[i-1]) * cos_prod
        cos_prod *= np.cos(tht[i-1])

    return dir_vec

# computes the mean of the envelopes and the mode amplitude estimate
def envelope_mean(signal, time, seq, numDirs, numSamples, numDims): #new
    numSymmetries = 2
    insufficientExtremaCount = 0

    envelopeMean = np.zeros((len(time), numDims))
    amplitude = np.zeros(len(time))
    numExtrema = np.zeros(numDirs)
    numZeroCrossings = np.zeros(numDirs)

    directionVector = np.zeros((numDims, 1))

    for directionIndex in range(numDirs):
        # Generate direction vectors
        directionVector = make_dir_vectors(seq, directionIndex, numDims, directionVector)

        projectedSignal = np.dot(signal, directionVector)

        minIndices, maxIndices = local_peaks(projectedSignal)
        numExtrema[directionIndex] = len(minIndices) + len(maxIndices)

        zeroCrossIndices = zero_crossings(projectedSignal)
        numZeroCrossings[directionIndex] = len(zeroCrossIndices)

        tMin, tMax, zMin, zMax, mode = boundary_conditions(
            minIndices, maxIndices, time, projectedSignal, signal, numSymmetries
        )

        # multidimensional envelopes with spline interpolation
        if mode:  # Only if the projected signal has enough extrema
            fMin = CubicSpline(tMin, zMin, bc_type="not-a-knot")
            envelopeMin = fMin(time)

            fMax = CubicSpline(tMax, zMax, bc_type="not-a-knot")
            envelopeMax = fMax(time)

            diff_env = envelopeMax - envelopeMin
            amplitude += 0.5 * np.sqrt(np.einsum('ij,ij->i', diff_env, diff_env))
            envelopeMean += 0.5 * (envelopeMax + envelopeMin)
        else:  # If not enough extrema
            insufficientExtremaCount += 1

    # Normalize 
    if numDirs > insufficientExtremaCount:
        envelopeMean /= (numDirs - insufficientExtremaCount)
        amplitude /= (numDirs - insufficientExtremaCount)
    else:
        envelopeMean = np.zeros((numSamples, numDims))
        amplitude = np.zeros(numSamples)
        numExtrema = np.zeros(numDirs)

    return envelopeMean, numExtrema, numZeroCrossings, amplitude


#Stopping criterion
def stop(mode, time, sd, sd2, tol, seq, nDirs, nSamples, nDims):
    try:
        envelopeMean, nExtrema, numZeroCrossings, amplitude = envelope_mean(
            mode, time, seq, nDirs, nSamples, nDims
        )
        sx = np.sqrt(np.einsum('ij,ij->i', envelopeMean, envelopeMean))

        if np.all(amplitude):  # Avoid division by zero
            sx /= amplitude

        if not ((np.mean(sx > sd) > tol or np.any(sx > sd2)) and np.any(nExtrema > 2)):
            stopFlag = 1
        else:
            stopFlag = 0
    except Exception as e:
        warnings.warn(f"Error in stopping criterion: {e}")
        envelopeMean = np.zeros((nSamples, nDims))
        stopFlag = 1

    return stopFlag, envelopeMean
    
    
def fix(mode, time, seq, nDirs, stopCount, counter, nSamples, nDims):
    try:
        envelopeMean, numExtrema, numZeroCrossings, amplitude = envelope_mean(
            mode, time, seq, nDirs, nSamples, nDims
        )

        if np.all(np.abs(numZeroCrossings - numExtrema) > 1):
            stopFlag = 0
            counter = 0
        else:
            counter += 1
            stopFlag = int(counter >= stopCount)
    except Exception as e:
        warnings.warn(f"Error in stopping criterion: {e}")
        envelopeMean = np.zeros((nSamples, nDims))
        stopFlag = 1

    return stopFlag, envelopeMean, counter

def peaks(signal):
    signalDiff = np.sign(np.diff(signal.T)).T
    peakLocs = np.where(np.logical_and(signalDiff[:-1] > 0, signalDiff[1:] < 0))[0] + 1
    peakValues = signal[peakLocs]

    return peakValues, peakLocs


def local_peaks(signal):
    if np.all(signal < 1e-5):
        signal = np.zeros((1, len(signal)))

    signalLength = len(signal) - 1

    # extrema of projected 
    signalDiff = np.diff(signal.T).T
    nonZeroDiffInds = np.where(signalDiff != 0)[0]
    extremaBreaks = np.where(np.diff(nonZeroDiffInds) != 1)[0] + 1
    extremaDistances = nonZeroDiffInds[extremaBreaks] - nonZeroDiffInds[extremaBreaks - 1]
    nonZeroDiffInds[extremaBreaks] -= np.floor(extremaDistances / 2).astype(int)
    nonZeroDiffInds = np.insert(nonZeroDiffInds, len(nonZeroDiffInds), signalLength)
    extremaVals = signal[nonZeroDiffInds]

    if len(extremaVals) > 1:
        maxPeaks, maxLocs = peaks(extremaVals)
        minPeaks, minLocs = peaks(-extremaVals)

        if len(minPeaks) > 0:
            minIndices = nonZeroDiffInds[minLocs]
        else:
            minIndices = np.asarray([])

        if len(maxPeaks) > 0:
            maxIndices = nonZeroDiffInds[maxLocs]
        else:
            maxIndices = np.asarray([])
    else:
        minIndices = np.array([])
        maxIndices = np.array([])

    return minIndices, maxIndices


def stop_emd(r,seq,ndir,N_dim):
    ner = np.zeros((ndir,1))
    dir_vec = np.zeros((N_dim,1))
    
    for it in range(0,ndir):
        dir_vec = make_dir_vectors(seq,it,N_dim,dir_vec)    
        # Projection of input signal on nth (out of total ndir) direction
        # vectors
        y = np.dot(r,dir_vec)
        indmin, indmax = local_peaks(y)
        ner[it] = len(indmin) + len(indmax)
    
    stp = all(ner<3)
    
    return (stp)

    
def memd(x, ndir, maxnIMF=None, stp_crit ='stop', sd=0.075, sd2=0.75, tol=0.075,stp_cnt=2, MaxIterations=1000):

    nbit=0
    N_dim = np.shape(x)[1]
    N = np.shape(x)[0]
    t = np.arange(1,np.shape(x)[0]+1)

    seq = Halton(d=N_dim).random(n=ndir)
    r=x.copy()
    n_imf=1
    q = []
    while not stop_emd(r,seq,ndir,N_dim):
        # current mode
        m = r.copy()
        
        # computation of mean and stopping criterion
        if stp_crit == 'stop':
            stop_sift,env_mean = stop(m,t,sd,sd2,tol,seq,ndir,N,N_dim)
        else:
            counter=0
            stop_sift,env_mean,counter = fix(m,t,seq,ndir,stp_cnt,counter,N,N_dim)
            
        # In case the current mode is so small that machine precision can cause
        # spurious extrema to appear
        if np.max(np.abs(m)) < (1e-10)*(np.max(np.abs(x))):
            if not stop_sift:
                warnings.warn('emd:warning','forced stop of EMD : too small amplitude')
            else:
                print('forced stop of EMD : too small amplitude')
            break
        
        # sifting loop
        while not stop_sift and nbit < MaxIterations:
            # sifting
            m = m - env_mean
            
            # computation of mean and stopping criterion
            if stp_crit =='stop':
                stop_sift,env_mean = stop(m,t,sd,sd2,tol,seq,ndir,N,N_dim)
            else:
                stop_sift,env_mean,counter = fix(m,t,seq,ndir,stp_cnt,counter,N,N_dim)
        
            nbit=nbit+1
            
            if nbit == (MaxIterations-1) and  nbit > 100:
                warnings.wanr('emd:warning','forced stop of sifting : too many erations')
            
        q.append(m.T)
        
        n_imf = n_imf+1
        if maxnIMF != None:
            if n_imf >= maxnIMF:
                break
        r = r - m
        nbit = 0
        
    # Stores the residue
    q.append(r.T)
    q = np.asarray(q)

    return(q) 
# =============================================================================