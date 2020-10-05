# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 16:25:03 2017

@author: Xiaoxuan Jia
"""
###

import numpy as np
from scipy import stats
import scipy
from jitter import jitter

def xcorrfft(a,b,NFFT):
    # first dimention of a should be length of time
    CCG = np.fft.fftshift(np.fft.ifft(np.multiply(np.fft.fft(a,NFFT), np.conj(np.fft.fft(b,NFFT)))))
    return CCG

def nextpow2(n):
    """get the next power of 2 that's greater than n"""
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i


def get_ccgjitter(spikes1, spikes2, jitterwindow=25):
    # spikes: trial*time

    n_t = np.shape(spikes)[-1]
    # triangle function
    t = np.arange(-(n_t-1),(n_t-1))
    theta = n_t-np.abs(t)
    del t
    NFFT = int(nextpow2(2*n_t))

    FR1 = np.squeeze(np.mean(np.sum(spikes1,axis=1), axis=0))
    FR2 = np.squeeze(np.mean(np.sum(spikes2,axis=1), axis=0))
    tempccg = xcorrfft(spikes1,spikes2,NFFT)

    temp1 = jitter(spikes1,jitterwindow);  
    temp2 = jitter(spikes2,jitterwindow);
    tempjitter = xcorrfft(temp1, temp2, NFFT)

    # normalize by rate and triangle function
    ccgjitter = (tempccg - tempjitter).T/np.multiply(np.sqrt(FR1*FR2), np.tile(theta.T.reshape(len(theta),1),(1,len(FR1))))

    return ccgjitter

