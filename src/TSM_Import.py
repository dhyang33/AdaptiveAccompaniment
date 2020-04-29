#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa as lb
import IPython.display as ipd
import scipy.signal as ss
import scipy.fft as sfft
import time
from skimage.measure import block_reduce



def getWindow(N, H_s):
  hann = ss.hann(N, sym=False)
  window = np.zeros(N)
  padded_window = np.ones(N)
  i = 0
  while(np.count_nonzero(padded_window) != 0):
    left_pad = H_s*i
    padded_window = np.pad(hann, (left_pad, 0), 'constant')[:N]
    window += padded_window
    i+=1
  padded_window = np.ones(N)
  i = 1
  while(np.count_nonzero(padded_window) != 0):
    right_pad = H_s*i
    padded_window = np.pad(hann, (0, right_pad), 'constant')[-N:]
    window += padded_window
    i+=1
  return window


# In[ ]:


def reconstructFromFrames(new_frames, H_s):
  final_size = H_s*(len(new_frames)-1)+len(new_frames[0])
  new_audio  = np.zeros(final_size)
  for idx, new_frame in enumerate(new_frames):
    new_frame = np.array(new_frame)
    left_pad = H_s*idx
    right_pad = final_size-(left_pad+len(new_frame))
    padded = np.pad(new_frame, (left_pad, right_pad), 'constant')
    new_audio = new_audio + padded
  return new_audio


# In[ ]:


def OLA(audio, stretch, sr=22050, N=220):
  # assume stretch to be time
  H_s = int(np.round(N/2))
  instances = [int(m*H_s) for m in range(len(audio)//H_s)]
  analysis_instances = [stretch[instance] for instance in instances]
  frames = [audio[analysis_instances[k]:analysis_instances[k]+N]             for k in range(1,len(analysis_instances)) if analysis_instances[k]+N< len(audio)]
  hann = ss.hann(N, sym=False)
  new_frames = []
  window = getWindow(N, H_s)
  for frame in frames:
    new_frames.append(frame*hann/window)
  new_audio=reconstructFromFrames(new_frames, H_s)
  return new_audio


# In[ ]:

def estimateIF(S, sr, hop):
    fftsize = (S.shape[0] - 1) * 2
    w_nom = np.arange(S.shape[0]) * sr / fftsize * 2 * np.pi
    w_nom = w_nom.reshape((-1, 1))
    unwrapped = np.angle(S[:, 1:]) - np.angle(S[:, 0:-1]) - w_nom * hop
    wrapped = (unwrapped + np.pi) % (2 * np.pi) - np.pi
    w_if = w_nom + wrapped / hop
    f_if = w_if / (2 * np.pi)
    return w_if


# In[ ]:


def getSquaredWindow(N, H_s):
  hann = ss.hann(N, sym=False)
  window = np.zeros(N)
  padded_window = np.ones(N)
  i = 0
  while(np.count_nonzero(padded_window) != 0):
    left_pad = H_s*i
    padded_window = np.pad(hann, (left_pad, 0), 'constant')[:N]
    window += padded_window*padded_window
    i+=1
  padded_window = np.ones(N)
  i = 1
  while(np.count_nonzero(padded_window) != 0):
    right_pad = H_s*i
    padded_window = np.pad(hann, (0, right_pad), 'constant')[-N:]
    window += padded_window*padded_window
    i+=1
  return window


# In[ ]:


def phase_vocoder(audio, stretch, sr=22050, N=2048):
  hann = ss.hann(N, sym=False)
  H_s = int(np.round(N/4))
  #instances = [int(m*H_s) for m in range(len(audio)//H_s)]
  instances = [int(m*H_s) for m in range(len(stretch)//H_s)]
  analysis_instances = [stretch[instance] for instance in instances]
  frames = [audio[analysis_instances[k]:analysis_instances[k]+N]             for k in range(1,len(analysis_instances)) if analysis_instances[k]+N< len(audio)]
  spectrogram = np.zeros((N//2+1,len(frames)))
  phase = np.zeros((N//2+1,len(frames)))
  for index, frame in enumerate(frames):
    spec = np.fft.rfft(frame*hann)
    spectrogram[:,index]= spec
  S = spectrogram
  analysis_instances = np.array(analysis_instances)
  hop = (analysis_instances[1:]-analysis_instances[:-1])[:len(frames)]/sr
  fftsize = (S.shape[0] - 1) * 2
  w_nom = np.arange(S.shape[0]) * sr / fftsize * 2 * np.pi
  w_nom = w_nom.reshape((-1, 1))
  print("w nom",w_nom.shape,"hop", hop.shape, "S", S.shape,'analysis_instances', analysis_instances.shape,'S[0]', S[0].shape)
  
  
  hop=np.ones(S.shape[0]).reshape((-1,1)) *hop
  
  time_step =  w_nom * hop
  difference = np.angle(S[:, 1:]) - np.angle(S[:, 0:-1])
  #print(difference.shape, time_step[:,1:].shape)
  unwrapped = difference - time_step[:,1:]
  wrapped = (unwrapped + np.pi) % (2 * np.pi) - np.pi
  w_if = w_nom + wrapped / hop[:,1:]
  f_if = w_if / (2 * np.pi)
  spectrogram = S
  phases = np.zeros(np.shape(spectrogram))
  phases[:,0] = np.angle(spectrogram[:,0])
  for i in range(0, phases.shape[1]-1):
    phases[:,i+1] = phases[:,i]+(H_s/sr)*w_if[:,i]
  
  new_spectrogram = np.abs(spectrogram)*np.exp(1j*phases)
  #
  #print(new_spectrogram.shape)
  new_frames = []
  window = getSquaredWindow(N, H_s)
  for freq_frame in new_spectrogram.T:
    #print(freq_frame.shape)
    #time_frame = sfft.ifft(freq_frame)
    time_frame = np.fft.irfft(freq_frame)
    new_frame = hann * time_frame/window
    new_frames.append(new_frame)
  new_audio = reconstructFromFrames(new_frames,H_s)
  return np.real(new_audio)


# In[ ]:


# new_audio = phase_vocoder(audio, stretch)
# ipd.Audio(new_audio, rate = sr)


# # HPS

# In[ ]:


#a=np.array([np.array([np.arange(2) for i in range(3) ])+i for i in range(4) ])


# In[ ]:


#np.median(np.roll(a,-1,-1),axis=0)


# In[ ]:


def median_filter(S, length, direction):
  
  med_fil = np.empty((length,S.shape[0],S.shape[1]))
  med = length //2 
  r = np.arange(length) - med
  if direction=='h':
    for i in r:
      med_fil[r,:,:] = np.roll(S,i,axis=-1)
  if direction=='v':
    for i in r:
      med_fil[r,:,:] = np.roll(S,i,axis=0)

  return np.median(med_fil,axis=0)
  '''
  if direction =="h":
    for i in range(S.shape[1]):
      left = max(0,i-length)
      right = min(i+length,S.shape[1])
      
      med_fil[:,i] = np.median(S[:,left:right],axis=1)
  if direction =="v":
    for i in range(S.shape[0]):
      top = max(0,i-length)
      bottom = min(i+length,S.shape[0])
      med_fil[i,:] = np.median(S[top:bottom,:],axis=0)
  '''
  return med_fil


# In[ ]:


def spec_to_audio(new_spectrogram, sr, N, hop):
  hann = ss.hann(N, sym=False)
  new_frames = []
  window = getSquaredWindow(N, hop)
  for freq_frame in new_spectrogram.T:
    time_frame = np.fft.irfft(freq_frame)
    new_frame = hann * time_frame/window
    new_frames.append(new_frame)
  new_audio = reconstructFromFrames(new_frames,hop)
  return np.real(new_audio)


# In[ ]:


def HPS(audio, filter_length=6,N=2048,sr=22050, hop = 128, verbal=False ):
    since = time.time()
    spectrogram = lb.core.stft(audio, N, hop)
    
    S = np.abs(spectrogram)
    if verbal: print("start hori filter", time.time()-since)
    median_h = median_filter(S, filter_length, "h")
    if verbal: print("start vert filter", time.time()-since)
    median_v = median_filter(S, filter_length, "v")

    
    
    #median_h = filtering(S, (1,6), np.median)

    #median_v = filtering(S, (6,1), np.median)
    if verbal: print("compare filtering", time.time()-since)
    M_h = np.where(median_h>median_v,1,0)
    if verbal: print("construct H P matrix", time.time()-since)
    H_Spec = spectrogram*M_h
    P_Spec = spectrogram*(1-M_h)
    if verbal: print("audio construction for H ", time.time()-since)
    H_audio = spec_to_audio(H_Spec, sr, N, hop)
    if verbal: print("audio construction for P", time.time()-since)
    P_audio = spec_to_audio(P_Spec, sr, N, hop)
    if verbal: print("complete reconustrction", time.time()-since)

    return H_audio,P_audio


# In[ ]:


def HPS_TSM(audio, time_function, verbal=False):
    if verbal:
      begin_time = time.time()
      print("start HPS",time.time()-begin_time)
    H_audio, P_audio = HPS(audio,verbal=verbal)
    minLength = min(len(H_audio), len(P_audio),len(audio))
    if verbal:
      print("H audio, P audio, audio, ", H_audio.shape, P_audio.shape, audio.shape)
    H_audio, P_audio = H_audio[:minLength], P_audio[:minLength]
    if verbal:
      print('start phase vocoder',time.time()-begin_time)
    new_H_audio = phase_vocoder(H_audio, time_function)
    if verbal:
      print('start phase OLA',time.time()-begin_time)
    new_P_audio = OLA(P_audio, time_function)
    minLength = min(len(new_H_audio), len(new_P_audio))
    if verbal:
      print('complete',time.time()-begin_time)
    return new_H_audio[:minLength]+new_P_audio[:minLength]







