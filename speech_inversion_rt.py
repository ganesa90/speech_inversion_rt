# -*- coding: utf-8 -*-
"""
Created on Tue May  2 00:03:45 2017

@author: ganesh
"""

import pyaudio
import time
import matplotlib
#import pylab
import numpy as np
from librosa import util
import scipy.fftpack as fft
from librosa.filters import get_window
import scipy
from librosa import load
import librosa
from contextualize import *
import pickle
from scipy.signal import filtfilt, firwin, kaiserord
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

class SWHear(object):
    """
    The SWHear class is made to provide access to continuously recorded
    (and mathematically processed) microphone data.
    """

    def __init__(self,device=None,startStreaming=True):
        """fire up the SWHear class."""
        print(" -- initializing SWHear")

        self.chunk = 160 # number of data points to read at a time
        self.rate = 8000 # time resolution of the recording device (Hz)
        self.tv_rate = 100 # time resolution of TVs (Hz)
        self.tv_dim = 6

        # for tape recording (continuous "tape" of recent audio)
        self.tapeLength=10 #seconds
#        self.tape=np.empty(self.rate*self.tapeLength)*np.nan
        self.tape=np.zeros(self.rate*self.tapeLength)
        self.spec_tape=np.zeros((257, self.tv_rate*self.tapeLength))
        
        self.tv_tape=np.zeros((self.tv_dim, self.tv_rate*self.tapeLength))
        self.tv_tape_filt = np.zeros((self.tv_dim, self.tv_rate*self.tapeLength))

        self.p=pyaudio.PyAudio() # start the PyAudio class
        if startStreaming:
            self.stream_start()
        self.sig = []
        self.mfcc = []
        self.tv_all = []
        self.model = pickle.load(open('xrmb_si_dnn_512_512_512_withDrop_bestmodel_weights.pkl','rb'))
        
        #Lowpass filter for TVs
        nyq = 0.5 * self.tv_rate
        normal_cutoff = 5.0 / nyq
        width = 5.0/nyq
        ripple_db = 60.0
        N, beta = kaiserord(ripple_db, width)
        taps = firwin(N, normal_cutoff, window=('kaiser', beta))
#        b, a = butter(10, normal_cutoff, btype='low', analog=False)
        self.lpf_b = taps
        self.lpf_a = 1.0
        
        # Plot line variables
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(4,1,1)
        self.ax2 = self.fig.add_subplot(4,1,2)
        self.ax3 = self.fig.add_subplot(4,1,3)
        self.ax4 = self.fig.add_subplot(4,1,4)
        
#        S = librosa.core.stft(self.tape, n_fft=512, hop_length=self.chunk/2, win_length=self.chunk, window='hamming', center=True)
#        D_spec = librosa.amplitude_to_db(S, ref=np.max)
#        mesh_coords = getattr(librosa.display, '__mesh_coords')
#        self.ax1_y_coords = mesh_coords('linear', None, D_spec.shape[0], sr=self.rate)
#        self.ax1_x_coords = mesh_coords(None, None, D_spec.shape[1], sr=self.rate)        
#        self.ax1.pcolormesh(self.ax1_x_coords, self.ax1_y_coords, D_spec, cmap='jet')
#        self.ax1_grid = self.ax1.contourf(self.ax1_x_coords, self.ax1_y_coords, D_spec, cmap='jet')
#        self.ax1_grid = self.ax1.contourf(D_spec, cmap='jet')
#        self.ax1.set_xlim(self.ax1_x_coords.min(), self.ax1_x_coords.max())
#        self.ax1.set_ylim(self.ax1_y_coords.min(), self.ax1_y_coords.max())
        
        self.l1, = self.ax1.plot(np.arange(self.tape.shape[0])/float(self.rate),self.tape/float(2**15))        
        self.l2, = self.ax2.plot(np.arange(self.tv_tape.shape[1])/float(self.tv_rate),self.tv_tape_filt[0,:])
        self.l3, = self.ax3.plot(np.arange(self.tv_tape.shape[1])/float(self.tv_rate),self.tv_tape_filt[3,:])
        self.l4, = self.ax4.plot(np.arange(self.tv_tape.shape[1])/float(self.tv_rate),self.tv_tape_filt[5,:])
        
        self.ax1.axis([0,self.tapeLength,-1,1])
        self.ax2.axis([0,self.tapeLength,-2,2])
        self.ax3.axis([0,self.tapeLength,-2,2])
        self.ax4.axis([0,self.tapeLength,-2,2])
        
        self.ax1.set_title("Waveform",{'fontsize':14, 'fontweight':'bold'})
        self.ax2.set_title("Lip Aperture",{'fontsize':14, 'fontweight':'bold'})
        self.ax3.set_title("Tongue Body Constriction Degree",{'fontsize':14, 'fontweight':'bold'})
        self.ax4.set_title("Tongue Tip Constriction Degree",{'fontsize':14, 'fontweight':'bold'})
        self.ax4.set_xlabel("Time",{'fontsize':14, 'fontweight':'bold'})
        
#        self.ax1.relim() 
#        self.ax1.autoscale_view(True,True,True)
        self.ax2.relim() 
        self.ax2.autoscale_view(True,True,True)
        self.ax3.relim() 
        self.ax3.autoscale_view(True,True,True)
        self.ax4.relim() 
        self.ax4.autoscale_view(True,True,True)
        self.fig.canvas.draw()        
        mng = plt.get_current_fig_manager()        
        mng.full_screen_toggle()
        plt.show(block=False)

    ### LOWEST LEVEL AUDIO ACCESSy_frames = librosa.util.frame(y, frame_length=512, hop_length=80)
    # pure access to microphone and stream operations
    # keep math, plotting, FFT, etc out of here.

    def stream_read(self):
        """return values for a single chunk"""
        data = np.fromstring(self.stream.read(self.chunk),dtype=np.int16)
        self.sig.append(data)
        #print(data)
        return data

    def stream_start(self):
        """connect to the audio device and start a stream"""
        print(" -- stream started")
        self.stream=self.p.open(format=pyaudio.paInt16,channels=1,
                                rate=self.rate,input=True,
                                frames_per_buffer=self.chunk)

    def stream_stop(self):
        """close the stream but keep the PyAudio instance alive."""
        if 'stream' in locals():
            self.stream.stop_stream()
            self.stream.close()
        print(" -- stream CLOSED")

    def close(self):
        """gently detach from things."""
        self.stream_stop()
        self.p.terminate()

    ### TAPE METHODS
    # tape is like a circular magnetic ribbon of tape that's continously
    # recorded and recorded over in a loop. self.tape contains this data.
    # the newest data is always at the end. Don't modify data on the type,
    # but rather do math on it (like FFT) as you read from it.
    def tv_tape_add(self, tv):
        """add a single chunk to the tape."""
        self.tv_tape[:, :-2]=self.tv_tape[:, 2:]
        self.tv_tape[:, -2:]=tv
        self.tv_all.append(tv)
        self.tv_tape_filt = self.butter_lowpass_filter(self.tv_tape)
    
    def spec_tape_add(self, spec):
        """add a single chunk to the tape."""
        self.spec_tape[:, :-1]=self.spec_tape[:, 1:]
        self.spec_tape[:, -1:]=spec

#    def tv_tape_flush(self):
#        """completely fill tape with new data."""
#        readsInTape=int(self.rate*self.tapeLength/self.chunk)
#        print(" -- flushing %d s tape with %dx%.2f ms reads"%\
#                  (self.tapeLength,readsInTape,self.chunk/self.rate))
#        for i in range(readsInTape):
#            self.tape_add()


    def tape_add(self):
        """add a single chunk to the tape."""
        self.tape[:-self.chunk]=self.tape[self.chunk:]
        self.tape[-self.chunk:]=self.stream_read()        

    def tape_flush(self):
        """completely fill tape with new data."""
        readsInTape=int(self.rate*self.tapeLength/self.chunk)
        print(" -- flushing %d s tape with %dx%.2f ms reads"%\
                  (self.tapeLength,readsInTape,self.chunk/self.rate))
        for i in range(readsInTape):
            self.tape_add()

    def tape_forever(self,plotSec=1.0):
        # This is going to control the whole flow
        t1=0
        tstart = time.time()        
        # Here is where[:-1,:-1].ravel()) I need to request for an utterance and wait for mvn
        try:
            while True:
                self.tape_add()
#                t1 = time.time()
                if len(self.mfcc) == 0:
                    self.mfcc.append(self.get_mfcc(self.tape[-self.chunk:]))
                else:
                    self.mfcc.append(self.get_mfcc(self.tape[-int(1.5*self.chunk):-int(0.5*self.chunk)]))
                self.mfcc.append(self.get_mfcc(self.tape[-self.chunk:]))
#                print('Time for mfcc', time.time()-t1)
                if len(self.mfcc) >= 34:
                    self.tv_tape_add(self.estimate_tv())
                if (time.time()-t1)>plotSec:
                    t1=time.time()
                    print(time.time()-tstart)
                    self.tape_plot()
#                    print(time.time()-t1)
#                    if t1-tstart >= 20:
#                        return
        except:
            print(" ~~ exception (keyboard?)")
            return

    def tape_plot(self,saveAs="03.png"):
        """plot what's in the tape."""

        self.l1.set_ydata(self.tape/float(2**15))        
        self.l2.set_ydata(self.tv_tape_filt[0,:])
        self.l3.set_ydata(self.tv_tape_filt[3,:])
        self.l4.set_ydata(self.tv_tape_filt[5,:])
        self.fig.canvas.draw()
#        print('Canvas draw: ',time.time()-t1)
        
#        if saveAs:
#            t1=time.time()
#            self.fig.savefig(saveAs,dpi=100)
#            print("plotting saving took %.02f ms"%((time.time()-t1)*1000))
#        else:
#            plt.show()
#            print() #good for IPython
#        plt.close('all')   
    
    def get_mfcc(self, sig_frm):
        sig_frm = sig_frm/32768.0
        window = 'hamming'
        win_length = sig_frm.shape[0]
        hop_length=win_length
        center=True
        n_fft=win_length
        fft_window = get_window(window, win_length, fftbins=True)
        fft_window = util.pad_center(fft_window, n_fft)
        fft_window = fft_window.reshape((-1, 1))
        util.valid_audio(sig_frm)
        sig_frm = sig_frm[:,None]
        stft_matrix = np.empty((int(1 + n_fft // 2), 1), dtype=np.complex64, order='F')
        stft = fft.fft(fft_window*sig_frm, axis=0)[:stft_matrix.shape[0]].conj()
        powspec = np.abs(stft)**2                
        melspec = librosa.feature.melspectrogram(S=powspec, hop_length=hop_length,
                                             n_fft=n_fft, n_mels=40)        
        mfcc = librosa.feature.mfcc(S=librosa.logamplitude(melspec), n_mfcc=13)
        
        n_fft = 512
        fft_window = get_window(window, win_length, fftbins=True)
        fft_window = util.pad_center(fft_window, n_fft)
        fft_window = fft_window.reshape((-1, 1))
        y = np.pad(sig_frm[:,0], int(n_fft // 2), mode='reflect')
        pad_frame = librosa.util.frame(y, frame_length=n_fft, hop_length=win_length*2)[:,0][:, None]
        stft_matrix = np.empty((int(1 + n_fft // 2), 1), dtype=np.complex64, order='F')
        stft = fft.fft(fft_window*pad_frame, axis=0)[:stft_matrix.shape[0]].conj()        
        powspec = np.abs(stft)**2
        power_to_db = getattr(librosa, 'power_to_db')
        spec = power_to_db(powspec)
        self.spec_tape_add(spec)
        return mfcc
            
    def estimate_tv(self):
        mean_feats = np.mean(np.asarray(self.mfcc), axis=0)
        std_feats = np.std(np.asarray(self.mfcc), axis=0)
        std_feats[std_feats<1] = 1
        feats = np.concatenate(self.mfcc[-34:], axis=1)
        feats = 0.25*(feats - mean_feats)/std_feats
        cont_feats = contextualize(feats, 8, 2)
        cur_inp = cont_feats[:,17:19]
        out = cur_inp.T
        for ii in [0, 2, 4]:
            out = np.tanh(np.dot(out, self.model[ii])+self.model[ii+1])
        out = np.dot(out, self.model[6]) + self.model[7]
        tv = out.T[range(0,18,3), :]
        return tv

    def butter_lowpass_filter(self, data):        
        y = filtfilt(self.lpf_b, self.lpf_a, data, axis=1)
        return y
        
    
if __name__=="__main__":
    ear=SWHear()
    ear.tape_forever()    
    sig = np.int16(np.concatenate(ear.sig, axis=0))
    scipy.io.wavfile.write('./recorded_sound.wav', 8000, sig)
    tape = ear.tape
    feats = ear.mfcc
    tv_tape = ear.tv_tape
    tv_tape_smth = ear.tv_tape_filt
    tv_all = np.concatenate(ear.tv_all, axis=1)
    np.save('./recorded_tv.npy', tv_all)
    W = ear.model
    b = ear.lpf_b
    a = ear.lpf_a
    ear.close()
    print("DONE")
#    