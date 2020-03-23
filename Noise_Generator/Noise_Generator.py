#!/usr/bin/env python

import pickle
import InstrumentDriver
import numpy as np
import ast
import sys, os
from scipy.fftpack import fft, ifft
import scipy.io as sio


def loadData(file_path):

    """
    Load a log file. (Use the built-in pickle module)
    
    Parameters
    ----------
    file_path: str 
        path of the log file


    Returns
    -------
    - data: arbitrary object
        arbitrary Python object which contains data
    """
    import pickle
    with open(file_path, 'rb') as _input:
        data = pickle.load(_input)
    return data

def white(w, S0=1.):
    return S0 * np.ones_like(w)
def lorentz(w, S0=1., wc=1., w0=0.):
    """ Lorentzian spectrum. """
    return S0/(2*np.pi*wc)*(1./(1+((w-w0)/wc)**2) + 1./(1+((w+w0)/wc)**2))

def genProcess(times, spectrum, T0):
    """ Generate one realization of a Gaussian random process. """

    # times:    numpy array containing all the times for which to generate the process
    # spectrum: numpy array containing the spectrum at all the harmonics omega_k = 2*pi*k/T0
    # T0:       period of the noise process. Has to be much longer than the duration of an
    #           experimental run
    # returns:  numpy array containing all the values of the random process for the times
    #           specified in input

    # Check that T0 is longer than the largest specified time
    if (T0 < max(times)):
        print("WARNING: T0 should be the longest timescale.")

    # Number of frequencies and times
    nfreqs = max(np.shape(spectrum))
    ntimes = max(np.shape(times))

    # Generation of harmonics and random Fourier coefficients
    vec_omega = 2*np.pi/T0 * np.arange(1,nfreqs+1)
    vec_sigma = np.sqrt(2.*spectrum/T0)
    vec_a = vec_sigma * np.random.normal(size=nfreqs)
    vec_b = vec_sigma * np.random.normal(size=nfreqs)

    # Matrix with times in rows
    mat_times = np.array(np.repeat(times[np.newaxis],nfreqs,0))

    # Matrix with frequencies in columns
    mat_freqs = np.array(np.repeat(np.transpose(vec_omega[np.newaxis]),ntimes,1))

    # Sum up the Fourier series
    mat_cos = np.cos(mat_freqs * mat_times);
    cos_term = np.dot(np.reshape(vec_a,(1,nfreqs)), mat_cos)
    mat_sin = np.sin(mat_freqs * mat_times);
    sin_term = np.dot(np.reshape(vec_b,(1,nfreqs)), mat_sin)

    return (cos_term + sin_term)[0]


class Driver(InstrumentDriver.InstrumentWorker):
    """ This class implements a Single-qubit pulse generator"""
    

    def performOpen(self, options={}):
        self.vEnvelope = np.array([], dtype=float)
        self.vNoise_Time = np.array([], dtype=float)
        self.vNoise_Time_Modulated = np.array([], dtype=float)
        self.vNoise_Freq = np.array([], dtype=float)
        self.vNoise_Freq_FFT = np.array([], dtype=float)
        self.vTime = np.array([], dtype=float)
        self.vFreq = np.array([], dtype=float)
        self.vFreq_AWG_frame = np.array([], dtype=float)
        self.vBinEdges = np.array([], dtype=float)
        self.vHistogram = np.array([], dtype=float)
        self.PrevNoiseIndex = -9.99e-9
        self.saved_data = None
        self.original_S0 = 0

    def performSetValue(self, quant, value, sweepRate=0.0, options={}):
        """Perform the Set Value instrument operation. This function should
        return the actual value set by the instrument"""
        # do nothing, just return value
        if quant.name == 'Read file':
            self.Readfile()
        elif quant.name == 'Read file when start':
            if (value == True):
                self.Readfile()
        return value

    def Readfile(self):
        # self.log('Read File')
        file_path = self.getValue("File path")
        self.saved_data = loadData(file_path)
        self.setValue("T0", self.saved_data["T0"])
        self.setValue("High Cutoff Freq.", self.saved_data["High Cutoff Freq."])
        self.original_S0 = self.saved_data["S0"]
        self.setValue("Noise Power", self.saved_data["S0"])
        self.setValue("Center Freq.", self.saved_data["Center Freq."])
        self.setValue("HWHM", self.saved_data["HWHM"])
        self.reportStatus('Successfully Read Data File!')

    def performGetValue(self, quant, options={}):
        """Perform the Get Value instrument operation"""
        # check type of quantity
        if quant.isVector():
            noise_indx = self.getValue("Noise Index")
            self.calculateNoise()
            # if self.PrevNoiseIndex != noise_indx:
            #     self.calculateNoise()
            #     self.PrevNoiseIndex = noise_indx

            # traces, check if waveform needs to be re-calculated
            if self.isConfigUpdated():
                self.calculateWaveform()


            # if ((quant.name == 'Trace - Modulated Noise (Time-Domain)') & (len(self.vEnvelope) != len(self.vNoise_Time))):
            #     self.calculateNoise()
                
            # get correct data and return as trace dict
            vData = self.getWaveformFromMemory(quant)
            length = len(vData)
            if (self.getValue("Turn Off Noise") == True):
                vData = np.zeros(length)
            if quant.name == 'Trace - Noise (Histogram)':
                value = quant.getTraceDict(vData, x=self.vBinEdges)
            elif quant.name == 'Trace - Noise (Freq-Domain, FFT)':
                value = quant.getTraceDict(vData, x=self.vFreq_AWG_frame)
            elif quant.name == 'Trace - Noise (Freq-Domain, Original)':
                value = quant.getTraceDict(vData, x=self.vFreq)
            # elif quant.name == 'Trace - Envelope':
            #     dt = 1/self.getValue('Sample rate')
            #     value = quant.getTraceDict(self.vEnvelope, dt = dt)
            else:
                dt = 1/self.getValue('Sample rate')
                value = quant.getTraceDict(vData, dt=dt)
        else:
            # for all other cases, do nothing
            value = quant.getValue()

        return value


    def getWaveformFromMemory(self, quant):

        dTrace = { 'Trace - Noise (Time-Domain)': self.vNoise_Time, 
                  'Trace - Noise (Freq-Domain, Original)': self.vNoise_Freq,
                  # 'Trace - Noise (Time-Domain, RAW)': self.vNoise_Time_RAW, 
                  'Trace - Noise (Freq-Domain, FFT)': self.vNoise_Freq_FFT,
                  'Trace - Noise (Histogram)': self.vHistogram,
                  'Trace - Envelope': self.vEnvelope,
                  'Trace - Modulated Noise (Time-Domain)': self.vNoise_Time * self.vEnvelope if (len(self.vNoise_Time) == len(self.vEnvelope)) else []
                  # 'Trace - Noise (Time-Domain, Histogram, RAW)': self.vTime_Histogram_RAW,
                  }
        vData = dTrace[quant.name]
        return vData
    def generateNoise(self):
        """Get the waveform of a given noise"""
        # get params
        nPoints = float(self.getValue('Number of points'))
        dSampleRate = self.getValue('Sample rate')
        str_NoiseType = self.getValue('Noise type')
        index = int(self.getValue('Noise Index'))
        use_interpolation = self.getValue('Use Interpolation')

        if (str_NoiseType == 'Custom'):
            vec_time = np.linspace(0, (nPoints-1)*1/dSampleRate,nPoints)
            vec_freq_AWG_frame = np.linspace(0.0, 1.0/(2.0/dSampleRate), nPoints*0.5)
            if (self.saved_data is None):
                self.reportStatus('No Data Available')
                return
            vec_T_noise = np.zeros_like(vec_time)
            index = np.clip(index, 0, len(self.saved_data['mat_T_noise'])-1)
            self.log('shape of mat_T_noise: ' + str(self.saved_data['mat_T_noise'].shape))
            vec_T_noise_data = np.copy(self.saved_data['mat_T_noise'][index]) * np.sqrt(self.getValue('Noise Power')/self.original_S0)
            self.log( self.saved_data['mat_T_noise'][index])
            if (len(vec_T_noise_data) >= len(vec_time)):
                vec_T_noise = vec_T_noise_data[:len(vec_time)]
            else:
                N_copy = int(len(vec_time) / len(vec_T_noise_data))
                for i in range(N_copy):
                    vec_T_noise[i*len(vec_T_noise_data):(i+1)*len(vec_T_noise_data)] = vec_T_noise_data
                vec_T_noise[(i+1)*len(vec_T_noise_data):] = vec_T_noise_data[0:len(vec_time)-(i+1)*len(vec_T_noise_data)]
            vec_F_noise = self.saved_data['vec_F_noise']
            vec_freq = self.saved_data['vec_freq']

        else:

            np.random.seed(int(index))
            T0 = self.getValue('T0')
            dHighCutoffFreq = self.getValue('High Cutoff Freq.')
            if use_interpolation:
                interval = self.getValue('Interpolation Interval')
                dSampleRate = int(dSampleRate / interval)
                nPoints = int(nPoints / self.getValue('Sample rate') * dSampleRate)
            # self.log(T0, dSampleRate, nPoints)
            vec_time = np.linspace(0, (nPoints-1)*1/dSampleRate,nPoints)
            vec_freq_AWG_frame = np.linspace(0.0, 1.0/(2.0/dSampleRate), nPoints*0.5)

            S0 = self.getValue('S0')
            freq_step = 1/T0
            N_freq = int(dHighCutoffFreq / freq_step)
            vec_freq = np.linspace(freq_step, dHighCutoffFreq, N_freq)

            if str_NoiseType == 'White':
                """Generate Gaussian Noise Signal(mean = 0, sigma = sqrt(Power Spectral Density))"""
                vec_F_noise = white(2 * np.pi * vec_freq, S0= S0)
                vec_T_noise = genProcess(vec_time, vec_F_noise, T0)

            elif str_NoiseType == 'Squared-Gaussian':
                """Generate Squared Gaussian Noise Signal"""
                vec_F_noise = white(2 * np.pi * vec_freq, S0= S0)
                vec_T_noise = genProcess(vec_time, vec_F_noise, T0) **2
      
            elif str_NoiseType == 'Lorentzian':
                f0 = self.getValue('Center Freq.')
                fwidth = self.getValue('HWHM')
                vec_F_noise = lorentz(2 * np.pi * vec_freq , S0= S0, wc = 2 * np.pi * fwidth , w0 = 2 * np.pi * f0)
                vec_T_noise = genProcess(vec_time, vec_F_noise, T0)

        #Filter noise in Time-Window
        dNoiseStart = self.getValue('Noise Start Time')
        dNoiseEnd = self.getValue('Noise End Time')
        index_Empty = np.where((vec_time < dNoiseStart) | (vec_time > dNoiseEnd))
        vec_T_noise[index_Empty] = 0


        bln_hist_FFT = self.getValue('Generate Histogram and FFT')
        d_bincounts = self.getValue('Histogram Bin Counts')
        vec_F_noise_FFT = []
        vec_histogram = []
        vec_binedges = []
        if (use_interpolation == True):
            nPoints = float(self.getValue('Number of points'))
            dSampleRate = self.getValue('Sample rate')
            vec_time_interp = np.linspace(0, (nPoints-1)*1/dSampleRate,nPoints)
            vec_freq_AWG_frame = np.linspace(0.0, 1.0/(2.0/dSampleRate), nPoints*0.5)
            interp_T_noise = np.interp(vec_time_interp, vec_time, vec_T_noise)
            vec_time = np.copy(vec_time_interp)
            vec_T_noise = np.copy(interp_T_noise)


        if (bln_hist_FFT):
            vec_histogram, vec_binedges = np.histogram(vec_T_noise, bins = int(d_bincounts))
            vec_F_noise_FFT = 2.0 / len(vec_time) * np.abs(fft(vec_T_noise).flatten())

        else:
            vec_F_noise_FFT = np.zeros_like(vec_freq_AWG_frame)
        return ({
                 "vec_time": vec_time, 
                 "vec_freq": vec_freq,
                 "vec_freq_AWG_frame": vec_freq_AWG_frame,
                 "vec_T_noise": vec_T_noise,
                 "vec_F_noise_FFT": vec_F_noise_FFT[:len(vec_time)//2],
                 "vec_F_noise":vec_F_noise,
                 "vec_histogram": vec_histogram,
                 "vec_binedges": vec_binedges[0:int(d_bincounts)],
                 })


    def calculateNoise(self):
        """Generate noise waveform"""
        # get config values
        nPoints = int(self.getValue('Number of points'))
        sampleRate = self.getValue('Sample rate')
        # start with allocating time and amplitude vectors
        self.vTime = np.arange(nPoints, dtype=float)/sampleRate
        
        # get noise waveform
        result= self.generateNoise()
        if (result is None):
            return
        self.vTime = result["vec_time"]
        self.vFreq = result["vec_freq"]
        self.vFreq_AWG_frame = result["vec_freq_AWG_frame"]
        self.vBinEdges = result["vec_binedges"]
        self.vNoise_Time = result["vec_T_noise"]
        self.vNoise_Freq = result["vec_F_noise"]
        self.vNoise_Freq_FFT = result["vec_F_noise_FFT"]
        self.vHistogram = result["vec_histogram"]



    def calculateWaveform(self):
        """Generate waveforms, including pre-pulses, readout and gates"""
        # get config values
        nPoints = int(self.getValue('Number of points'))
        sampleRate = self.getValue('Sample rate')
        firstDelay = self.getValue('First pulse delay')
        # start with allocating time and amplitude vectors
        self.vTime = np.arange(nPoints, dtype=float)/sampleRate
        # create list of output vectors
        self.vEnvelope = np.zeros_like(self.vTime) 
        # go on depending on with sequence
        self.generateSequence(startTime=firstDelay)


    def generateSequence(self, startTime):
        # get config values
        sSequence = self.getValue('Sequence')
        nPulses = int(self.getValue('# of pulses'))
        seqPeriod = self.getValue('Pulse period')
        # go on depending on waveform
        if sSequence == 'CP/CPMG':
            # get length of actual pulses
            dPulseT1 = self.getPulseDuration(1)
            dPulseT2 = self.getPulseDuration(2)
            dPulseTot = 2*dPulseT1 + dPulseT2*nPulses
            # add the first pi/2 pulses
            self.addPulse(1, startTime + dPulseT1/2)
            # add more pulses
            if nPulses <= 0:
                # no pulses = ramsey
                vTimePi = []
                # second pi/2 pulse
                self.addPulse(1, startTime + seqPeriod + dPulseTot - dPulseT1/2)
            elif nPulses == 1:
                # one pulse, echo experiment
                vTimePi = [startTime + dPulseT1 + seqPeriod/2 + dPulseT2/2]
                # second pi/2 pulse
                self.addPulse(1, startTime + seqPeriod + dPulseTot - dPulseT1/2)
            elif nPulses > 1:
                # figure out timing of pi pulses
                vTimePi = startTime  + dPulseT1 + seqPeriod/2 + dPulseT2/2 + \
                          (seqPeriod + dPulseT2)*np.arange(nPulses)
                # second pi/2 pulse
                self.addPulse(1, startTime + nPulses*seqPeriod + dPulseTot - dPulseT1/2)
            # add pi pulses, one by one
            for dTimePi in vTimePi:
                self.addPulse(2, dTimePi)

        elif sSequence == 'Generic sequence':
            # generic pulse sequence, add the pulses specified in the pulse list
            t = startTime
            for n in range(nPulses):
                pulseType = 1 + (n % 8)
                # get length of current pulse
                dPulseT = self.getPulseDuration(pulseType)
                self.addPulse(pulseType, t + dPulseT/2)
                # add spacing as defined for this pulse
                t += dPulseT + self.getValue('Spacing #%d' % (pulseType))

    def addPulse(self, nType, dTime, nOutput=None, bTimeStart=False, phase=None):
        """Add pulse to waveform"""
        vTime, vPulse, vIndx = self.getPulseEnvelope(nType, dTime, bTimeStart)
        if len(vTime) == 0:
            return
        self.vEnvelope[vIndx] += vPulse



    def getPulseEnvelope(self, nType, dTime, bTimeStart=False):
        """Get pulse envelope for a given pulse"""
        sPulseType = self.getValue('Pulse type')
        dSampleRate = self.getValue('Sample rate')
        truncRange = self.getValue('Truncation range')
        start_at_zero = self.getValue('Start at zero')
        # get pulse params
        dAmp = self.getValue('Amplitude #%d' % nType)
        dWidth = self.getValue('Width #%d' % nType)
        dPlateau = self.getValue('Plateau #%d' % nType)
        # get pulse width
        if sPulseType == 'Square':
            dTotTime = dWidth+dPlateau
        elif sPulseType == 'Ramp':
            dTotTime = 2*dWidth+dPlateau
        elif sPulseType == 'Gaussian':
            dTotTime = truncRange*dWidth + dPlateau
        # shift time to mid point if user gave start point
        if bTimeStart:
            dTime = dTime + dTotTime/2
        # get the range of indices in use
        vIndx = np.arange(max(np.round((dTime-dTotTime/2)*dSampleRate), 0),
                          min(np.round((dTime+dTotTime/2)*dSampleRate), len(self.vTime)))
        vIndx = np.int0(vIndx)
        self.log(len(vIndx), len(self.vTime))
        # calculate time values for the pulse indices
        vTime = vIndx/dSampleRate
        # calculate the actual value for the selected indices
        if sPulseType == 'Square':
            vPulse = (vTime >= (dTime-(dWidth+dPlateau)/2)) & \
                 (vTime < (dTime+(dWidth+dPlateau)/2))
        elif sPulseType == 'Ramp':
            # rising and falling slopes
            vRise = (vTime-(dTime-dPlateau/2-dWidth))/dWidth
            vRise[vRise<0.0] = 0.0
            vRise[vRise>1.0] = 1.0
            vFall = ((dTime+dPlateau/2+dWidth)-vTime)/dWidth
            vFall[vFall<0.0] = 0.0
            vFall[vFall>1.0] = 1.0
            vPulse = vRise * vFall
#            vPulse = np.min(1, np.max(0, (vTime-(dTime-dPlateau/2-dWidth))/dWidth)) * \
#               np.min(1, np.max(0, ((dTime+dPlateau/2+dWidth)-vTime)/dWidth))
        elif sPulseType == 'Gaussian':
            # width is two times std
            #dStd = dWidth/2;
            # alternate def; std is set to give total pulse area same as a square
            dStd = dWidth/np.sqrt(2*np.pi)
            # cut the tail part and increase the amplitude, if necessary
            dOffset = 0
            if dPlateau > 0:
                # add plateau
                vPulse = (vTime >= (dTime-dPlateau/2)) & \
                    (vTime < (dTime+dPlateau/2))
                if dStd > 0:
                    # before plateau
                    vPulse = vPulse + (vTime < (dTime-dPlateau/2)) * \
                        (np.exp(-(vTime-(dTime-dPlateau/2))**2/(2*dStd**2))-dOffset)/(1-dOffset)
                    # after plateau
                    vPulse = vPulse + (vTime >= (dTime+dPlateau/2)) * \
                        (np.exp(-(vTime-(dTime+dPlateau/2))**2/(2*dStd**2))-dOffset)/(1-dOffset)
            else:
                if dStd > 0:
                    vPulse = (np.exp(-(vTime-dTime)**2/(2*dStd**2))-dOffset)/(1-dOffset)
                else:
                    vPulse = np.zeros_like(vTime)
#        # add the pulse to the previous ones
#        vY[iPulse] = vY[iPulse] + dAmp * vPulse
        vPulse = dAmp * vPulse
        if start_at_zero:
            vPulse = vPulse - vPulse.min()
            vPulse = vPulse/vPulse.max()*dAmp
        # return both time, envelope, and indices
        return (vTime, vPulse, vIndx)


    def getPulseDuration(self, nType):
        """Get total pulse duration waveform, for timimg purposes"""
        # check if edge-to-edge
        if self.getValue('Edge-to-edge pulses'):
            width = self.getValue('Width #%d' % nType)
            plateau = self.getValue('Plateau #%d' % nType)
            pulseEdgeWidth = self.getValue('Edge position')
            return pulseEdgeWidth * width + plateau
        else:
            return 0.0
