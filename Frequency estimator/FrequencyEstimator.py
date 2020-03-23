import InstrumentDriver
import numpy as np

import sklearn
import sklearn.svm

class Driver(InstrumentDriver.InstrumentWorker):
    """ This class implements a simple signal generator driver"""
    

    def performOpen(self, options={}):
        self.state_vector = []
        self.tau = 1e-6
        self.scaling_0 = 0
        self.scaling_1 = 1
        self.points = 10
        self.step_between_points = 1
        self.time_between_points = -1
        self.xor_points = False
        """Perform the operation of opening the instrument connection"""
        pass


    def performClose(self, bError=False, options={}):
        """Perform the close instrument connection operation"""
        pass


    def performSetValue(self, quant, value, sweepRate=0.0, options={}):
        """Perform the Set Value instrument operation. This function should
        return the actual value set by the instrument"""
        # just return the value
        if quant.name.startswith('State vector, '):
            quant.setValue(value)
            self.state_vector = quant.getValueArray()
        if quant.name == 'tau':
            quant.setValue(value)
            self.tau = quant.getValue()
        if quant.name == 'Scaling 0':
            quant.setValue(value)
            self.scaling_0 = quant.getValue()
        if quant.name == 'Scaling 1':
            quant.setValue(value)
            self.scaling_1 = quant.getValue()
        if quant.name == 'Points in estimate':
            quant.setValue(value)
            self.points = int(quant.getValue())
        if quant.name == 'Step between points':
            quant.setValue(value)
            self.step_between_poins = int(quant.getValue())
        if quant.name == 'Time between points':
            quant.setValue(value)
            self.time_between_points = quant.getValue()
        if quant.name == 'XOR points':
            quant.setValue(value)
            self.xor_points = quant.getValue()
        return value


    def performGetValue(self, quant, options={}):
        """Perform the Get Value instrument operation"""
        # proceed depending on quantity
        if quant.name.startswith('Frequency estimate,'):
            state_vector = self.getValueArray('State vector, QB1')
            if self.getValue('XOR points'):
                state_vector = np.concatenate(([state_vector[0]],np.logical_xor(state_vector[0:-1],state_vector[1:])))
            return get_frequency(state_vector,self.getValue('tau'),self.getValue('Scaling 0'),self.getValue('Scaling 1'))
        if quant.name.startswith('Running frequency estimate, '):
            n_freqs = len(self.getValueArray('State vector, QB1')) - int(self.getValue('Points in estimate')) + 1
            frequencies = []
            state_vector = self.getValueArray('State vector, QB1')
            if self.getValue('XOR points'):
                state_vector = np.concatenate(([state_vector[0]],np.logical_xor(state_vector[0:-1],state_vector[1:])))
            for ii in range(0,n_freqs,int(self.getValue('Step between points'))):
                frequencies.append(get_frequency(state_vector[ii:ii+int(self.getValue('Points in estimate'))],self.getValue('tau'),self.getValue('Scaling 0'),self.getValue('Scaling 1')))
            # dt is the time actual step between points in running frequency estimate. If time between points is negative, step between points is used instead, making the time axis unitless.
            if self.time_between_points > 0:
                dt = self.getValue('Time between points')*self.getValue('Step between points')
            else:
                dt = self.getValue('Step between points')
            return quant.getTraceDict(np.array(frequencies),dt=dt)
            #return self.get_frequency()
        else: 
            # for other quantities, just return current value of control
            return quant.getValue()

    # def get_frequency(self):
    #     #return np.mean(self.state_vector)
    #     state_mean = np.mean(self.state_vector)
    #     if(state_mean>self.scaling_1):
    #         state_mean=1
    #     elif(state_mean<self.scaling_0):
    #         state_mean=0
    #     else:
    #         state_mean = (state_mean - self.scaling_0)/(self.scaling_1 - self.scaling_0)
    #     return np.arccos(2*state_mean-1)/self.tau/(2*np.pi)

def get_frequency(states,tau,scaling_0,scaling_1):
    if tau <= 0:
        raise Exception('Tau cannot be smaller than zero.')
    #return np.mean(self.state_vector)
    state_mean = np.mean(states)
    if(state_mean>scaling_1):
        state_mean=1
    elif(state_mean<scaling_0):
        state_mean=0
    else:
        state_mean = (state_mean - scaling_0)/(scaling_1 - scaling_0)
    return np.arccos(2*state_mean-1)/tau/(2*np.pi)