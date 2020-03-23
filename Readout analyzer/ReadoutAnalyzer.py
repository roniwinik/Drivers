import InstrumentDriver
import numpy as np

import sklearn
import sklearn.svm

class Driver(InstrumentDriver.InstrumentWorker):
    """ This class implements a simple signal generator driver"""
    

    def performOpen(self, options={}):
        self.data_valid = True
        self.ground_state_vector = None
        self.excited_state_vector = None

        """Perform the operation of opening the instrument connection"""
        pass


    def performClose(self, bError=False, options={}):
        """Perform the close instrument connection operation"""
        pass


    def performSetValue(self, quant, value, sweepRate=0.0, options={}):
        """Perform the Set Value instrument operation. This function should
        return the actual value set by the instrument"""
        # just return the value
        if quant.name.startswith('Single-shot input,'):
            quant.setValue(value)
            if(self.data_valid):
                self.ground_state_vector = quant.getValueArray()
                self.data_valid = False
            else:
                self.excited_state_vector = quant.getValueArray()
                self.data_valid = True
        else:
            return value


    def performGetValue(self, quant, options={}):
        """Perform the Get Value instrument operation"""
        # proceed depending on quantity
        if quant.name.startswith('Readout fidelity, '):
            if(self.data_valid is True):
                return self.get_fidelity()
            else:
                return -1
        else: 
            # for other quantities, just return current value of control
            return quant.getValue()


    def get_fidelity(self):
        if((self.ground_state_vector is not None) and (self.excited_state_vector is not None)):
            data = np.array([self.ground_state_vector,self.excited_state_vector])
            return calc_fid(data)
        else:
            return -1


def calc_fid(measurement_data):
    #data =  np.ndarray.flatten(measurement_data[x,:,:])
    data = np.ndarray.flatten(measurement_data,order='C')
    value = [0 for _ in range(data.shape[0]//2)] + [1 for _ in range(data.shape[0]//2)]
    return assignment_fid(value,data)

def assignment_fid( value , data ):
    # value is a list of 0 or 1
    # data is a list of (I,Q) data
    #t0 = time.time()
    train_data = [0 for _ in range(len(data))]
    if type(data[0]) != list:
        for i in range(len(data)):
            train_data[i] = (np.real(data[i]) , np.imag(data[i]) )
    else:
        train_data = data
    #print('conversion',time.time() - t0)
    #t0 = time.time()
    model = sklearn.svm.LinearSVC()
    model.fit(train_data , value)
    #print('training',time.time() - t0)
    return model.score(train_data , value)