from numpy.fft import fftfreq

from Base import SignalInterface

try:
    import arrayfire as af
except Exception as e:
    raise Exception('please install arrayfire as af')

try:
    af.set_backend('cuda')

except Exception as e:
    print('cuda not supported,cpu will be used to simulate the signal propagation in fiber')
    af.set_backend('cpu')

print('the simulation information is:', af.info())

from oInstrument.channel import LinearFiber
import numpy as np


class NonlinearFiber(LinearFiber):

    def __init__(self, alpha, D, length, gamma, reference_length = 1550,is_vector=True, **kwargs):
        super(NonlinearFiber, self).__init__(alpha,D,length,reference_length)
        self.gamma = gamma
        if is_vector:
            try:

                self.nplates = kwargs['nplates']
            except KeyError as e:
                print('the number of PMF is not set, default number 100 will be used')
                self.nplates = 100

            try:
                self.dgd = kwargs['dgd']
            except KeyError as e:
                self.dgd = 10
        self.step_length = kwargs['step_length']

    def length_eff(self, length):

        return (1 - np.exp(-self.alpha_lin * self.length)) / self.alpha_lin

    @property
    def betat(self):
        return 1

    def vector_prop(self, signal: SignalInterface.WdmSignal):
        print("The Mankov equation of CNLSE will be solved using split step fourier method")
        sample_x = signal.data_sample_in_fiber[0, :]
        sample_y = signal.data_sample_in_fiber[1, :]

        sample_x_gpu = af.from_ndarray(sample_x)
        sample_y_gpu = af.from_ndarray(sample_y)
        
        # signal_frequences = signal.absolute_frequences
        center_frequence = signal.center_frequence


        beta2 = self.beta2 + self.beta3 * (center_frequence - SignalInterface.Signal.lamb2freq(self.wave_length * 1e-9))

        freq = fftfreq(sample_y.shape[0],1/signal.fs)
        omeg = 2*np.pi*freq
        betat = 0.5 * omeg**2 * beta2
        dgdrms = np.sqrt((3*np.pi)/8)*self.dgd/np.sqrt(self.nplates)
        dgdrms = dgdrms * signal.signals[0].symbol_rate # s
        db1 = dgdrms * omeg
        self.__vector_prop(sample_x_gpu,sample_y_gpu,betat,self.gamma)


    def __vector_prop(self,ux,uy,betat,gamma):
        betat = af.from_ndarray(betat)
        sig0 = af.identity(2,2)
        sig2 = np.array([[0,1],[1,0]])
        sig2 = af.from_ndarray(sig2)
        sig3i = np.array([[0,1],[-1,0]])
        sig3i = af.from_ndarray(sig3i)

        gamma = gamma * 8/9
        lcorr = self.length/self.nplates
        halfalpha = 0.5 * self.alpha_lin

        ux ,uy = self.__vector_nonlinear_step(ux,uy,gamma)



    def __vector_nonlinear_step(self,ux,uy,gamma_mankorv):
        '''

        :param ux: samples of x-pol
        :param uy: samples of y-pol
        :param gamma_mankorv: 8/9*gamma，represent average over 那个啥球上
        :return: samples after nonlinear propagation
        '''
        leff = self.length_eff(self.step_length)
        power = af.abs(ux)**2 + af.abs(uy)**2
        gamma_mankorv = gamma_mankorv * leff
        ux = ux * af.exp(-1j*gamma_mankorv*power)
        uy = uy * af.exp(-1j*gamma_mankorv*power)
        return ux,uy





