'''
    This file contains optical instrument along the signal's propagation
'''
import numpy as np
from scipy.constants import h, c
# from .. import Base
from Base.SignalInterface import QamSignal, Signal

import sys

print(sys.path)


class Edfa:

    def __init__(self, gain_db, nf, is_ase=True, mode='ConstantGain', expected_power=0):
        '''

        :param gain_db:
        :param nf:
        :param is_ase: 是否添加ase噪声
        :param mode: ConstantGain or ConstantPower
        :param expected_power: 当mode为ConstantPoower  时候，此参数有效
        '''

        self.gain_db = gain_db
        self.nf = nf
        self.is_ase = is_ase
        self.mode = mode
        self.expected_power = expected_power

    def one_ase(self, signal):
        '''

        :param signal:
        :return:
        '''

        lamb = (2 * max(signal.lamb) * min(signal.lamb)) / (max(signal.lamb) + min(signal.lamb))
        one_ase = (h * c / lamb) * (self.gain_lin * 10 ^ (self.nf_lin / 10) - 1) / 2
        return one_ase

    @property
    def gain_lin(self):
        return 10 ** (self.gain_db / 10)

    @property
    def nf_lin(self):
        return 10 ** (self.nf / 10)

    def traverse(self, signal):
        if self.is_ase:
            noise = self.one_ase(signal) * signal.fs
        else:
            noise = 0

        if self.mode == 'ConstantGain':
            signal.data_sample = np.sqrt(self.gain_lin) * signal.data_sample + noise
            return
        if self.mode == 'ConstantPower':
            signal_power = np.mean(np.abs(signal.data_sample[0, :]) ** 2) + np.mean(
                np.abs(signal.data_sample[1, :]) ** 2)
            desired_power_linear = (10 ** (self.expected_power / 10)) / 1000
            linear_gain = desired_power_linear / signal_power
            signal.data_sample = np.sqrt(linear_gain) * signal.data_sample + noise

    def __call__(self, signal):
        self.traverse(signal)

    def __str__(self):

        string = f"Model is {self.mode}\n" \
            f"Gain is {self.gain_db} db\n" \
            f"ase is {self.is_ase}\n" \
            f"noise figure is {self.nf}"
        return string

    def __repr__(self):
        return self.__str__()


class LaserSource:

    def __init__(self, laser_power, line_width, is_phase_noise, center_frequence):
        '''

        :param laser_power: [dbm]
        :param line_width: [hz]
        :param is_phase_noise:[bool]
        '''
        self.laser_power = laser_power
        self.line_width = line_width
        self.is_phase_noise = is_phase_noise
        self.center_frequence = center_frequence

    @property
    def linear_laser_power(self):
        return (10 ** (self.laser_power / 10)) * 0.001

    def __call__(self, signal: Signal):
        signal.signal_power = self.linear_laser_power
        signal.normalize_power()
        signal.data_sample_in_fiber = np.sqrt(self.linear_laser_power) * signal.data_sample_in_fiber
        signal.lamb = Signal.freq2lamb(self.center_frequence)
        if self.is_phase_noise:
            initial_phase = -np.pi + 2 * np.pi * np.random.randn(1)
            dtheta = np.sqrt(2 * np.pi * 1 / signal.fs_in_fiber * self.line_width) * np.random.randn(1,
                                                                                                     signal.data_sample_in_fiber.shape[
                                                                                                         1])
            dtheta[0, 0] = 0
            phase_noise = initial_phase + np.cumsum(dtheta, axis=1)

            signal[:] = signal.data_sample_in_fiber * np.exp(1j * phase_noise)


class IQ:

    def __init__(self, iq_ratio, iq_difference, i_signal, q_signal, **kwargs):
        self.i_signal = i_signal
        self.q_signal = q_signal
        self.iq_ratio = iq_ratio
        self.iq_difference = iq_difference

        self.mzm1_config = kwargs['mzm1']
        self.mzm2_config = kwargs['mzm2']

    def prop(self, laser_signal):
        iqratio_amplitude = 10 ** (self.iq_ratio / 20)
        laser_signal_i = laser_signal * (iqratio_amplitude / (iqratio_amplitude + 1))
        laser_signal_q = laser_signal * (1 - iqratio_amplitude / (iqratio_amplitude + 1))

        mzm1 = MZM(laser_signal_i, self.i_signal, **self.mzm1_config)
        mzm2 = MZM(laser_signal_q, self.q_signal, **self.mzm2_config)

        i_waveform = mzm1()
        q_waveform = mzm2()

        return i_waveform + q_waveform * np.exp(1j * np.pi / 2 + 1j * self.iq_difference)

    def __call__(self, laser_signal):
        return self.prop(laser_signal)


class MZM:

    def __init__(self, optical_signal, electrical_signal, vpi=5, extinction_ratio=np.inf, vbias=5 / 2):
        self.optical_signal = optical_signal
        self.electrical_signal = electrical_signal
        self.vpi = vpi
        self.extinction_ratio = extinction_ratio
        self.vbias = vbias

    def traverse(self):

        if self.extinction_ratio is not np.inf:
            ER = 10 ** (self.extinction_ratio / 10)
            eplio = np.sqrt(1 / ER)
        else:
            eplio = 0

        ysplit_upper = np.sqrt(0.5 + eplio)
        ysplit_lowwer = np.sqrt(0.5 - eplio)

        phi_upper = np.pi * self.electrical_signal / (self.vpi)
        phi_lower = np.pi * (-self.electrical_signal + self.vbias) / self.vpi

        # attuenation = 10 ** (self.insertloss / 10)

        h = ysplit_upper * np.exp(1j * phi_upper) + ysplit_lowwer * np.exp(1j * phi_lower)
        # h = h / attuenation
        h = (1 / np.sqrt(2)) * h
        sample_out = h * self.optical_signal
        return sample_out

    def __call__(self):
        return self.traverse()


class OpticalFilter:
    pass


if __name__ == '__main__':
    symbol_rate = 35e9
    mf = '16-qam'
    signal_power = 0
    symbol_length = 2 ** 16
    sps = 2
    sps_infiber = 4

    parameter = dict(symbol_rate=symbol_rate, mf=mf, symbol_length=symbol_length, sps=sps,
                     sps_in_fiber=sps_infiber)

    signal = QamSignal(**parameter)
    signal[:] = np.array([[1, 2, 3, 4, 5]])

    laser = LaserSource(0, 0.001, True, 193.1e12)
    laser(signal)
