import warnings

import numpy as np
from scipy.signal import square


class VelocityProfile:
    def __init__(self, velocity_profile_array, timestep):
        self._velocity_profile_array = velocity_profile_array
        self._timestep = timestep

    @property
    def values(self):
        return self._velocity_profile_array

    @property
    def timestep(self):
        return self._timestep

    @property
    def size(self):
        return self._velocity_profile_array.size

    def reset(self):
        return self._velocity_profile_array


class PeriodicVelocityProfile(VelocityProfile):
    def __init__(self, velocity_profile_array: np.ndarray, period: (float, int), timestep: (float, int)):

        if 1 / timestep < 2 * (1 / period):
            raise ValueError("This timestep doesn't respect the Nyquist theorem for this given period")

        sampling = period / timestep
        rest = sampling - int(sampling)
        if rest != 0:
            warnings.warn(
                'Velocity Profile doesnt have a full period or a set of full periods. There will be some desync due to sampling.')
        super().__init__(velocity_profile_array, timestep)


class SinVelocityProfile(PeriodicVelocityProfile):
    def __init__(self, amplitude, period, timestep, offset=0, phase=0):
        time_array = np.arange(0, period, timestep)
        phase_array = 2 * np.pi * (time_array / period)
        phase_array += phase
        wave = amplitude * np.sin(phase_array) + offset
        super(SinVelocityProfile, self).__init__(wave, period, timestep)


class ConstantVelocityProfile(VelocityProfile):
    def __init__(self, value):
        super(ConstantVelocityProfile, self).__init__(np.array([value]), None)


class RandomConstantVelocityProfile(ConstantVelocityProfile):
    def __init__(self, min, max):
        self._max = max
        self._min = min
        super().__init__(self.get_random_val())

    def reset(self):
        self._velocity_profile_array[:] = self.get_random_val()
        return super().reset()

    def get_random_val(self):
        return np.random.random() * (self._max - self._min) + self._min


class SquareWaveVelocityProfile(PeriodicVelocityProfile):
    def __init__(self, amplitude, period, timestep, duty=0.5, offset=0, phase=0):
        time_array = np.arange(0, period, timestep)
        phase_array = 2 * np.pi * (time_array / period)
        phase_array += phase
        wave = amplitude * square(phase_array, duty) + offset
        super(SquareWaveVelocityProfile, self).__init__(wave, period, timestep)


class VelocityProfile3D:
    def __init__(self, velocity_profiles):
        self._profileslist = velocity_profiles

        timestep = None
        size = None

        for i in range(len(self._profileslist)):
            if not isinstance(self._profileslist[i], ConstantVelocityProfile):
                if timestep is None:
                    timestep = self._profileslist[i].timestep
                else:
                    if timestep != self._profileslist[i].timestep:
                        raise ValueError('Values of timesteps differ in velocity profiles')

                if size is None:
                    size = self._profileslist[i].size
                else:
                    if size != self._profileslist[i].size:
                        raise ValueError('Size of values buffer differ in velocity profiles')

        if size == None:
            size = 1

        self._timestep = timestep
        self._size = size

    @property
    def values(self):
        values = []
        for profile in self._profileslist:
            if isinstance(profile, ConstantVelocityProfile):
                vals = np.tile(profile.values, (self.size))
            else:
                vals = profile.values
            values.append(vals)
        return np.vstack(values).T

    @property
    def timestep(self):
        return self._timestep

    @property
    def size(self):
        return self._size

    def reset(self):
        for profile in self._profileslist:
            profile.reset()
        return self.values
