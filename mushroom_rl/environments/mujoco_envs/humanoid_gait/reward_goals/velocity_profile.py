import warnings

import numpy as np
from scipy.signal import square


class VelocityProfile:
    """
    Interface that represents and handles the velocity profile of the center of
    mass of the humanoid that must be matched at each timestep.

    """
    def __init__(self, velocity_profile_array, timestep):
        """
        Constructor.

        Args:
            velocity_profile_array (np.ndarray): velocity of the center at each
                timestep;
            timestep (float): time corresponding to each step of simulation.

        """
        self._velocity_profile_array = velocity_profile_array
        self._timestep = timestep

    @property
    def values(self):
        """
        Returns:
             The velocity profile.

        """
        return self._velocity_profile_array

    @property
    def timestep(self):
        """
        Returns:
            The time corresponding to each step of simulation.

        """
        return self._timestep

    @property
    def size(self):
        """
        Returns:
             The length of the velocity profile.

        """
        return self._velocity_profile_array.size

    def reset(self):
        """
        Create a new velocity profile, if needed.

        Returns:
            The new velocity profile.

        """
        return self._velocity_profile_array


class PeriodicVelocityProfile(VelocityProfile):
    """
    Interface that represents a cyclic velocity profile.

    """
    def __init__(self, velocity_profile_array, period, timestep):
        """
        Constructor.

        Args:
            velocity_profile_array (np.ndarray): velocity of the center at each
                timestep;
            period (float): time corresponding to one cycle;
            timestep (float): time corresponding to each step of simulation.

        """
        if 1 / timestep < 2 * (1 / period):
            raise ValueError("This timestep doesn't respect the Nyquist theorem"
                             "for this given period")

        sampling = period / timestep
        rest = sampling - int(sampling)
        if rest != 0:
            warnings.warn(
                'Velocity Profile doesnt have a full period or a set of full'
                'periods. There will be some desync due to sampling.')

        super().__init__(velocity_profile_array, timestep)


class SinVelocityProfile(PeriodicVelocityProfile):
    """
    Interface that represents velocity profile with a sine shape.

    """
    def __init__(self, amplitude, period, timestep, offset=0, phase=0):
        """
        Constructor.

        Args:
            amplitude (np.ndarray): amplitude of the sine wave;
            period (float): time corresponding to one cycle;
            timestep (float): time corresponding to each step of simulation;
            offset (float, 0): increment of velocity to each velocity value;
            phase (float, 0): angle in rads of the phase of the sine wave.

        """
        time_array = np.arange(0, period, timestep)
        phase_array = 2 * np.pi * (time_array / period)
        phase_array += phase
        wave = amplitude * np.sin(phase_array) + offset
        super(SinVelocityProfile, self).__init__(wave, period, timestep)


class ConstantVelocityProfile(VelocityProfile):
    """
    Interface that represents velocity profile with constant value.

    """
    def __init__(self, value):
        """
        Constructor.

        Args:
            value (float): constant value of the velocity profile.

        """
        super(ConstantVelocityProfile, self).__init__(np.array([value]), 0.0)


class RandomConstantVelocityProfile(ConstantVelocityProfile):
    """
    Interface that represents velocity profile with a constant value
    per episode but random limited between two values between each episode.

    """
    def __init__(self, min, max):
        """
        Constructor.

        Args:
            min (float): minimum value of the velocity profile.
            max (float): maximum value of the velocity profile.

        """
        self._max = max
        self._min = min
        super().__init__(self.get_random_val())

    def reset(self):
        self._velocity_profile_array[:] = self.get_random_val()
        return super().reset()

    def get_random_val(self):
        return np.random.random() * (self._max - self._min) + self._min


class SquareWaveVelocityProfile(PeriodicVelocityProfile):
    """
    Interface that represents velocity profile with a square wave shape.

    """
    def __init__(self, amplitude, period, timestep, duty=0.5, offset=0,
                 phase=0):
        """
        Constructor.

        Args:
            amplitude (np.ndarray): amplitude of the square wave;
            period (float): time corresponding to one cycle;
            timestep (float): time corresponding to each step of simulation;
            duty (float, 0.5): value between 0 and 1 and determines the relative
                time that the step transition occurs between the start and the
                end of the cycle;
            offset (float, 0): increment of velocity to each velocity value;
            phase (float, 0): angle in rads of the phase of the sine wave.

        """
        time_array = np.arange(0, period, timestep)
        phase_array = 2 * np.pi * (time_array / period)
        phase_array += phase
        wave = amplitude * square(phase_array, duty) + offset
        super(SquareWaveVelocityProfile, self).__init__(wave, period, timestep)


class VelocityProfile3D:
    """
    Class that represents the ensemble of velocity profiles of the center
    of mass of the Humanoid on 3 axis (X, Y, Z).

    """
    def __init__(self, velocity_profiles):
        """
        Constructor.

        Args:
            velocity_profiles (list): list of ``VelocityProfile`` instances.

        """
        self._profileslist = velocity_profiles

        timestep = None
        size = None

        for i in range(len(self._profileslist)):
            if not isinstance(self._profileslist[i], ConstantVelocityProfile):
                if timestep is None:
                    timestep = self._profileslist[i].timestep
                else:
                    if timestep != self._profileslist[i].timestep:
                        raise ValueError('Values of timesteps differ in'
                                         'velocity profiles')

                if size is None:
                    size = self._profileslist[i].size
                else:
                    if size != self._profileslist[i].size:
                        raise ValueError('Size of values buffer differ in'
                                         'velocity profiles')

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
