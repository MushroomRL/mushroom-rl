
import numpy as np
from .humanmuscle import HAB, HAM, HAD, HFL, BFSH, GAS, GLU, REF, VAS, SOL, TIA
from .ExternalSimulation import ExternalSimulationInterface


class MuscleSimulation(ExternalSimulationInterface):
    def __init__(self, sim):
        super().__init__(sim)
        self.musc = self._create_muscles()  # container with all muscle objects [muscle name: muscle object]
        self.nmusc = len(self.musc)

        self._metcost = 0  # metabolic cost of all muscles for an action
        self.reset()

    def get_action_space(self):
        high = np.ones(self.nmusc)
        low = np.ones(self.nmusc) * 0.001
        return low, high

    def get_observation_space(self):
        obs = self.get_observation()
        high = np.inf * np.ones(len(obs))
        low = -np.inf * np.ones(len(obs))
        return low, high

    def reward(self, state, action, next_state):
        # returns cost value between [0, 1]
        reward = 1 - (5 * self._metcost)
        return reward

    def is_absorving(self, state):
        return False

    def update_state(self):
        self._metcost = np.exp(-self._metcost / 50)

    def get_observation(self):
        musc_obs = np.zeros(self.nmusc * 4)
        for m, musc in enumerate(self.musc.values()):
            musc_obs[m + (0 * self.nmusc)] = musc.lce
            musc_obs[m + (1 * self.nmusc)] = musc.vce
            musc_obs[m + (2 * self.nmusc)] = musc.frcmtc / 1000.0
            musc_obs[m + (3 * self.nmusc)] = musc.act
        return musc_obs

    def _create_muscles(self):
        angHipFroR, angHipSagR, angKneR, angAnkR, \
        angHipFroL, angHipSagL, angKneL, angAnkL = self.sim.data.qpos[7:15]

        angHipAbdR = -angHipFroR
        angHipAbdL = angHipFroL
        angHipSagR = angHipSagR + np.pi
        angHipSagL = angHipSagL + np.pi
        angKneR = np.pi - angKneR
        angKneL = np.pi - angKneL
        angAnkR = angAnkR + np.pi / 2.0
        angAnkL = angAnkL + np.pi / 2.0

        timestep = self.sim.model.opt.timestep
        musc = {"HABR":  HAB(angHipAbdR, timestep),
                "HADR":  HAD(angHipAbdR, timestep),
                "GLUR":  GLU(angHipSagR, timestep),
                "HFLR":  HFL(angHipSagR, timestep),
                "HAMR":  HAM(angHipSagR, angKneR, timestep),
                "REFR":  REF(angHipSagR, angKneR, timestep),
                "BFSHR": BFSH(angKneR, timestep),
                "VASR":  VAS(angKneR, timestep),
                "GASR":  GAS(angKneR, angAnkR, timestep),
                "SOLR":  SOL(angAnkR, timestep),
                "TIAR":  TIA(angAnkR, timestep),
                "HABL":  HAB(angHipAbdL, timestep),
                "HADL":  HAD(angHipAbdL, timestep),
                "GLUL":  GLU(angHipSagL, timestep),
                "HFLL":  HFL(angHipSagL, timestep),
                "HAML":  HAM(angHipSagL, angKneL, timestep),
                "REFL":  REF(angHipSagL, angKneL, timestep),
                "BFSHL": BFSH(angKneL, timestep),
                "VASL":  VAS(angKneL, timestep),
                "GASL":  GAS(angKneL, angAnkL, timestep),
                "SOLL":  SOL(angAnkL, timestep),
                "TIAL":  TIA(angAnkL, timestep),
                }
        return musc

    def reset(self):
        self.musc = self._create_muscles()

    def preprocess_action(self, stimu):
        return np.clip(stimu, 0.001, 1.0)

    def initialize_internal_states(self, state, stimu):
        self._metcost = 0
        for i, musc in enumerate(self.musc.values()):
            musc.reset_state()

        self.musc["HABR"].stim = stimu[0]
        self.musc["HADR"].stim = stimu[1]
        self.musc["HFLR"].stim = stimu[2]
        self.musc["GLUR"].stim = stimu[3]
        self.musc["HAMR"].stim = stimu[4]
        self.musc["REFR"].stim = stimu[5]
        self.musc["VASR"].stim = stimu[6]
        self.musc["BFSHR"].stim = stimu[7]
        self.musc["GASR"].stim = stimu[8]
        self.musc["SOLR"].stim = stimu[9]
        self.musc["TIAR"].stim = stimu[10]
        self.musc["HABL"].stim = stimu[11]
        self.musc["HADL"].stim = stimu[12]
        self.musc["HFLL"].stim = stimu[13]
        self.musc["GLUL"].stim = stimu[14]
        self.musc["HAML"].stim = stimu[15]
        self.musc["REFL"].stim = stimu[16]
        self.musc["VASL"].stim = stimu[17]
        self.musc["BFSHL"].stim = stimu[18]
        self.musc["GASL"].stim = stimu[19]
        self.musc["SOLL"].stim = stimu[20]
        self.musc["TIAL"].stim = stimu[21]

    def external_stimulus_to_joint_torques(self, stimu):
        angHipFroR, angHipSagR, angKneR, angAnkR, \
        angHipFroL, angHipSagL, angKneL, angAnkL = self.sim.data.qpos[7:15]

        angHipAbdR = -angHipFroR
        angHipAbdL = angHipFroL
        angHipSagR = angHipSagR + np.pi
        angHipSagL = angHipSagL + np.pi
        angKneR = np.pi - angKneR
        angKneL = np.pi - angKneL
        angAnkR = angAnkR + np.pi / 2.0
        angAnkL = angAnkL + np.pi / 2.0

        self.musc["HABR"].stepUpdateState(np.array((angHipAbdR,)))
        self.musc["HADR"].stepUpdateState(np.array((angHipAbdR,)))
        self.musc["GLUR"].stepUpdateState(np.array((angHipSagR,)))
        self.musc["HFLR"].stepUpdateState(np.array((angHipSagR,)))
        self.musc["HAMR"].stepUpdateState(np.array((angHipSagR, angKneR)))
        self.musc["REFR"].stepUpdateState(np.array((angHipSagR, angKneR)))
        self.musc["BFSHR"].stepUpdateState(np.array((angKneR,)))
        self.musc["VASR"].stepUpdateState(np.array((angKneR,)))
        self.musc["GASR"].stepUpdateState(np.array((angKneR, angAnkR)))
        self.musc["SOLR"].stepUpdateState(np.array((angAnkR,)))
        self.musc["TIAR"].stepUpdateState(np.array((angAnkR,)))
        self.musc["HABL"].stepUpdateState(np.array((angHipAbdL,)))
        self.musc["HADL"].stepUpdateState(np.array((angHipAbdL,)))
        self.musc["GLUL"].stepUpdateState(np.array((angHipSagL,)))
        self.musc["HFLL"].stepUpdateState(np.array((angHipSagL,)))
        self.musc["HAML"].stepUpdateState(np.array((angHipSagL, angKneL)))
        self.musc["REFL"].stepUpdateState(np.array((angHipSagL, angKneL)))
        self.musc["BFSHL"].stepUpdateState(np.array((angKneL,)))
        self.musc["VASL"].stepUpdateState(np.array((angKneL,)))
        self.musc["GASL"].stepUpdateState(np.array((angKneL, angAnkL)))
        self.musc["SOLL"].stepUpdateState(np.array((angAnkL,)))
        self.musc["TIAL"].stepUpdateState(np.array((angAnkL,)))

        self._metcost += np.sum([musc.MR for musc in self.musc.values()])

        torHipAbdR = self.musc["HABR"].frcmtc * self.musc["HABR"].levelArm - self.musc["HADR"].frcmtc * self.musc["HADR"].levelArm

        torHipExtR = self.musc["GLUR"].frcmtc * self.musc["GLUR"].levelArm - self.musc["HFLR"].frcmtc * self.musc["HFLR"].levelArm + \
                     self.musc["HAMR"].frcmtc * self.musc["HAMR"].levelArm[0] - self.musc["REFR"].frcmtc * \
                     self.musc["REFR"].levelArm[0]

        torKneFleR = self.musc["BFSHR"].frcmtc * self.musc["BFSHR"].levelArm - self.musc["VASR"].frcmtc * self.musc["VASR"].levelArm + \
                     self.musc["HAMR"].frcmtc * self.musc["HAMR"].levelArm[1] - self.musc["REFR"].frcmtc * \
                     self.musc["REFR"].levelArm[1] + self.musc["GASR"].frcmtc * self.musc["GASR"].levelArm[0]

        torAnkExtR = self.musc["SOLR"].frcmtc * self.musc["SOLR"].levelArm - self.musc["TIAR"].frcmtc * self.musc["TIAR"].levelArm + \
                     self.musc["GASR"].frcmtc * self.musc["GASR"].levelArm[1]

        torHipAbdL = self.musc["HABL"].frcmtc * self.musc["HABL"].levelArm - self.musc["HADL"].frcmtc * self.musc["HADL"].levelArm

        torHipExtL = self.musc["GLUL"].frcmtc * self.musc["GLUL"].levelArm - self.musc["HFLL"].frcmtc * self.musc["HFLL"].levelArm + \
                     self.musc["HAML"].frcmtc * self.musc["HAML"].levelArm[0] - self.musc["REFL"].frcmtc * \
                     self.musc["REFL"].levelArm[0]

        torKneFleL = self.musc["BFSHL"].frcmtc * self.musc["BFSHL"].levelArm - self.musc["VASL"].frcmtc * self.musc["VASL"].levelArm + \
                     self.musc["HAML"].frcmtc * self.musc["HAML"].levelArm[1] - self.musc["REFL"].frcmtc * \
                     self.musc["REFL"].levelArm[1] +  self.musc["GASL"].frcmtc * self.musc["GASL"].levelArm[0]

        torAnkExtL = self.musc["SOLL"].frcmtc * self.musc["SOLL"].levelArm - self.musc["TIAL"].frcmtc * self.musc["TIAL"].levelArm + \
                     self.musc["GASL"].frcmtc * self.musc["GASL"].levelArm[1]

        tor = [-torHipAbdR, torHipExtR, torKneFleR, torAnkExtR,
               torHipAbdL, torHipExtL, torKneFleL, torAnkExtL]
        return np.squeeze(tor)