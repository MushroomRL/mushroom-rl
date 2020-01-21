import numpy as np

"""
It is created for using MTC in Mujoco. The dynamics in this model is not continuous. The integration error will be
accumulated overtime. And the system might get unstable if the timestep is too large. It is recommended to set the
timestamp lower than 5e-4 to get decent results.

The model is created based on Song's and Geyer's 2015 paper:
Song, S. and Geyer, H., 2015. A neural circuitry that emphasizes spinal feedback generates diverse behaviours of human
locomotion. The Journal of physiology, 593(16), pp.3493-3511.

V0.1
Passed basic tests. There're slightly difference compared to the simmechanics model.

V0.2
1. Verified with the simmechanics model. Difference in most of the cases can be ignored.
2. Changed the integration method from forward Euler to trapezoid.
3. Muscle force vce etc might vibrate/jitter if in some cases if the timestep is not low enough.
   Need to improve this in the next version.
   
"""

class MuscleTendonComplex:
    def __init__(self, paraMuscle, stateMuscle, paraMusAttach, offsetCorr, timestep, nameMuscle, angJoi):
        self.frcmax, self.vmax, self.eref, self.lslack, self.lopt, self.tau, self.w, self.c, self.N, self.K = paraMuscle
        self.stim, self.act, self.lmtc, self.lce, self.vce, self.frcmtc = stateMuscle
        self.timestep = timestep
        self.nameMuscle = nameMuscle
        self.angJoi = angJoi
        self.offsetCorr = offsetCorr
        self.r, self.phiref, self.phimaxref, self.rho, self.dirAng, self.phiScale = paraMusAttach
        self.MR =  0.01
        self.typeMuscle = self.angJoi.size
        nJoi = self.typeMuscle
        self.levelArm = np.zeros(nJoi)

        tmpL = np.zeros(nJoi)
        for i in range(0, nJoi):
            if self.offsetCorr[i] == 0:
                tmpL[i] = self.dirAng[i] * (self.angJoi[i] - self.phiref[i]) * self.r[i] * self.rho[i]
                self.levelArm[i] = self.r[i]
            elif self.offsetCorr[i] == 1:
                tmp1 = np.sin((self.phiref[i] - self.phimaxref[i]) * self.phiScale[i])
                tmp2 = np.sin((self.angJoi[i] - self.phimaxref[i]) * self.phiScale[i])
                tmpL[i] = self.dirAng[i] * (tmp2 - tmp1) * self.r[i] * self.rho[i] / self.phiScale[i]
                self.levelArm[i] = np.cos((self.angJoi[i] - self.phimaxref[i]) * self.phiScale[i]) * self.r[i]
            else:
                raise ValueError('Invalid muscle level arm offset correction type. ')
        self.lmtc = self.lslack + self.lopt + np.sum(tmpL)

        self.lce = self.lmtc - self.lslack
        self.lse = self.lmtc - self.lce
        # unitless parameters
        self.Lse = self.lse / self.lslack
        self.Lce = self.lce / self.lopt

        self.actsubstep = (self.stim - self.act) * self.timestep / 2.0 / self.tau + self.act
        self.lcesubstep = self.vce * self.timestep / 2.0 + self.lce

        # test
        self.lce_avg = self.lce
        self.vce_avg = self.vce
        self.frcmtc_avg = 0
        self.act_avg = self.act
        self.frame = 0
        # self.Fse = 0.0
        # self.Fbe = 0.0
        # self.Fpe = 0.0
        # self.Fce = 0.0


    def stepUpdateState(self, angJoi):
        """
        Muscle Tendon Complex Dynamics
        update muscle states based on the muscle dynamics
        Muscle state stim has to be updated outside before this function is called
        """
        # update lmtc and level arm based on the geometry
        self.angJoi = angJoi
        nJoi = self.typeMuscle
        tmpL = np.zeros(nJoi)
        for i in range(0, nJoi):
            if self.offsetCorr[i] == 0:
                tmpL[i] = self.dirAng[i] * (self.angJoi[i] - self.phiref[i]) * self.r[i] * self.rho[i]
                self.levelArm[i] = self.r[i]
            elif self.offsetCorr[i] == 1:
                tmp1 = np.sin((self.phiref[i] - self.phimaxref[i]) * self.phiScale[i])
                tmp2 = np.sin((self.angJoi[i] - self.phimaxref[i]) * self.phiScale[i])
                tmpL[i] = self.dirAng[i] * (tmp2 - tmp1) * self.r[i] * self.rho[i] / self.phiScale[i]
                self.levelArm[i] = np.cos((self.angJoi[i] - self.phimaxref[i]) * self.phiScale[i]) * self.r[i]
            else:
                raise ValueError('Invalid muscle level arm offset correction type. ')
        self.lmtc = self.lslack + self.lopt + np.sum(tmpL)

        # update muscle activation
        # integration, forward-Euler method
        # self.act = (self.stim - self.act) * self.timestep / self.tau + self.act
        # integration, trapezoidal method, 2-step
        self.act = (self.stim - self.actsubstep) * self.timestep / 2.0 / self.tau + self.actsubstep
        self.actsubstep = (self.stim - self.act) * self.timestep / 2.0 / self.tau + self.act

        # update lce and lse based on the lmtc
        # integration, forward-Euler method
        # self.lce = self.vce * self.timestep + self.lce
        # integration, trapezoidal method, 2-step
        self.lce = self.vce * self.timestep / 2.0 + self.lcesubstep
        self.lcesubstep = self.vce * self.timestep / 2.0 + self.lce

        self.lse = self.lmtc - self.lce
        self.Lse = self.lse / self.lslack
        self.Lce = self.lce / self.lopt

        # Serial Elastic element (tendon) force-length relationship
        if self.Lse > 1.0:
            Fse = np.power((self.Lse - 1.0) / self.eref, 2)
        else:
            Fse = 0.0

        # Parallel Elasticity PE
        if self.Lce > 1.0:
            Fpe = np.power((self.Lce - 1.0) / self.w, 2)
        else:
            Fpe = 0.0

        # update frcmtc
        self.frcmtc = Fse * self.frcmax
        #self.frcmtc =  np.clip(self.frcmtc, 0, self.frcmax)

        # Buffer Elasticity BE
        if (self.Lce - (1.0 - self.w)) < 0:
            Fbe = np.power((self.Lce - (1.0 - self.w)) / (self.w / 2), 2)
        else:
            Fbe = 0.0

        # Contractile Element force-length relationship
        tmp = np.power(np.absolute(self.Lce - 1.0) / self.w, 3)
        Fce = np.exp(tmp * np.log(self.c))

        #Fv = (Fse + Fbe) / (Fpe + Fce * self.act)
        if (Fpe + Fce * self.act) < 1e-10:  # avoid numerical error
            if (Fse + Fbe) < 1e-10:
                Fv = 1.0
            else:
                Fv = (Fse + Fbe) / 1e-10
        else:
            Fv = (Fse + Fbe) / (Fpe + Fce * self.act)

        # Contractile Element inverse force-velocity relationship
        if Fv <= 1.0:
            # Concentric
            v = (Fv - 1) / (Fv * self.K + 1.0)
        elif Fv <= self.N:
            # excentric
            tmp = (Fv - self.N) / (self.N - 1.0)
            v = (tmp + 1.0) / (1.0 - tmp * 7.56 * self.K)
        else:
            # excentric overshoot
            v = ((Fv - self.N) * 0.01 + 1)

        self.vce = v * self.lopt * self.vmax
        v_frac = self.vce /  self.vmax
        mr_scale =  self.act * np.absolute(self.frcmax*self.vmax) *self.timestep
        if self.vce <= 1:
            self.MR =  0.01 - 0.11*(v_frac) + 0.06*np.exp(-8*v_frac)
        else:
            self.MR =  0.23 - 0.16*np.exp(-8*v_frac) 
        self.MR *= mr_scale
        self.frame += 1
        self.lce_avg = (self.lce_avg*(self.frame - 1) +  self.lce) / self.frame
        self.vce_avg = (self.vce_avg*(self.frame - 1) +  self.vce) / self.frame
        self.frcmtc_avg = (self.frcmtc_avg*(self.frame - 1) +  self.frcmtc) / self.frame
        self.act_avg = (self.act_avg*(self.frame - 1) +  self.act) / self.frame
        #self.MR = np.exp(-self.MR)
        # print(self.MR, np.exp(-self.MR))
        # self.Fv = Fv
        # self.Fse = Fse
        # self.Fbe = Fbe
        # self.Fpe = Fpe
        # self.Fce = Fce

    def reset_state(self):
        self.frame = 0
        self.lce_avg = 0
        self.frcmtc_avg = 0
        self.act_avg = 0
        self.vce_avg = 0

