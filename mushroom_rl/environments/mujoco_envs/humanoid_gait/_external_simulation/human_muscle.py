import numpy as np
from .muscle_simulation_stepupdate import step_update_state


class MuscleTendonComplex:
    def __init__(self, nameMuscle, frcmax, vmax, lslack, lopt,
                 lce, r, phiref, phimaxref, rho, dirAng, phiScale,
                 offsetCorr, timestep, angJoi, eref=0.04, act=0.01,
                 tau=0.01, w=0.56, c=0.05, N=1.5, K=5.0, stim=0.0,
                 vce=0.0, frcmtc=0.0, lmtc=0.0):

        self.init_nameMuscle = nameMuscle
        self.init_frcmax = float(frcmax)
        self.init_vmax = float(vmax)
        self.init_eref = float(eref)
        self.init_lslack = float(lslack)
        self.init_lopt = float(lopt)
        self.init_tau = float(tau)
        self.init_w = float(w)
        self.init_c = float(c)
        self.init_N = float(N)
        self.init_K = float(K)
        self.init_stim = float(stim)
        self.init_act = float(act)
        self.init_lmtc = float(lmtc)
        self.init_lce = float(lce)
        self.init_vce = float(vce)
        self.init_frcmtc = float(frcmtc)
        self.init_r = r.astype('float')
        self.init_phiref = phiref.astype('float')
        self.init_phimaxref = phimaxref.astype('float')
        self.init_rho = rho.astype('float')
        self.init_dirAng = dirAng.astype('float')
        self.init_phiScale = phiScale.astype('float')
        self.init_offsetCorr = offsetCorr.astype('int')
        self.init_timestep = float(timestep)
        self.init_angJoi = angJoi.astype('float')

        self.reset_state()

        self.MR = float(0.01)
        self.typeMuscle = int(self.angJoi.size)
        self.levelArm = np.zeros(self.typeMuscle).astype('float')

        tmpL = np.zeros(self.typeMuscle)
        for i in range(0, self.typeMuscle):
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
        self.lse = float(self.lmtc - self.lce)
        # unitless parameters
        self.Lse = float(self.lse / self.lslack)
        self.Lce = float(self.lce / self.lopt)

        self.actsubstep = float((self.stim - self.act) * self.timestep / 2.0 / self.tau + self.act)
        self.lcesubstep = float(self.vce * self.timestep / 2.0 + self.lce)

        # test
        self.lce_avg = float(self.lce)
        self.vce_avg = float(self.vce)
        self.frcmtc_avg = float(0)
        self.act_avg = float(self.act)
        self.frame = 0

    def stepUpdateState(self, angJoi):
        """
        Muscle Tendon Complex Dynamics
        update muscle states based on the muscle dynamics
        Muscle state stim has to be updated outside before this function is called
        """
        self.frcmax, self.vmax, self.eref, self.lslack, self.lopt, self.tau, \
        self.w, self.c, self.N, self.K, self.stim, self.act, self.lmtc, self.lce, \
        self.vce, self.frcmtc, \
        self.r, self.phiref, \
        self.phimaxref, self.rho, \
        self.dirAng, self.phiScale, \
        self.angJoi, self.levelArm, self.offsetCorr, \
        self.timestep, self.MR, self.typeMuscle, \
        self.lse, self.Lse, self.Lce, self.actsubstep, \
        self.lcesubstep, self.lce_avg, self.vce_avg, self.frcmtc_avg, self.act_avg, self.frame = \
            step_update_state(
                self.frcmax, self.vmax, self.eref, self.lslack, self.lopt, self.tau,
                self.w, self.c, self.N, self.K, self.stim, self.act, self.lmtc, self.lce,
                self.vce, self.frcmtc,
                self.r, self.phiref,
                self.phimaxref, self.rho,
                self.dirAng, self.phiScale,
                angJoi, self.levelArm, self.offsetCorr,
                self.timestep, self.MR, self.typeMuscle,
                self.lse, self.Lse, self.Lce, self.actsubstep,
                self.lcesubstep, self.lce_avg, self.vce_avg, self.frcmtc_avg, self.act_avg, self.frame)

    def reset_state(self):
        self.frame = int(0)
        self.lce_avg = float(0)
        self.frcmtc_avg = float(0)
        self.act_avg = float(0)
        self.vce_avg = float(0)

        self.nameMuscle = self.init_nameMuscle
        self.frcmax = self.init_frcmax
        self.vmax = self.init_vmax
        self.eref = self.init_eref
        self.lslack = self.init_lslack
        self.lopt = self.init_lopt
        self.tau = self.init_tau
        self.w = self.init_w
        self.c = self.init_c
        self.N = self.init_N
        self.K = self.init_K
        self.stim = self.init_stim
        self.act = self.init_act
        self.lmtc = self.init_lmtc
        self.lce = self.init_lce
        self.vce = self.init_vce
        self.frcmtc = self.init_frcmtc
        self.r = self.init_r
        self.phiref = self.init_phiref
        self.phimaxref = self.init_phimaxref
        self.rho = self.init_rho
        self.dirAng = self.init_dirAng
        self.phiScale = self.init_phiScale
        self.offsetCorr = self.init_offsetCorr
        self.timestep = self.init_timestep
        self.angJoi = self.init_angJoi


class TIA(MuscleTendonComplex):
    """
     Tibialis Anterior (TIA): The Tibialis anterior (Tibialis anticus) is situated on the lateral
        side of the tibia. In real human it serves multiple function which are, Dorsal Flexion
        of the ankle, Inversion of the foot, Adduction of the foot and also Contributing in
        maintaining the medial arch of the foot. Here TIA is modelled as muscle actuating the
        ankle dorsiflexion in the sagittal plane.
    """

    def __init__(self, angAnk, timestep):
        frcmax = 800  # maximum isometric force [N]
        lopt = 0.06  # optimum fiber length CE [m]
        vmax = 12.0  # maximum contraction velocity [lopt/s]
        lslack = 0.24  # tendon slack length [m]
        # Tibialis Anterior attachment
        rTIAmax = 0.04  # [m]   maximum lever contribution
        rTIAmin = 0.01  # [m]   minimum lever contribution
        phimaxTIA = 80 * np.pi / 180  # [rad] angle of maximum lever contribution
        phiminTIA = 180 * np.pi / 180  # [rad] angle of minimum lever contribution
        phirefTIA = 110 * np.pi / 180  # [rad] reference angle at which MTU length equals
        phiScaleTIA = np.arccos(rTIAmin / rTIAmax) / (phiminTIA - phimaxTIA)
        rhoTIA = 0.7

        r = np.array((rTIAmax,))
        phiref = np.array((phirefTIA,))
        phimaxref = np.array((phimaxTIA,))
        rho = np.array((rhoTIA,))
        dirAng = np.array((1.0,))
        offsetCorr = np.array((1,))
        phiScale = np.array((phiScaleTIA,))

        lce = lopt

        angJoi = np.array((angAnk,))
        nameMuscle = "TIA"

        super().__init__(nameMuscle, frcmax, vmax, lslack, lopt,
                         lce, r, phiref, phimaxref, rho, dirAng, phiScale,
                         offsetCorr, timestep, angJoi)


class SOL(MuscleTendonComplex):
    """
     Soleus (SOL): Soleus muscles is Located in superficial posterior compartment of the
        leg, along with GAS it helps in the plantarflexion of the ankle joint. Here SOL is
        modelled as a muscle actuating the ankle plantarflexion in the sagittal plane.

    """

    def __init__(self, angAnk, timestep):
        frcmax = 4000  # maximum isometric force [N]
        lopt = 0.04  # optimum fiber length CE [m]
        vmax = 6.0  # maximum contraction velocity [lopt/s]
        lslack = 0.26  # tendon slack length [m]
        # SOLeus attachment
        rSOLmax = 0.06  # [m]   maximum lever contribution
        rSOLmin = 0.02  # [m]   minimum lever contribution
        phimaxSOL = 100 * np.pi / 180  # [rad] angle of maximum lever contribution
        phiminSOL = 180 * np.pi / 180  # [rad] angle of minimum lever contribution
        phirefSOL = 90 * np.pi / 180  # [rad] reference angle at which MTU length equals
        rhoSOL = 0.5  # sum of lopt and lslack
        phiScaleSOL = np.arccos(rSOLmin / rSOLmax) / (phiminSOL - phimaxSOL)

        r = np.array((rSOLmax,))
        phiref = np.array((phirefSOL,))
        phimaxref = np.array((phimaxSOL,))
        rho = np.array((rhoSOL,))
        dirAng = np.array((-1.0,))
        offsetCorr = np.array((1,))
        phiScale = np.array((phiScaleSOL,))

        lce = lopt

        angJoi = np.array((angAnk,))
        nameMuscle = "SOL"

        super().__init__(nameMuscle, frcmax, vmax, lslack, lopt,
                         lce, r, phiref, phimaxref, rho, dirAng, phiScale,
                         offsetCorr, timestep, angJoi)


class GAS(MuscleTendonComplex):
    """
     Gastrocnemius (GAS): Gastrocnemius muscle which the major bulk at the back of
        lower leg is a bi-articular muscle having two heads and runs from back of knee to the
        heel. The gastrocnemius helps plantarflexion of the ankle joint and flexion of the knee
        joint. Here GAS is modelled as a bi-articular MTU contributing to the knee flexion
        and ankle plantarflexion actuations in the sagittal plane.
    """

    def __init__(self, angKne, angAnk, timestep):
        frcmax = 1500  # maximum isometric force [N]
        lopt = 0.05  # optimum fiber length CE [m]
        vmax = 12.0  # maximum contraction velocity [lopt/s]
        lslack = 0.40  # tendon slack length [m]
        # GAStrocnemius attachment (knee joint)
        rGASkmax = 0.05  # [m]   maximum lever contribution
        rGASkmin = 0.02  # [m]   minimum lever contribution
        phimaxGASk = 140 * np.pi / 180  # [rad] angle of maximum lever contribution
        phiminGASk = 45 * np.pi / 180  # [rad] angle of minimum lever contribution
        phirefGASk = 165 * np.pi / 180  # [rad] reference angle at which MTU length equals
        rhoGASk = 0.7  # sum of lopt and lslack
        # rhoGASk     = 0.045           #       sum of lopt and lslack
        phiScaleGASk = np.arccos(rGASkmin / rGASkmax) / (phiminGASk - phimaxGASk)
        # GAStrocnemius attachment (ankle joint)
        rGASamax = 0.06  # [m]   maximum lever contribution
        rGASamin = 0.02  # [m]   minimum lever contribution
        phimaxGASa = 100 * np.pi / 180  # [rad] angle of maximum lever contribution
        phiminGASa = 180 * np.pi / 180  # [rad] angle of minimum lever contribution
        phirefGASa = 80 * np.pi / 180  # [rad] reference angle at which MTU length equals
        rhoGASa = 0.7  # sum of lopt and lslack
        # rhoGASa     =        0.045    #       sum of lopt and lslack
        phiScaleGASa = np.arccos(rGASamin / rGASamax) / (phiminGASa - phimaxGASa)

        r = np.array((rGASkmax, rGASamax))
        phiref = np.array((phirefGASk, phirefGASa))
        phimaxref = np.array((phimaxGASk, phimaxGASa))
        rho = np.array((rhoGASk, rhoGASa))
        dirAng = np.array((1.0, -1.0))
        offsetCorr = np.array((1, 1))
        phiScale = np.array((phiScaleGASk, phiScaleGASa))

        lce = lopt

        nameMuscle = "GAS"
        angJoi = np.array((angKne, angAnk))

        super().__init__(nameMuscle, frcmax, vmax, lslack, lopt,
                         lce, r, phiref, phimaxref, rho, dirAng, phiScale,
                         offsetCorr, timestep, angJoi)


class BFSH(MuscleTendonComplex):
    """
    Biceps Femoris Short Head(BFSH): This is a part of hamstring muscle in the real hu-
        man and is responsible for knee flexion. Here BFSH is modelled as muscle contributing
        to the knee flexion actuation in the sagittal plane.
    """

    def __init__(self, angKne, timestep):
        frcmax = 350  # maximum isometric force [N]
        lopt = 0.12  # optimum fiber length CE [m]
        vmax = 12.0  # 6 # maximum contraction velocity [lopt/s]
        lslack = 0.10  # tendon slack length [m]

        # BFSH group attachment
        rBFSH = 0.04  # [m]   constant lever contribution
        phirefBFSH = 160 * np.pi / 180  # [rad] reference angle at which MTU length equals
        rhoBFSH = 0.7  # sum of lopt and lslack

        r = np.array((rBFSH,))
        phiref = np.array((phirefBFSH,))
        phimaxref = np.array((0.0,))
        rho = np.array((rhoBFSH,))
        dirAng = np.array((1.0,))
        offsetCorr = np.array((0,))
        phiScale = np.array((0.0,))

        lce = lopt

        nameMuscle = "BFSH",
        angJoi = np.array((angKne,))

        super().__init__(nameMuscle, frcmax, vmax, lslack, lopt,
                         lce, r, phiref, phimaxref, rho, dirAng, phiScale,
                         offsetCorr, timestep, angJoi)


class VAS(MuscleTendonComplex):
    """
     Vasti (VAS): Vasti is a group of 3 muscles located in the thigh and is responsible for
        knee extension. Here VAS is modelled as a muscle actuating the knee extension in the
        sagittal plane.
    """

    def __init__(self, angKne, timestep):
        frcmax = 6000  # maximum isometric force [N]
        lopt = 0.08  # optimum fiber length CE [m]
        vmax = 12.0  # maximum contraction velocity [lopt/s]
        lslack = 0.23  # tendon slack length [m]
        # VAS group attachment
        rVASmax = 0.06  # [m]   maximum lever contribution
        rVASmin = 0.04  # [m]   minimum lever contribution
        phimaxVAS = 165 * np.pi / 180  # [rad] angle of maximum lever contribution
        phiminVAS = 45 * np.pi / 180  # [rad] angle of minimum lever contribution
        phirefVAS = 120 * np.pi / 180  # [rad] reference angle at which MTU length equals
        rhoVAS = 0.6  # sum of lopt and lslack
        phiScaleVAS = np.arccos(rVASmin / rVASmax) / (phiminVAS - phimaxVAS)

        r = np.array((rVASmax,))
        phiref = np.array((phirefVAS,))
        phimaxref = np.array((phimaxVAS,))
        rho = np.array((rhoVAS,))
        dirAng = np.array((-1.0,))
        offsetCorr = np.array((1,))
        phiScale = np.array((phiScaleVAS,))

        lce = lopt

        nameMuscle = "VAS"
        angJoi = np.array((angKne,))

        super().__init__(nameMuscle, frcmax, vmax, lslack, lopt,
                         lce, r, phiref, phimaxref, rho, dirAng, phiScale,
                         offsetCorr, timestep, angJoi)


class REF(MuscleTendonComplex):
    """
     Rectus Femoris (RF): The Rectus femoris muscle is one of the four quadriceps mus-
        cles. It is located in the middle of the front of the thigh and is responsible for knee
        extension and hip flexion. Here RF is modelled as a bi-articular MTU contributing to
        the hip flexion and knee extension actuations in the sagittal plane.
    """

    def __init__(self, angHip, angKne, timestep):
        frcmax = 1200  # maximum isometric force [N]
        lopt = 0.08  # optimum fiber length CE [m]
        vmax = 12.0  # maximum contraction velocity [lopt/s]
        lslack = 0.35  # tendon slack length [m]
        # REF group attachement (hip)
        rREFh = 0.08  # [m]   constant lever contribution
        phirefREFh = 170 * np.pi / 180  # [rad] reference angle at which MTU length equals
        rhoREFh = 0.3  # sum of lopt and lslack
        # REF group attachment (knee)
        rREFkmax = 0.06  # [m]   maximum lever contribution
        rREFkmin = 0.04  # [m]   minimum lever contribution
        phimaxREFk = 165 * np.pi / 180  # [rad] angle of maximum lever contribution
        phiminREFk = 45 * np.pi / 180  # [rad] angle of minimum lever contribution
        phirefREFk = 125 * np.pi / 180  # [rad] reference angle at which MTU length equals
        rhoREFk = 0.5  # sum of lopt and lslack
        phiScaleREFk = np.arccos(rREFkmin / rREFkmax) / (phiminREFk - phimaxREFk)

        r = np.array((rREFh, rREFkmax))
        phiref = np.array((phirefREFh, phirefREFk))
        phimaxref = np.array((0.0, phimaxREFk))
        rho = np.array((rhoREFh, rhoREFk))
        dirAng = np.array((1.0, -1.0))
        offsetCorr = np.array((0, 1))
        phiScale = np.array((0.0, phiScaleREFk))

        lce = lopt

        nameMuscle = "REF"
        angJoi = np.array((angHip, angKne))

        super().__init__(nameMuscle, frcmax, vmax, lslack, lopt,
                         lce, r, phiref, phimaxref, rho, dirAng, phiScale,
                         offsetCorr, timestep, angJoi)


class HAM(MuscleTendonComplex):
    """
    Hamstrings (HAM): The hamstring muscles are a group of four muscles located in the
        back of the thigh. They are bi-articular muscles crossing the hip and knee joints, so
        they can help in both knee flexion and hip extension at the same time. Here HAM
        is modelled as a bi-articular MTU contributing to the hip extension and knee flexion
        actuations in the sagittal plane.
    """

    def __init__(self, angHip, angKne, timestep):
        frcmax = 3000  # maximum isometric force [N]
        lopt = 0.10  # optimum fiber length CE [m]
        vmax = 12.0  # maximum contraction velocity [lopt/s]
        lslack = 0.31  # tendon slack length [m]
        # hamstring hip level arm and refernce angle
        rHAMh = 0.08  # [m]   constant lever contribution
        phirefHAMh = 150 * np.pi / 180  # [rad] reference angle at which MTU length equals
        rhoHAMh = 0.5  # sum of lopt and lslack
        # hamstring knee level arm and reference angle
        rHAMk = 0.05  # [m]   constant lever contribution
        phirefHAMk = 180 * np.pi / 180  # [rad] reference angle at which MTU length equals
        rhoHAMk = 0.5  # sum of lopt and lslack

        r = np.array((rHAMh, rHAMk))
        phiref = np.array((phirefHAMh, phirefHAMk))
        phimaxref = np.array((0.0, 0.0))
        rho = np.array((rhoHAMh, rhoHAMk))
        dirAng = np.array((-1.0, 1.0))
        offsetCorr = np.array((0, 0))
        phiScale = np.array((0.0, 0.0))

        lce = lopt

        nameMuscle = "HAM"
        angJoi = np.array((angHip, angKne))

        super().__init__(nameMuscle, frcmax, vmax, lslack, lopt,
                         lce, r, phiref, phimaxref, rho, dirAng, phiScale,
                         offsetCorr, timestep, angJoi)


class HFL(MuscleTendonComplex):
    """
    Hip Flexor (HFL): The hip flexors are a group of muscles that help to bring the legs
        and trunk together in a flexion movement. HFL allow us to move our leg or knee up
        towards your torso, as well as to bend your torso forward at the hip. The HLF modelled
        here is one of the actuator for the hip flexion in the sagittal plane.
    """

    def __init__(self, angHip, timestep):
        frcmax = 2000  # maximum isometric force [N]
        lopt = 0.11  # optimum fiber length CE [m]
        vmax = 12.0  # maximum contraction velocity [lopt/s]
        lslack = 0.10  # tendon slack length [m]
        # level arm and reference angle
        r = np.array((0.08,))  # [m]   constant lever contribution
        phiref = np.array((160 * np.pi / 180,))  # [rad] reference angle at which MTU length equals
        phimaxref = np.array((0.0, 0.0))
        rho = np.array((0.5,))  # sum of lopt and lslack
        dirAng = np.array((1.0,))  # angle increase leads to MTC length increase
        offsetCorr = np.array((0,))  # no level arm correction
        phiScale = np.array((0.0,))

        lce = lopt

        angJoi = np.array((angHip,))
        nameMuscle = "HFL"

        super().__init__(nameMuscle, frcmax, vmax, lslack, lopt,
                         lce, r, phiref, phimaxref, rho, dirAng, phiScale,
                         offsetCorr, timestep, angJoi)


class GLU(MuscleTendonComplex):
    """
     Glutei (GLU): The glutei muscles are a group muscles in the gluteal region, in real life
        locomotion their functions include extension, abduction, external rotation and internal
        rotation of the hip joint. But here in the model GLU is modelled antagonistic to HFL
        as hip extensor, acting as one of the hip joint actuator in the sagittal plane.
    """

    def __init__(self, angHip, timestep):
        frcmax = 1500.0  # maximum isometric force [N]
        lopt = 0.11  # optimum fiber length CE [m]
        vmax = 12.0  # maximum contraction velocity [lopt/s]
        lslack = 0.13  # tendon slack length [m]
        # level arm and reference angle
        r = np.array((0.08,))  # [m]   constant lever contribution
        phiref = np.array((120 * np.pi / 180,))  # [rad] reference angle at which MTU length equals
        phimaxref = np.array((0.0, 0.0))
        rho = np.array((0.5,))  # sum of lopt and lslack
        dirAng = np.array((-1.0,))  # angle increase leads to MTC length decrease
        offsetCorr = np.array((0,))  # no level arm correction
        phiScale = np.array((0.0,))

        lce = lopt  # will be computed in the initialization

        nameMuscle = "GLU"
        angJoi = np.array((angHip,))

        super().__init__(nameMuscle, frcmax, vmax, lslack, lopt,
                         lce, r, phiref, phimaxref, rho, dirAng, phiScale,
                         offsetCorr, timestep, angJoi)


class HAD(MuscleTendonComplex):
    """
     Hip Adductor (HAD): Hip adductors in the thigh are a group of muscles near the groin
        area which helps in moving the leg towards the midline of the body in the coronal
        plane. They are basically the are antagonistic to the hip abductors and also help in
        stabilizing the hip joint in real life locomotion. The HAD modelled here will act as the
        second actuator for the hip adduction in the coronal plane.
    """

    def __init__(self, angHipFront, timestep):
        frcmax = 4500.0  # maximum isometric force [N]
        lopt = 0.10  # optimum fiber length CE [m]
        vmax = 12.0  # maximum contraction velocity [lopt/s]
        lslack = 0.18  # tendon slack length [m]

        rHAD = 0.03  # [m]   constant lever contribution
        phirefHAD = 15 * np.pi / 180  # [rad] reference angle at which MTU length equals
        rhoHAD = 1.0  # sum of lopt and lslack

        r = np.array((rHAD,))
        phiref = np.array((phirefHAD,))
        phimaxref = np.array((0.0,))
        rho = np.array((rhoHAD,))
        dirAng = np.array((1.0,))
        offsetCorr = np.array((0,))
        phiScale = np.array((0.0,))

        lce = lopt

        nameMuscle = "HAD"
        angJoi = np.array((angHipFront,))

        super().__init__(nameMuscle, frcmax, vmax, lslack, lopt,
                         lce, r, phiref, phimaxref, rho, dirAng, phiScale,
                         offsetCorr, timestep, angJoi)


class HAB(MuscleTendonComplex):
    """
     Hip Abductor (HAB): The hip abductor muscles in the thigh include a group of muscles
        which helps in moving the leg away from the midline of the body in the coronal plane.
        They also help to rotate the thigh in the hip socket and to stabilize the hip joint. The
        HAB modelled here will act as an actuator for the hip adbuction in the coronal plane.
    """

    def __init__(self, angHipFront, timestep):
        frcmax = 3000.0  # maximum isometric force [N]
        lopt = 0.09  # optimum fiber length CE [m]
        vmax = 12.0  # maximum contraction velocity [lopt/s]
        lslack = 0.07  # tendon slack length [m]

        rHAB = 0.06  # [m]   constant lever contribution
        phirefHAB = 10 * np.pi / 180  # [rad] reference angle at which MTU length equals
        rhoHAB = 0.7  # sum of lopt and lslack

        r = np.array((rHAB,))
        phiref = np.array((phirefHAB,))
        phimaxref = np.array((0.0,))
        rho = np.array((rhoHAB,))
        dirAng = np.array((-1.0,))
        offsetCorr = np.array((0,))
        phiScale = np.array((0.0,))

        lce = lopt

        nameMuscle = "HAB"
        angJoi = np.array((angHipFront,))

        super().__init__(nameMuscle, frcmax, vmax, lslack, lopt,
                         lce, r, phiref, phimaxref, rho, dirAng, phiScale,
                         offsetCorr, timestep, angJoi)
