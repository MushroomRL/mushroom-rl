import numpy as np
from .mtcmodel import MuscleTendonComplex

# timestep = 1e-3  # 2e-4 -> original value

# general muscle parameters:

# Series elastic element (SE) force-length relationship
eref = 0.04  # [lslack] tendon reference strain

# excitation-contraction coupling
preAct = 0.01  # 0.01     # [] preactivation
tau = 0.01  # [s] delay time constant

# contractile element (CE) force-length relationship
w = 0.56  # [lopt] width
c = 0.05  # []; remaining force at +/- width

# CE force-velocity relationship
N = 1.5  # Fmax] eccentric force enhancement
K = 5.0  # [] shape factor
stim = 0.0  # initial stimulation
vce = 0.0
frcmtc = 0.0
l_mtc = 0.0


def HAB(angHipFront, timestep):
    """
    HAB, hip abductor
    :param angHipFront:
    :return:
    """
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

    act = preAct
    lmtc = l_mtc
    lce = lopt

    paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
    stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
    paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
    musHAB = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle,
                                 paraMusAttach=paraMusAttach,
                                 offsetCorr=offsetCorr, timestep=timestep, nameMuscle="HAB",
                                 angJoi=np.array((angHipFront,)))

    return musHAB


def HAD(angHipFront, timestep):
    """
    HAD, hip adductor
    :param angHipFront:
    :return:
    """
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

    act = preAct
    lmtc = l_mtc
    lce = lopt

    paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
    stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
    paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
    musHAD = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle,
                                 paraMusAttach=paraMusAttach,
                                 offsetCorr=offsetCorr, timestep=timestep, nameMuscle="HAD",
                                 angJoi=np.array((angHipFront,)))

    return musHAD


def GLU(angHip, timestep):
    """
    GLU, gluteus maximus
    :param angHip: angle between trunk and thigh, rad, it is 180 deg in standing
    :return:
    """
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
    # typeMuscle = 1  # monoarticular
    phiScale = np.array((0.0,))

    act = preAct
    lmtc = l_mtc  # will be computed in the initialization
    lce = lopt  # will be computed in the initialization

    paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
    stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
    paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
    musGLU = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle,
                                 paraMusAttach=paraMusAttach,
                                 offsetCorr=offsetCorr, timestep=timestep, nameMuscle="GLU",
                                 angJoi=np.array((angHip,)))

    return musGLU


def HFL(angHip, timestep):
    """
    HFL, hip flexor
    :param angHip:
    :return:
    """
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
    # typeMuscle = 1  # monoarticular
    phiScale = np.array((0.0,))

    # act = preAct
    act = 0.0
    lmtc = l_mtc  # should be computed based on the joint angle and joint geometry
    lce = lopt

    paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
    stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
    paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
    musHFL = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle,
                                 paraMusAttach=paraMusAttach,
                                 offsetCorr=offsetCorr, timestep=timestep, nameMuscle="HFL",
                                 angJoi=np.array((angHip,)))

    return musHFL


def HAM(angHip, angKne, timestep):
    """
    HAM, hamstring
    :param angHip:
    :param angKne:
    :return:
    """
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
    # typeMuscle = 2
    phiScale = np.array((0.0, 0.0))

    act = preAct
    lmtc = l_mtc
    lce = lopt

    paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
    stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
    paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
    musHAM = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle,
                                 paraMusAttach=paraMusAttach,
                                 offsetCorr=offsetCorr, timestep=timestep, nameMuscle="HAM",
                                 angJoi=np.array((angHip, angKne)))

    return musHAM


def REF(angHip, angKne, timestep):
    """
    REF, rectus femoris
    :param angKne:
    :param angHip:
    :return:
    """
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

    act = preAct
    lmtc = l_mtc
    lce = lopt

    paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
    stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
    paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
    musREF = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle,
                                 paraMusAttach=paraMusAttach,
                                 offsetCorr=offsetCorr, timestep=timestep, nameMuscle="REF",
                                 angJoi=np.array((angHip, angKne)))

    return musREF


def VAS(angKne, timestep):
    """
    VAS, vastus muscle
    :param angKne:
    :return:
    """
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

    act = preAct
    lmtc = l_mtc
    lce = lopt

    paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
    stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
    paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
    musVAS = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle,
                                 paraMusAttach=paraMusAttach,
                                 offsetCorr=offsetCorr, timestep=timestep, nameMuscle="VAS",
                                 angJoi=np.array((angKne,)))

    return musVAS


def BFSH(angKne, timestep):
    """
    BFSH, biceps femoris short head muscle
    :param angKne:
    :return:
    """
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

    act = preAct
    lmtc = l_mtc
    lce = lopt

    paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
    stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
    paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
    musBFSH = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle,
                                  paraMusAttach=paraMusAttach,
                                  offsetCorr=offsetCorr, timestep=timestep, nameMuscle="BFSH",
                                  angJoi=np.array((angKne,)))

    return musBFSH


def GAS(angKne, angAnk, timestep):
    """
    GAS, gastronemius msucle
    :param angKne:
    :param angAnk:
    :return:
    """
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

    act = preAct
    lmtc = l_mtc
    lce = lopt

    paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
    stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
    paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
    musGAS = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle,
                                 paraMusAttach=paraMusAttach,
                                 offsetCorr=offsetCorr, timestep=timestep, nameMuscle="GAS",
                                 angJoi=np.array((angKne, angAnk)))

    return musGAS


def SOL(angAnk, timestep):
    """
    SOL, soleus muscle
    :param angAnk:
    :return:
    """
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

    act = preAct
    lmtc = l_mtc
    lce = lopt

    paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
    stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
    paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
    musSOL = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle,
                                 paraMusAttach=paraMusAttach,
                                 offsetCorr=offsetCorr, timestep=timestep, nameMuscle="SOL",
                                 angJoi=np.array((angAnk,)))

    return musSOL


def TIA(angAnk, timestep):
    """
    TIA, tibialis anterior muscle
    :param angAnk:
    :return:
    """
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

    act = preAct
    lmtc = l_mtc
    lce = lopt

    paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
    stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
    paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
    musTIA = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle,
                                 paraMusAttach=paraMusAttach,
                                 offsetCorr=offsetCorr, timestep=timestep, nameMuscle="TIA",
                                 angJoi=np.array((angAnk,)))

    return musTIA
