cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport pow as cpow
from libc.math cimport fabs as cabs
from libc.math cimport log as clog
from libc.math cimport exp as cexp
from libc.math cimport sin as csin
from libc.math cimport cos as ccos


@cython.boundscheck(False)  # Deactivate python standard checks
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple step_update_state(float frcmax, float vmax, float eref, float lslack, float lopt, float tau,
                              float w, float c, float N, float K, float stim, float act, float lmtc, float lce,
                              float vce, float frcmtc,
                              np.ndarray[np.float_t, ndim=1] r, np.ndarray[np.float_t, ndim=1] phiref,
                              np.ndarray[np.float_t, ndim=1] phimaxref, np.ndarray[np.float_t, ndim=1] rho,
                              np.ndarray[np.float_t, ndim=1] dirAng, np.ndarray[np.float_t, ndim=1] phiScale,
                              np.ndarray[np.float_t, ndim=1] angJoi, np.ndarray[np.float_t, ndim=1] levelArm,
                              np.ndarray[np.int_t, ndim=1] offsetCorr,
                              float timestep, float MR, int typeMuscle,
                              float lse, float Lse, float Lce, float actsubstep,
                              float lcesubstep, float lce_avg, float vce_avg, float frcmtc_avg, float act_avg, int frame):

    """
    Muscle Tendon Complex Dynamics
    update muscle states based on the muscle dynamics
    Muscle state stim has to be updated outside before this function is called
    """
    cdef:
        float tmpL [2]
        float tmp
        int i
        float tmpL_sum
        float v_frac
        float mr_scale

    tmpL[:] = [.0,.0]

    for i in range(0, typeMuscle):
        if offsetCorr[i] == 0:
            tmpL[i] = dirAng[i] * (angJoi[i] - phiref[i]) * r[i] * rho[i]
            levelArm[i] = r[i]
        else:
            tmp1 = csin((phiref[i] - phimaxref[i]) * phiScale[i])
            tmp2 = csin((angJoi[i] - phimaxref[i]) * phiScale[i])
            tmpL[i] = dirAng[i] * (tmp2 - tmp1) * r[i] * rho[i] / phiScale[i]
            levelArm[i] = ccos((angJoi[i] - phimaxref[i]) * phiScale[i]) * r[i]

    tmpL_sum = .0
    for i in range(0, typeMuscle):
        tmpL_sum += tmpL[i]

    lmtc = lslack + lopt + tmpL_sum

    # update muscle activation
    # integration, forward-Euler method
    act = (stim - actsubstep) * timestep / 2.0 / tau + actsubstep
    actsubstep = (stim - act) * timestep / 2.0 / tau + act

    # update lce and lse based on the lmtc
    # integration, forward-Euler method
    lce = vce * timestep / 2.0 + lcesubstep
    lcesubstep = vce * timestep / 2.0 + lce

    lse = lmtc - lce
    Lse = lse / lslack
    Lce = lce / lopt

    # Serial Elastic element (tendon) force-length relationship
    if Lse > 1.0:
        Fse = cpow((Lse - 1.0) / eref, 2)
    else:
        Fse = 0.0

    # Parallel Elasticity PE
    if Lce > 1.0:
        Fpe = cpow((Lce - 1.0) / w, 2)
    else:
        Fpe = 0.0

    # update frcmtc
    frcmtc = Fse * frcmax

    # Buffer Elasticity BE
    if (Lce - (1.0 - w)) < 0:
        Fbe = cpow((Lce - (1.0 - w)) / (w / 2), 2)
    else:
        Fbe = 0.0

    # Contractile Element force-length relationship
    tmp = cpow(cabs(Lce - 1.0) / w, 3)
    Fce = cexp(tmp * clog(c))

    if (Fpe + Fce * act) < 1e-10:  # avoid numerical error
        if (Fse + Fbe) < 1e-10:
            Fv = 1.0
        else:
            Fv = (Fse + Fbe) / 1e-10
    else:
        Fv = (Fse + Fbe) / (Fpe + Fce * act)


    # Contractile Element inverse force-velocity relationship
    if Fv <= 1.0:
        # Concentric
        v = (Fv - 1) / (Fv * K + 1.0)
    elif Fv <= N:
        # excentric
        tmp = (Fv - N) / (N - 1.0)
        v = (tmp + 1.0) / (1.0 - tmp * 7.56 * K)
    else:
        # excentric overshoot
        v = ((Fv - N) * 0.01 + 1)

    vce = v * lopt * vmax
    v_frac = vce /  vmax
    mr_scale =  act * cabs(frcmax*vmax) *timestep
    if vce <= 1:
        MR =  0.01 - 0.11*(v_frac) + 0.06*cexp(-8*v_frac)
    else:
        MR =  0.23 - 0.16*cexp(-8*v_frac)
    MR *= mr_scale
    frame += 1
    lce_avg = (lce_avg*(frame - 1) +  lce) / frame
    vce_avg = (vce_avg*(frame - 1) +  vce) / frame
    frcmtc_avg = (frcmtc_avg*(frame - 1) +  frcmtc) / frame
    act_avg = (act_avg*(frame - 1) +  act) / frame

    return frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K, stim, act, lmtc, lce, vce, frcmtc, r, phiref,\
            phimaxref, rho, dirAng,  phiScale, angJoi,  levelArm, offsetCorr, timestep,  MR,  typeMuscle,\
            lse,  Lse,  Lce,  actsubstep, lcesubstep,  lce_avg,  vce_avg,  frcmtc_avg,  act_avg,  frame