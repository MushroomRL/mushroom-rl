import numpy as np
from copy import deepcopy

def value_iteration(P, R, gamma, eps):
    stateN = P.shape[0]
    actionN = P.shape[1]

    V = np.zeros(stateN)

    while True:
        Vold = deepcopy(V)

        for s in xrange(stateN):
            vmax = -float('inf')
            for a in xrange(actionN):
                Psa = P[s, a, :]
                Rsa = R[s, a, :]
                va = Psa.T.dot(Rsa + gamma * Vold)
                vmax = max(va, vmax)

            V[s] = vmax
        if np.linalg.norm(V - Vold) <= eps:
            break

    return V


def policy_iteration(P, R, gamma):
    stateN = P.shape[0]
    actionN = P.shape[1]

    pi = np.zeros(stateN,dtype=int)
    V = np.zeros(stateN)

    changed = True
    while(changed):
        #Compute value function
        Ppi = np.zeros((stateN, stateN))
        Rpi = np.zeros(stateN)
        I= np.eye(stateN)

        for s in xrange(stateN):
            a = pi[s]
            PpiS = P[s, a, :]
            RpiS = R[s, a, :]

            Ppi[s, :] = PpiS.T
            Rpi[s] = PpiS.T.dot(RpiS)

        V = np.linalg.inv(I - gamma * Ppi).dot(Rpi)

        #Compute policy
        changed = False

        for s in xrange(stateN):
            vmax = V[s]
            for a in xrange(actionN):
                if a != pi[s]:
                    Psa = P[s, a]
                    Rsa = R[s, a]
                    va = Psa.T.dot(Rsa + gamma * V)
                    if (va > vmax):
                        pi[s] = a
                        vmax = va
                        changed = True

    return V, pi