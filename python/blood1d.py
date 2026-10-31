"""One-dimensional blood flow in a single elastic vessel.

Python translation of Saray Busto's MATLAB code in ``blood_flow/`` (``model.m``, ``vessel.m``, ``Main.m``).
Conserved variables Q = (A, q) with q = A u. Tube law p = K (sqrt(A / a0) - 1), i.e. m = 1/2, and no source
term, as in the MATLAB code. Explicit finite volume scheme with transparent boundary conditions and the exact
solution of the Riemann problem for validation.
"""
import numpy as np

MMHG = 133.322


class Vessel:
    """Vessel parameters and mesh (``vessel.m`` plus the parameter block of ``Main.m``)."""

    def __init__(self, ncells=200, L=0.2, E=4.0e5, h=1.1e-3, r0=9.99e-3, rho=1050.0, m=0.5):
        self.ncells = ncells
        self.L = L
        self.rho = rho
        self.a0 = np.pi * r0**2
        self.K = np.sqrt(np.pi) / (1 - 0.5**2) * E * h / np.sqrt(self.a0)
        self.gamma = m * self.K / rho / (m + 1) / self.a0**m
        self.dx = L / ncells
        self.xC = (np.arange(ncells) + 0.5) * self.dx

    # Physics
    def flux(self, A, q):
        return q, q**2 / A + self.gamma * A**1.5

    def wave_speed(self, A):
        return np.sqrt(self.gamma * 1.5 * np.sqrt(A))

    def pressure(self, A):
        return self.K * (np.sqrt(A / self.a0) - 1.0)

    def time_step(self, A, q, CFL):
        lam = np.abs(q / A) + self.wave_speed(A)
        return CFL * self.dx / lam.max()


# Exact Riemann solver (solveERP, sampleERP, fK, mK, rarL, rarR in model.m)

def _fK(v, aK, aS, cK):
    cS = v.wave_speed(aS)
    rare = 4.0 * (cS - cK)
    with np.errstate(invalid="ignore", divide="ignore"):
        shock = np.sqrt(v.gamma * (aS - aK) * (aS**1.5 - aK**1.5) / (aK * aS))
    return np.where(aS <= aK, rare, shock)


def _mK(v, aK, aS):
    # Shock mass flux; its limit as aS -> aK is aK cK, which avoids 0/0.
    d = aS - aK
    safe = np.abs(d) > 1e-14 * aK
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.sqrt(v.gamma * aS * aK * (aS**1.5 - aK**1.5) / np.where(safe, d, 1.0))
    return np.where(safe, m, aK * v.wave_speed(aK))


def solve_erp(v, aL, aR, uL, uR, tol=1e-6, eps=1e-12, maxit=100):
    """Star state (aS, uS) by Newton's method with a centred numerical derivative, as in solveERP."""
    aL, aR, uL, uR = map(np.asarray, (aL, aR, uL, uR))
    cL, cR = v.wave_speed(aL), v.wave_speed(aR)
    # Initial guess from the two-rarefaction approximation when it is positive, else the mean area as in
    # solveERP. Near-vacuum states (the collapse test) need it: the mean area makes Newton overshoot below 0.
    cS0 = 0.5 * (cL + cR) - 0.125 * (uR - uL)
    A = np.where(cS0 > 0, (np.maximum(cS0, 0.0)**2 / (1.5 * v.gamma))**2, 0.5 * (aL + aR))
    A = np.where(A > 0, A, 0.5 * (aL + aR))
    for _ in range(maxit):
        g = _fK(v, aL, A, cL) + _fK(v, aR, A, cR) + uR - uL
        gp = _fK(v, aL, A + eps, cL) + _fK(v, aR, A + eps, cR)
        gm = _fK(v, aL, A - eps, cL) + _fK(v, aR, A - eps, cR)
        Aold = A
        A = A - g / ((gp - gm) / (2 * eps))
        # Keep the area positive (MATLAB would silently go complex here).
        A = np.where(A > 0, A, 0.1 * Aold)
        if np.all(np.abs(A - Aold) / (0.5 * (A + Aold)) <= tol):
            break
    uS = 0.5 * (uL + uR) + 0.5 * (_fK(v, aR, A, cR) - _fK(v, aL, A, cL))
    return A, uS


def sample_erp(v, aL, aR, uL, uR, aS, uS, xt):
    """Exact solution at x/t = xt (sampleERP). Returns (A, u)."""
    aL, aR, uL, uR, aS, uS, xt = np.broadcast_arrays(*map(np.asarray, (aL, aR, uL, uR, aS, uS, xt)))
    cS, cL, cR = v.wave_speed(aS), v.wave_speed(aL), v.wave_speed(aR)
    sL = uL - _mK(v, aL, aS) / aL
    sR = uR + _mK(v, aR, aS) / aR
    a = np.empty_like(xt, dtype=float)
    u = np.empty_like(xt, dtype=float)

    # Rarefaction fans (rarL, rarR)
    def rar_left(x):
        cr = (uL + 4 * cL - x) / 5
        return cr**4 * 4 / 9 / v.gamma**2, (uL + 4 * cL + 4 * x) / 5

    def rar_right(x):
        cr = (-uR + 4 * cR + x) / 5
        return cr**4 * 4 / 9 / v.gamma**2, (uR - 4 * cR + 4 * x) / 5

    left = xt < uS
    # Left wave
    lr = left & (aS <= aL)
    ls = left & (aS > aL)
    m = lr & (xt < uL - cL); a[m], u[m] = aL[m], uL[m]
    m = lr & (xt >= uL - cL) & (xt < uS - cS); ar, ur = rar_left(xt); a[m], u[m] = ar[m], ur[m]
    m = lr & (xt >= uS - cS); a[m], u[m] = aS[m], uS[m]
    m = ls & (xt < sL); a[m], u[m] = aL[m], uL[m]
    m = ls & (xt >= sL); a[m], u[m] = aS[m], uS[m]
    # Right wave
    rr = ~left & (aS <= aR)
    rs = ~left & (aS > aR)
    m = rr & (xt < uS + cS); a[m], u[m] = aS[m], uS[m]
    m = rr & (xt >= uS + cS) & (xt < uR + cR); ar, ur = rar_right(xt); a[m], u[m] = ar[m], ur[m]
    m = rr & (xt >= uR + cR); a[m], u[m] = aR[m], uR[m]
    m = rs & (xt < sR); a[m], u[m] = aS[m], uS[m]
    m = rs & (xt >= sR); a[m], u[m] = aR[m], uR[m]
    return a, u


def exact_solution(v, aL, aR, uL, uR, gate, t, x):
    aS, uS = solve_erp(v, aL, aR, uL, uR)
    return sample_erp(v, aL, aR, uL, uR, aS, uS, (x - gate) / (t + 1e-10))


# Numerical fluxes (model.m). Each takes the left and right states at every interface.

def flux_godunov_exact(v, AL, qL, AR, qR, dt, dx):
    uL, uR = qL / AL, qR / AR
    aS, uS = solve_erp(v, AL, AR, uL, uR)
    a, u = sample_erp(v, AL, AR, uL, uR, aS, uS, np.zeros_like(AL))
    return v.flux(a, a * u)


def flux_lax_friedrichs(v, AL, qL, AR, qR, dt, dx):
    FL, FR = v.flux(AL, qL), v.flux(AR, qR)
    return tuple(0.5 * (fl + fr) - 0.5 * dx / dt * (r - l) for fl, fr, l, r in zip(FL, FR, (AL, qL), (AR, qR)))


def flux_lax_wendroff(v, AL, qL, AR, qR, dt, dx):
    FL, FR = v.flux(AL, qL), v.flux(AR, qR)
    A = 0.5 * (AL + AR) - 0.5 * dt / dx * (FR[0] - FL[0])
    q = 0.5 * (qL + qR) - 0.5 * dt / dx * (FR[1] - FL[1])
    return v.flux(A, q)


def flux_force(v, AL, qL, AR, qR, dt, dx):
    lf = flux_lax_friedrichs(v, AL, qL, AR, qR, dt, dx)
    lw = flux_lax_wendroff(v, AL, qL, AR, qR, dt, dx)
    return tuple(0.5 * (a + b) for a, b in zip(lf, lw))


FLUXES = {
    "godunov": flux_godunov_exact,
    "lax-friedrichs": flux_lax_friedrichs,
    "lax-wendroff": flux_lax_wendroff,
    "force": flux_force,
}


# Riemann problems of Main.m and Mainrcolapso.m: (aL/a0, aR/a0, uL, uR, tEnd, gate, vessel parameters)
PROBLEMS = {
    1: dict(name="rarefacción-choque", aL=2.2, aR=1.1, uL=0.0, uR=0.0, tEnd=0.014, gate=0.1),
    2: dict(name="dúas rarefaccións", aL=1.6, aR=1.6, uL=-1.5, uR=1.5, tEnd=0.01, gate=0.1),
    3: dict(name="dous choques", aL=1.6, aR=1.6, uL=1.5, uR=-1.5, tEnd=0.014, gate=0.1),
    4: dict(name="choque", aL=1.6, aR=1.6, uL=-2.6575e-5, uR=6.1230e-6, tEnd=0.014, gate=0.1),
    5: dict(name="punto sónico", aL=10.0, aR=1.0, uL=0.0, uR=0.0, tEnd=0.007, gate=0.1),
    6: dict(name="colapso", aL=None, aR=None, uL=-12.5, uR=12.5, tEnd=0.005, gate=0.25,
            vessel=dict(L=0.5, E=3.0e5, h=5.0e-4, r0=1.0e-2, rho=1000.0)),
}


def make_problem(rp, ncells):
    P = PROBLEMS[rp]
    v = Vessel(ncells=ncells, **P.get("vessel", {}))
    if rp == 6:
        cte = 1.0e-4 / 1.0e-2**2
        aL = aR = cte * v.a0
    else:
        aL, aR = P["aL"] * v.a0, P["aR"] * v.a0
    return v, aL, aR, P["uL"], P["uR"], P["tEnd"], P["gate"]


def run(rp=1, ncells=200, CFL=0.9, flux="godunov", tEnd=None, nmax=100000):
    """Solve a Riemann problem as Main.m does. Returns the vessel, x, A, u, final time and number of steps."""
    v, aL, aR, uL, uR, tE, gate = make_problem(rp, ncells)
    tEnd = tE if tEnd is None else tEnd
    fnum = FLUXES[flux]
    A = np.where(v.xC <= gate, aL, aR).astype(float)
    q = np.where(v.xC <= gate, aL * uL, aR * uR).astype(float)
    t, n = 0.0, 0
    while t < tEnd and n < nmax:
        dt = v.time_step(A, q, CFL)
        if not np.isfinite(dt) or dt <= 0:
            break
        dt = min(dt, tEnd - t)
        # Transparent boundary conditions (BCL = BCR = 1)
        Ae = np.concatenate(([A[0]], A, [A[-1]]))
        qe = np.concatenate(([q[0]], q, [q[-1]]))
        with np.errstate(all="ignore"):
            F0, F1 = fnum(v, Ae[:-1], qe[:-1], Ae[1:], qe[1:], dt, v.dx)
            A = A - dt / v.dx * (F0[1:] - F0[:-1])
            q = q - dt / v.dx * (F1[1:] - F1[:-1])
        t += dt
        n += 1
        if not (np.all(np.isfinite(A)) and np.all(A > 0)):
            break
    with np.errstate(all="ignore"):
        u = q / A
    return v, v.xC, A, u, t, n
