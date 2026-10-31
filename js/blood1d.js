/* One-dimensional blood flow in a single elastic vessel.

   JavaScript port of python/blood1d.py, itself a translation of Saray Busto's MATLAB code in blood_flow/
   (model.m, vessel.m, Main.m). Conserved variables Q = (A, q) with q = A u. Tube law p = K (sqrt(A / a0) - 1),
   i.e. m = 1/2, and no source term. Explicit finite volume scheme with transparent boundary conditions and the
   exact solution of the Riemann problem for validation. Scalar loops instead of NumPy arrays. */
(function (root) {
  "use strict";

  // Vessel parameters and mesh (vessel.m plus the parameter block of Main.m)
  function Vessel(opts) {
    const o = Object.assign({ ncells: 200, L: 0.2, E: 4.0e5, h: 1.1e-3, r0: 9.99e-3, rho: 1050.0, m: 0.5 }, opts);
    this.ncells = o.ncells;
    this.L = o.L;
    this.rho = o.rho;
    this.a0 = Math.PI * o.r0 * o.r0;
    this.K = Math.sqrt(Math.PI) / (1 - 0.5 * 0.5) * o.E * o.h / Math.sqrt(this.a0);
    this.gamma = o.m * this.K / o.rho / (o.m + 1) / Math.pow(this.a0, o.m);
    this.dx = o.L / o.ncells;
    this.xC = new Float64Array(o.ncells);
    for (let i = 0; i < o.ncells; i++) this.xC[i] = (i + 0.5) * this.dx;
  }
  Vessel.prototype.waveSpeed = function (A) { return Math.sqrt(this.gamma * 1.5 * Math.sqrt(A)); };
  Vessel.prototype.pressure = function (A) { return this.K * (Math.sqrt(A / this.a0) - 1.0); };

  // Exact Riemann solver (solveERP, sampleERP, fK, mK, rarL, rarR in model.m)

  function fK(v, aK, aS, cK) {
    if (aS <= aK) return 4.0 * (v.waveSpeed(aS) - cK);
    return Math.sqrt(v.gamma * (aS - aK) * (Math.pow(aS, 1.5) - Math.pow(aK, 1.5)) / (aK * aS));
  }

  function mK(v, aK, aS) {
    // Shock mass flux; its limit as aS -> aK is aK cK, which avoids 0/0.
    const d = aS - aK;
    if (Math.abs(d) <= 1e-14 * aK) return aK * v.waveSpeed(aK);
    return Math.sqrt(v.gamma * aS * aK * (Math.pow(aS, 1.5) - Math.pow(aK, 1.5)) / d);
  }

  // Star state [aS, uS] by Newton's method with a centred numerical derivative, as in solveERP.
  function solveERP(v, aL, aR, uL, uR, tol, eps, maxit) {
    tol = tol || 1e-6; eps = eps || 1e-12; maxit = maxit || 100;
    const cL = v.waveSpeed(aL), cR = v.waveSpeed(aR);
    // Initial guess from the two-rarefaction approximation when it is positive, else the mean area.
    const cS0 = 0.5 * (cL + cR) - 0.125 * (uR - uL);
    let A = cS0 > 0 ? Math.pow(cS0 * cS0 / (1.5 * v.gamma), 2) : 0.5 * (aL + aR);
    if (!(A > 0)) A = 0.5 * (aL + aR);
    for (let k = 0; k < maxit; k++) {
      const g = fK(v, aL, A, cL) + fK(v, aR, A, cR) + uR - uL;
      const gp = fK(v, aL, A + eps, cL) + fK(v, aR, A + eps, cR);
      const gm = fK(v, aL, A - eps, cL) + fK(v, aR, A - eps, cR);
      const Aold = A;
      A = A - g / ((gp - gm) / (2 * eps));
      // Keep the area positive (MATLAB would silently go complex here).
      if (!(A > 0)) A = 0.1 * Aold;
      if (Math.abs(A - Aold) / (0.5 * (A + Aold)) <= tol) break;
    }
    const uS = 0.5 * (uL + uR) + 0.5 * (fK(v, aR, A, cR) - fK(v, aL, A, cL));
    return [A, uS];
  }

  // Exact solution at x/t = xt (sampleERP). Returns [A, u].
  function sampleERP(v, aL, aR, uL, uR, aS, uS, xt) {
    const cS = v.waveSpeed(aS), cL = v.waveSpeed(aL), cR = v.waveSpeed(aR);
    const g2 = v.gamma * v.gamma;
    if (xt < uS) {
      // Left wave
      if (aS <= aL) {
        if (xt < uL - cL) return [aL, uL];
        if (xt < uS - cS) {
          const cr = (uL + 4 * cL - xt) / 5;
          return [Math.pow(cr, 4) * 4 / 9 / g2, (uL + 4 * cL + 4 * xt) / 5];
        }
        return [aS, uS];
      }
      const sL = uL - mK(v, aL, aS) / aL;
      return xt < sL ? [aL, uL] : [aS, uS];
    }
    // Right wave
    if (aS <= aR) {
      if (xt < uS + cS) return [aS, uS];
      if (xt < uR + cR) {
        const cr = (-uR + 4 * cR + xt) / 5;
        return [Math.pow(cr, 4) * 4 / 9 / g2, (uR - 4 * cR + 4 * xt) / 5];
      }
      return [aR, uR];
    }
    const sR = uR + mK(v, aR, aS) / aR;
    return xt < sR ? [aS, uS] : [aR, uR];
  }

  // Exact solution at time t on the points x. Returns {A, u}.
  function exactSolution(v, aL, aR, uL, uR, gate, t, x) {
    const s = solveERP(v, aL, aR, uL, uR);
    const A = new Float64Array(x.length), u = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const r = sampleERP(v, aL, aR, uL, uR, s[0], s[1], (x[i] - gate) / (t + 1e-10));
      A[i] = r[0]; u[i] = r[1];
    }
    return { A: A, u: u };
  }

  // Numerical fluxes (model.m). Each takes the left and right states at one interface and returns [F0, F1].

  function physFlux(v, A, q) { return [q, q * q / A + v.gamma * Math.pow(A, 1.5)]; }

  function fluxGodunovExact(v, AL, qL, AR, qR) {
    const uL = qL / AL, uR = qR / AR;
    const s = solveERP(v, AL, AR, uL, uR);
    const r = sampleERP(v, AL, AR, uL, uR, s[0], s[1], 0);
    return physFlux(v, r[0], r[0] * r[1]);
  }

  function fluxLaxFriedrichs(v, AL, qL, AR, qR, dt, dx) {
    const FL = physFlux(v, AL, qL), FR = physFlux(v, AR, qR), c = 0.5 * dx / dt;
    return [0.5 * (FL[0] + FR[0]) - c * (AR - AL), 0.5 * (FL[1] + FR[1]) - c * (qR - qL)];
  }

  function fluxLaxWendroff(v, AL, qL, AR, qR, dt, dx) {
    const FL = physFlux(v, AL, qL), FR = physFlux(v, AR, qR), c = 0.5 * dt / dx;
    return physFlux(v, 0.5 * (AL + AR) - c * (FR[0] - FL[0]), 0.5 * (qL + qR) - c * (FR[1] - FL[1]));
  }

  function fluxForce(v, AL, qL, AR, qR, dt, dx) {
    const lf = fluxLaxFriedrichs(v, AL, qL, AR, qR, dt, dx), lw = fluxLaxWendroff(v, AL, qL, AR, qR, dt, dx);
    return [0.5 * (lf[0] + lw[0]), 0.5 * (lf[1] + lw[1])];
  }

  const FLUXES = {
    "godunov": fluxGodunovExact,
    "lax-friedrichs": fluxLaxFriedrichs,
    "lax-wendroff": fluxLaxWendroff,
    "force": fluxForce,
  };

  // Riemann problems of Main.m and Mainrcolapso.m (aL, aR relative to a0)
  const PROBLEMS = {
    1: { name: "rarefacción-choque", aL: 2.2, aR: 1.1, uL: 0.0, uR: 0.0, tEnd: 0.014, gate: 0.1 },
    2: { name: "dúas rarefaccións", aL: 1.6, aR: 1.6, uL: -1.5, uR: 1.5, tEnd: 0.01, gate: 0.1 },
    3: { name: "dous choques", aL: 1.6, aR: 1.6, uL: 1.5, uR: -1.5, tEnd: 0.014, gate: 0.1 },
    4: { name: "choque", aL: 1.6, aR: 1.6, uL: -2.6575e-5, uR: 6.1230e-6, tEnd: 0.014, gate: 0.1 },
    5: { name: "punto sónico", aL: 10.0, aR: 1.0, uL: 0.0, uR: 0.0, tEnd: 0.007, gate: 0.1 },
    6: { name: "colapso", aL: null, aR: null, uL: -12.5, uR: 12.5, tEnd: 0.005, gate: 0.25,
         vessel: { L: 0.5, E: 3.0e5, h: 5.0e-4, r0: 1.0e-2, rho: 1000.0 } },
  };

  function makeProblem(rp, ncells) {
    const P = PROBLEMS[rp];
    const v = new Vessel(Object.assign({ ncells: ncells }, P.vessel || {}));
    let aL, aR;
    if (rp === 6) {
      aL = aR = 1.0e-4 / 1.0e-2 / 1.0e-2 * v.a0;
    } else {
      aL = P.aL * v.a0; aR = P.aR * v.a0;
    }
    return { v: v, aL: aL, aR: aR, uL: P.uL, uR: P.uR, tEnd: P.tEnd, gate: P.gate };
  }

  /* Time-stepping solver for one Riemann problem, as Main.m does, but advanced step by step so that the web
     page can animate it. step() returns false once tEnd is reached or the solution stops being admissible. */
  function Simulation(rp, ncells, CFL, flux) {
    const p = makeProblem(rp, ncells);
    Object.assign(this, p);
    this.rp = rp;
    this.CFL = CFL;
    this.fnum = FLUXES[flux];
    const n = ncells;
    this.A = new Float64Array(n);
    this.q = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const left = p.v.xC[i] <= p.gate;
      this.A[i] = left ? p.aL : p.aR;
      this.q[i] = left ? p.aL * p.uL : p.aR * p.uR;
    }
    this.F0 = new Float64Array(n + 1);
    this.F1 = new Float64Array(n + 1);
    this.t = 0;
    this.n = 0;
    this.blownUp = false;
  }

  Simulation.prototype.timeStep = function () {
    const v = this.v;
    let lmax = 0;
    for (let i = 0; i < this.A.length; i++) {
      lmax = Math.max(lmax, Math.abs(this.q[i] / this.A[i]) + v.waveSpeed(this.A[i]));
    }
    return this.CFL * v.dx / lmax;
  };

  Simulation.prototype.done = function () { return this.blownUp || this.t >= this.tEnd; };

  Simulation.prototype.step = function () {
    if (this.done()) return false;
    const v = this.v, A = this.A, q = this.q, F0 = this.F0, F1 = this.F1, N = A.length;
    let dt = this.timeStep();
    if (!(isFinite(dt) && dt > 0)) { this.blownUp = true; return false; }
    dt = Math.min(dt, this.tEnd - this.t);
    // Transparent boundary conditions (BCL = BCR = 1): ghost cells copy the boundary cells.
    for (let j = 0; j <= N; j++) {
      const l = Math.max(j - 1, 0), r = Math.min(j, N - 1);
      const F = this.fnum(v, A[l], q[l], A[r], q[r], dt, v.dx);
      F0[j] = F[0]; F1[j] = F[1];
    }
    const c = dt / v.dx;
    let ok = true;
    for (let i = 0; i < N; i++) {
      A[i] -= c * (F0[i + 1] - F0[i]);
      q[i] -= c * (F1[i + 1] - F1[i]);
      if (!(isFinite(A[i]) && A[i] > 0)) ok = false;
    }
    this.t += dt;
    this.n += 1;
    if (!ok) this.blownUp = true;
    return ok;
  };

  // Velocity u = q / A on the cells.
  Simulation.prototype.velocity = function () {
    const u = new Float64Array(this.A.length);
    for (let i = 0; i < u.length; i++) u[i] = this.q[i] / this.A[i];
    return u;
  };

  Simulation.prototype.exact = function () {
    return exactSolution(this.v, this.aL, this.aR, this.uL, this.uR, this.gate, this.t, this.v.xC);
  };

  // Run to the final time, as python/blood1d.run.
  function run(rp, ncells, CFL, flux, nmax) {
    const s = new Simulation(rp || 1, ncells || 200, CFL || 0.9, flux || "godunov");
    nmax = nmax || 100000;
    while (s.n < nmax && s.step()) { /* advance */ }
    return s;
  }

  const api = { Vessel, solveERP, sampleERP, exactSolution, FLUXES, PROBLEMS, makeProblem, Simulation, run };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Blood1D = api;
})(this);
