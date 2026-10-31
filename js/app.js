/* Mobile page: animate one Riemann problem of blood1d.js against its exact solution. */
(function () {
  "use strict";
  const B = window.Blood1D;
  const SHOWN = [1, 2, 3, 5, 6];   // RP4 of Main.m is an almost constant state, so it is not offered
  const DURATION = 4000;           // animation length in ms for the whole simulated time
  const $ = id => document.getElementById(id);
  const el = { rp: $("rp"), flux: $("flux"), nc: $("nc"), cfl: $("cfl"), ncOut: $("ncOut"), cflOut: $("cflOut"),
               play: $("play"), reset: $("reset"), time: $("time"), msg: $("msg"),
               vessel: $("vessel"), plotA: $("plotA"), plotU: $("plotU") };
  const fmt = (x, d) => x.toFixed(d).replace(".", ",");

  for (const k of SHOWN) {
    const o = document.createElement("option");
    o.value = k;
    o.textContent = `${SHOWN.indexOf(k) + 1}. ${B.PROBLEMS[k].name}`;
    el.rp.appendChild(o);
  }

  let sim, xFine, ranges, playing = false, simElapsed = 0, lastFrame = 0;

  // Axis ranges from the initial states and the exact solution at the final time, so that they stay fixed.
  function computeRanges(s) {
    const v = s.v, n = 801;
    xFine = new Float64Array(n);
    for (let i = 0; i < n; i++) xFine[i] = v.L * i / (n - 1);
    const ex = B.exactSolution(v, s.aL, s.aR, s.uL, s.uR, s.gate, s.tEnd, xFine);
    let a = [s.aL / v.a0, s.aR / v.a0], u = [s.uL, s.uR];
    for (let i = 0; i < n; i++) { a.push(ex.A[i] / v.a0); u.push(ex.u[i]); }
    const pad = (lo, hi, minSpan) => {
      const span = Math.max(hi - lo, minSpan), mid = 0.5 * (lo + hi);
      return [mid - 0.6 * span, mid + 0.6 * span];
    };
    const ra = pad(Math.min(...a), Math.max(...a), 0.2), ru = pad(Math.min(...u), Math.max(...u), 0.5);
    return { A: [Math.max(0, ra[0]), ra[1]], u: ru, rmax: Math.sqrt(ra[1]) };
  }

  function reset() {
    playing = false;
    simElapsed = 0;
    sim = new B.Simulation(+el.rp.value, +el.nc.value, +el.cfl.value, el.flux.value);
    ranges = computeRanges(sim);
    el.ncOut.textContent = el.nc.value;
    el.cflOut.textContent = fmt(+el.cfl.value, 2);
    el.msg.textContent = "";
    el.play.textContent = "▶ Simular";
    draw();
  }

  function frame(now) {
    if (!playing) return;
    simElapsed += Math.min(now - lastFrame, 100);
    lastFrame = now;
    const target = sim.tEnd * Math.min(1, simElapsed / DURATION);
    while (sim.t < target && sim.step()) { /* advance to the time of this frame */ }
    draw();
    if (sim.done()) {
      playing = false;
      el.play.textContent = "↺ Repetir";
      if (sim.blownUp) {
        el.msg.textContent = sim.CFL > 1
          ? `O esquema explotou no paso ${sim.n}: con CFL > 1 non é estable.`
          : `A solución deixou de ser válida no paso ${sim.n}.`;
      }
      return;
    }
    requestAnimationFrame(frame);
  }

  el.play.addEventListener("click", () => {
    if (sim.done()) reset();
    playing = !playing;
    el.play.textContent = playing ? "❚❚ Pausa" : "▶ Seguir";
    if (playing) { lastFrame = performance.now(); requestAnimationFrame(frame); }
  });
  el.reset.addEventListener("click", reset);
  for (const c of [el.rp, el.flux, el.nc, el.cfl]) c.addEventListener("input", reset);

  // Drawing

  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function setup(canvas) {
    const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function niceTicks(lo, hi, n) {
    const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * p).find(s => s >= raw);
    const t = [];
    for (let x = Math.ceil(lo / step) * step; x <= hi + 1e-12; x += step) t.push(Math.abs(x) < 1e-12 ? 0 : x);
    return t;
  }

  function plot(canvas, xr, yr, exact, num, showX) {
    const { ctx, w, h } = setup(canvas);
    const L = 40, R = 8, T = 6, Bm = showX ? 22 : 8;
    const X = x => L + (x - xr[0]) / (xr[1] - xr[0]) * (w - L - R);
    const Y = y => T + (yr[1] - y) / (yr[1] - yr[0]) * (h - T - Bm);
    const muted = css("--muted"), line = css("--line");
    ctx.font = "12px system-ui, sans-serif";
    ctx.lineWidth = 1;
    ctx.strokeStyle = line;
    ctx.fillStyle = muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yt = niceTicks(yr[0], yr[1], 4), yd = Math.max(0, -Math.floor(Math.log10(yt[1] - yt[0] || 1)));
    for (const y of yt) {
      ctx.beginPath(); ctx.moveTo(L, Y(y)); ctx.lineTo(w - R, Y(y)); ctx.stroke();
      ctx.fillText(fmt(y, yd), L - 5, Y(y));
    }
    if (showX) {
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (const x of niceTicks(xr[0], xr[1], 4)) ctx.fillText(fmt(x, 2), X(x), h - Bm + 5);
    }
    ctx.save();
    ctx.beginPath(); ctx.rect(L, T, w - L - R, h - T - Bm); ctx.clip();
    ctx.strokeStyle = css("--exact");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < exact.x.length; i++) {
      const px = X(exact.x[i]), py = Y(exact.y[i]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.stroke();
    ctx.fillStyle = css("--blood");
    const r = num.x.length > 200 ? 1.8 : 2.6;
    for (let i = 0; i < num.x.length; i++) {
      if (!isFinite(num.y[i])) continue;
      ctx.beginPath(); ctx.arc(X(num.x[i]), Y(num.y[i]), r, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.restore();
  }

  // Vessel contour: radius proportional to sqrt(A / a0), numerical filled, exact dashed.
  function drawVessel(ex) {
    const { ctx, w, h } = setup(el.vessel);
    const v = sim.v, mid = h / 2, scale = (h / 2 - 4) / ranges.rmax;
    const X = x => x / v.L * w;
    const rad = a => (isFinite(a) && a > 0 ? Math.min(Math.sqrt(a / v.a0), ranges.rmax) : 0) * scale;
    const xs = v.xC, A = sim.A;
    ctx.fillStyle = css("--blood-soft");
    ctx.strokeStyle = css("--blood");
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, mid - rad(A[0]));
    for (let i = 0; i < xs.length; i++) ctx.lineTo(X(xs[i]), mid - rad(A[i]));
    ctx.lineTo(w, mid - rad(A[xs.length - 1]));
    ctx.lineTo(w, mid + rad(A[xs.length - 1]));
    for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(X(xs[i]), mid + rad(A[i]));
    ctx.lineTo(0, mid + rad(A[0]));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = css("--exact");
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 4]);
    for (const s of [-1, 1]) {
      ctx.beginPath();
      for (let i = 0; i < xFine.length; i++) {
        const px = X(xFine[i]), py = mid + s * rad(ex.A[i]);
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  function draw() {
    const v = sim.v;
    const ex = B.exactSolution(v, sim.aL, sim.aR, sim.uL, sim.uR, sim.gate, sim.t, xFine);
    const u = sim.velocity();
    const aRel = Float64Array.from(sim.A, a => a / v.a0), exRel = Float64Array.from(ex.A, a => a / v.a0);
    drawVessel(ex);
    plot(el.plotA, [0, v.L], ranges.A, { x: xFine, y: exRel }, { x: v.xC, y: aRel }, false);
    plot(el.plotU, [0, v.L], ranges.u, { x: xFine, y: ex.u }, { x: v.xC, y: u }, true);
    el.time.textContent = `t = ${fmt(sim.t * 1000, 1)} ms`;
  }

  window.addEventListener("resize", () => sim && draw());
  reset();
})();
