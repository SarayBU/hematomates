"""Write the Python results that tests/check.html compares the JavaScript port against (tests/ref.js)."""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
from blood1d import PROBLEMS, run, make_problem, exact_solution  # noqa: E402

CASES = [(rp, "godunov", 0.9) for rp in PROBLEMS]
CASES += [(1, f, 0.9) for f in ("lax-friedrichs", "lax-wendroff", "force")]
CASES += [(1, "godunov", 1.5)]


def case(rp, flux, CFL, ncells=200):
    v, x, A, u, t, n = run(rp, ncells, CFL, flux)
    _, aL, aR, uL, uR, tEnd, gate = make_problem(rp, ncells)
    Ae, ue = exact_solution(v, aL, aR, uL, uR, gate, t, x)
    fin = lambda a: [float(z) if np.isfinite(z) else None for z in a]
    return dict(rp=rp, flux=flux, CFL=CFL, ncells=ncells, a0=v.a0, t=t, n=n,
                A=fin(A), u=fin(u), Aexact=fin(Ae), uexact=fin(ue))


ref = [case(*c) for c in CASES]
(HERE / "ref.js").write_text("window.REF = " + json.dumps(ref) + ";\n", encoding="utf-8")
print(len(ref), "cases written")
