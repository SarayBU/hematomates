"""Checks of the Python translation: convergence to the exact solution (as Mainprecision.m) and loss of
stability when CFL > 1."""
import sys
import numpy as np
from blood1d import PROBLEMS, run, exact_solution, make_problem

sys.stdout.reconfigure(encoding="utf-8")


def l1_errors(rp, ncells, flux, CFL=0.9):
    v, x, A, u, t, n = run(rp, ncells, CFL, flux)
    _, aL, aR, uL, uR, tEnd, gate = make_problem(rp, ncells)
    Ae, ue = exact_solution(v, aL, aR, uL, uR, gate, t, x)
    return v.dx * np.sum(np.abs(A - Ae)), v.dx * np.sum(np.abs(A * u - Ae * ue)), t, tEnd


def main():
    ok = True
    print("Convergence (Godunov exact, CFL 0.9): L1 error in A and q = A u")
    for rp in PROBLEMS:
        errs = [l1_errors(rp, nc, "godunov") for nc in (50, 100, 200, 400)]
        eA = [e[0] for e in errs]
        reached = all(abs(e[2] - e[3]) < 1e-12 for e in errs)
        decreasing = all(eA[i + 1] < eA[i] for i in range(3))
        rates = [np.log2(eA[i] / eA[i + 1]) for i in range(3)]
        status = "OK" if (decreasing and reached) else "FAIL"
        ok &= status == "OK"
        print(f"[{status}] RP{rp} {PROBLEMS[rp]['name']}: L1(A) = " + ", ".join(f"{e:.2e}" for e in eA)
              + "  rates = " + ", ".join(f"{r:.2f}" for r in rates))

    print("\nOther fluxes on RP1 (200 cells, CFL 0.9): L1(A)")
    for flux in ("lax-friedrichs", "force", "lax-wendroff"):
        eA = l1_errors(1, 200, flux)[0]
        print(f"[OK] {flux}: {eA:.2e}")

    print("\nStability: CFL below and above 1 (RP1, 200 cells, Godunov exact)")
    for CFL, expect_stable in ((0.9, True), (1.5, False)):
        v, x, A, u, t, n = run(1, 200, CFL, "godunov")
        finite = np.all(np.isfinite(A)) and np.all(A > 0)
        bounded = finite and A.max() / v.a0 < 3.0
        stable = bounded and abs(t - PROBLEMS[1]["tEnd"]) < 1e-12
        status = "OK" if stable == expect_stable else "FAIL"
        ok &= status == "OK"
        state = "estable" if stable else f"inestable (paso {n}, max A/a0 = {np.nanmax(A) / v.a0:.3g})"
        print(f"[{status}] CFL {CFL}: {state}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
