"""
maximin_solver.py

branch-and-prune for:

    min  f0(x)
    s.t. fi(x) <= 0,  i = 1 ... m
         lb <= x <= ub

Algorithm overview
------------------
The current best primal value U is folded into the Lagrangian as an
explicit constraint f0(x) <= U with its own dual variable y_U:

    L(x, y) = (1 + y_U) f0(x) + sum_i y_i fi(x) - y_U U

Per-box, the solver alternates between primal and dual updates:
  1. Fix y.  Minimize L(x, y) over x in the box via projected gradient
     descent until x is stationary (partial stationarity in x).
  2. Take one dual ascent step:  y <- max(0, y + lr_y * grad_y L)
     where grad_y L = constraint values at x*(y).
  3. Repeat until full joint stationarity (KKT) is reached, or the
     dual variables grow without bound.

Termination per box
-------------------
Three outcomes are possible:

  KKT found       -- (x, y) satisfies KKT to within tolerance.
                      Update U if improved, then branch.

  y diverges      -- ||y|| exceeds div_thresh; the box is pruned. A
                      numerical pruning heuristic, not a general
                      infeasibility proof -- see paper.md Sec. 5.2-5.3
                      (Theorem 1) for exactly what case this is
                      justified in and what remains open.

  iteration limit -- max_outer reached with neither of the above; the
                      box is accepted or rejected from the final
                      constraint check alone, with no convergence
                      guarantee.

branch-and-prune structure
--------------------------
  - Upper bound U: best primal value found across all accepted points
    (a genuine KKT point, or an iteration-limit timeout whose final
    iterate happened to be feasible -- these are recorded as distinct
    statuses, see Status below, never conflated).
  - Branch: split along the widest box dimension whenever a box's point
    is accepted (KKT satisfied, or iteration limit reached feasible).
  - Prune: discard a box when the dual variables diverge, or when the
    iteration limit is reached with an infeasible final iterate. Only
    the former is a divergence signal in Theorem 1's sense; the latter
    is an inconclusive timeout, not a claim of infeasibility.
"""

import enum

import numpy as np
import sympy as sp


class Status(enum.Enum):
    """Per-box outcome of _kkt, kept as distinct, explicit values so a
    genuine KKT point, a proven-divergence prune, and an inconclusive
    iteration-limit timeout can never be confused with one another."""
    KKT = "kkt"
    DIVERGED = "diverged"
    ITERATION_LIMIT_FEASIBLE = "iteration_limit_feasible"
    ITERATION_LIMIT_INFEASIBLE = "iteration_limit_infeasible"


# ─────────────────────────────────────────────────────────── parsing

def parse_pd(f0_str, fi_strs):
    """
    Parse objective and constraint expression strings.

    Only first derivatives are needed (gradient descent, no Newton).

    Returns
    -------
    f0_fn, grad_f0           objective and its gradient
    fi_fns, grad_fi_fns      constraint functions and gradients
    syms                      list of sympy.Symbol (alphabetical)
    """
    all_exprs = [sp.sympify(s) for s in [f0_str] + list(fi_strs)]
    syms = sorted(
        set().union(*[e.free_symbols for e in all_exprs]),
        key=lambda s: s.name,
    )

    def _compile(expr):
        fn_raw   = sp.lambdify(syms, expr, modules='numpy')
        grad_raw = sp.lambdify(syms, [sp.diff(expr, s) for s in syms],
                               modules='numpy')
        def fn(x):   return float(fn_raw(*x))
        def grad(x): return np.array([float(v) for v in grad_raw(*x)], float)
        return fn, grad

    f0_fn, grad_f0 = _compile(all_exprs[0])
    fi_fns, grad_fi_fns = [], []
    for expr in all_exprs[1:]:
        fn, grad = _compile(expr)
        fi_fns.append(fn)
        grad_fi_fns.append(grad)

    return f0_fn, grad_f0, fi_fns, grad_fi_fns, syms


# ─────────────────────────────────────────────────────────── saddle-point solver

def _kkt(f0_fn, grad_f0, fi_fns, grad_fi_fns, lo, hi, U,
         lr_x=0.02, lr_y=0.5, max_outer=300, max_inner=1000,
         tol=1e-6, div_thresh=1e8):
    """
    Per-box alternating primal-dual iteration.

    Each outer iteration:
      - Minimizes L(x, y) in x (inner loop) until x is stationary.
      - Checks KKT (joint stationarity of x and y).
      - Takes one dual ascent step if KKT is not yet met.

    Divergence (||y|| > div_thresh) is treated as a pruning signal, not
    a general infeasibility proof -- see paper.md Sec. 5.2-5.3 (Theorem 1)
    for exactly what's established and what remains open. If neither KKT
    nor divergence is reached within max_outer iterations, the box falls
    through to a final constraint check with no convergence guarantee --
    that outcome is its own status, never relabeled as a proven KKT point
    or a proven divergence.

    Returns (x, y, status), status a Status enum member:
        Status.KKT                        KKT satisfied to tolerance.
        Status.DIVERGED                   ||y|| exceeded div_thresh.
        Status.ITERATION_LIMIT_FEASIBLE   max_outer reached; final iterate
                                           satisfies all constraints to
                                           tolerance anyway.
        Status.ITERATION_LIMIT_INFEASIBLE max_outer reached; final iterate
                                           does not.
    """
    d = len(lo)
    m = len(fi_fns)
    # No finite incumbent yet, so f0(x) <= U is not a real constraint.
    # Leave it structurally inactive (never contributes to the gradient,
    # the KKT check, or the dual update) rather than approximating
    # "inactive" with a large constant. y[m] is simply never updated
    # while target_active is False, so it stays exactly 0.
    target_active = np.isfinite(U)
    active = slice(0, m + 1) if target_active else slice(0, m)

    def constraints(x):
        c = np.empty(m + 1)
        for i in range(m):
            c[i] = fi_fns[i](x)
        c[m] = (f0_fn(x) - U) if target_active else 0.0
        return c

    x = (lo + hi) / 2.0
    y = np.zeros(m + 1)    # y[0..m-1] for fi(x)<=0,  y[m] for f0(x)-U<=0
    stagnant = 0            # consecutive outer iterations with no primal movement

    for _ in range(max_outer):

        x_before_outer = x.copy()

        # ── Inner loop: fix y, minimize L(x,y) in x ─────────────────
        pgx_norm = np.inf
        for _ in range(max_inner):
            gx = (1.0 + y[m]) * grad_f0(x)
            for i in range(m):
                gx = gx + y[i] * grad_fi_fns[i](x)

            pgx = gx.copy()
            for k in range(d):
                if x[k] <= lo[k] + 1e-12 and pgx[k] > 0: pgx[k] = 0.0
                if x[k] >= hi[k] - 1e-12 and pgx[k] < 0: pgx[k] = 0.0

            x = np.clip(x - lr_x * gx, lo, hi)
            pgx_norm = float(np.linalg.norm(pgx))
            if pgx_norm < tol:
                break

        # ── Constraint values ─────────────────────────────────────────
        c = constraints(x)

        # ── KKT check: primal feasible + complementary slackness ─────
        if pgx_norm < tol and np.all(c[active] <= tol) and np.all(np.abs(y * c)[active] <= tol):
            return x, y, Status.KKT

        # ── Stagnation escape ────────────────────────────────────────
        # A box symmetric about a degenerate stationary point of L (e.g. any problem
        # where dL/dx factors as (x - midpoint) * (...), so the midpoint has zero
        # gradient for every y) leaves x pinned at that point every outer iteration
        # while y climbs, even when the point is an unstable local max of L rather
        # than a min and the box genuinely contains feasible points elsewhere.
        #
        # This must NOT be confused with the ordinary, correct way most target-
        # infeasible boxes get pruned: x settles at the box's boundary-constrained
        # optimum (pinned there by np.clip) and rightly stays put while y climbs
        # toward the divergence threshold -- the pruning signal Theorem 1 justifies
        # only in its specific conditional case (original constraints feasible,
        # target unattainable; see paper.md Sec. 5.2-5.3), not a general certificate
        # for every box. That is x stopping because it hit a wall,
        # not because its gradient vanished -- clipping is not degeneracy. Escaping
        # that case too was the bug: it fired on ~half the boxes in a plain convex
        # unconstrained problem and each escape re-ran the full inner loop, causing
        # a ~60x slowdown for no benefit, since those boxes were already on track.
        #
        # So only treat it as the pathological case when x is stuck in the box's
        # INTERIOR (away from every bound) with an unprojected gradient that is
        # itself ~0 -- a genuine critical point of L, not a clipped boundary point.
        interior = np.all(x > lo + 1e-9) and np.all(x < hi - 1e-9)
        if interior and np.allclose(x, x_before_outer, atol=1e-10) and pgx_norm < tol:
            stagnant += 1
        else:
            stagnant = 0
        if stagnant >= 3:
            x = np.clip(x + 1e-3 * (hi - lo), lo, hi)
            stagnant = 0

        # ── One dual ascent step ──────────────────────────────────────
        if target_active:
            y = np.maximum(0.0, y + lr_y * c)
        else:
            y[:m] = np.maximum(0.0, y[:m] + lr_y * c[:m])
            # y[m] left at 0: the target constraint is inactive.

        if np.linalg.norm(y) > div_thresh:
            return x, y, Status.DIVERGED

    # Iteration limit reached: neither KKT nor divergence was established.
    # Report which way the final iterate happened to fall without calling it
    # a KKT point or a divergence -- it's neither, just an inconclusive timeout.
    c = constraints(x)
    feasible = bool(np.all(c[active] <= tol))
    return x, y, (Status.ITERATION_LIMIT_FEASIBLE if feasible
                  else Status.ITERATION_LIMIT_INFEASIBLE)


# ─────────────────────────────────────────────────────────── branch-and-prune

def solve_pd(f0_fn, grad_f0, fi_fns, grad_fi_fns, lb, ub,
             eps=1e-3, max_boxes=2000, lr_x=0.02, lr_y=0.5, verbose=True):
    """
    branch-and-prune via alternating primal-dual iteration.

    Each box is processed by alternating between minimizing x and ascending y
    until a KKT point is reached or y diverges (box pruned).

    Parameters
    ----------
    f0_fn, grad_f0           objective and its gradient  (from parse_pd)
    fi_fns, grad_fi_fns      inequality constraint functions and gradients
    lb, ub        array-like   initial bounding box
    eps           float        branch until box diameter < eps
    max_boxes     int          maximum boxes to process
    lr_x          float        primal step size (inner minimization)
    lr_y          float        dual step size (outer ascent)
    verbose       bool         print progress

    Returns
    -------
    x_best    ndarray   global minimizer found
    f_best    float     optimal primal value
    n_boxes   int       total boxes processed
    """
    lb = np.atleast_1d(np.asarray(lb, float))
    ub = np.atleast_1d(np.asarray(ub, float))
    d  = len(lb)
    m  = len(fi_fns)

    if verbose:
        print(f"\n[maximin_solver]  d={d}  constraints={m}  eps={eps}  lr_x={lr_x}  lr_y={lr_y}")

    U       = np.inf
    x_best  = None
    n_boxes = 0
    stack   = [(lb.copy(), ub.copy())]    # LIFO: depth-first

    while stack and n_boxes < max_boxes:
        lo_box, hi_box = stack.pop()
        n_boxes += 1

        x, y, box_status = _kkt(f0_fn, grad_f0, fi_fns, grad_fi_fns,
                                 lo_box, hi_box, U,
                                 lr_x=lr_x, lr_y=lr_y)

        if box_status is Status.DIVERGED:
            if verbose:
                print(f"    box {n_boxes:5d}: pruned  (dual diverged)")
            continue

        if box_status is Status.ITERATION_LIMIT_INFEASIBLE:
            if verbose:
                print(f"    box {n_boxes:5d}: pruned  (iteration limit, infeasible)")
            continue

        # box_status is Status.KKT or Status.ITERATION_LIMIT_FEASIBLE: x is
        # accepted as a candidate point -- a genuine KKT point in the former
        # case, or merely feasible-looking at the iteration limit in the
        # latter. Both are treated identically as incumbent/branch candidates,
        # exactly as before; only the internal label is now explicit about
        # which one actually happened.
        f0_val   = f0_fn(x)
        improved = f0_val < U - 1e-8

        if improved:
            U, x_best = f0_val, x.copy()
            if verbose:
                print(f"    box {n_boxes:5d}: U = {U:+.8f}  x = {x}")

        # Branch when a new best was found, OR when a candidate was found that
        # didn't improve U — the local minimizer may be in the wrong basin;
        # sub-boxes with a different initialization may find a genuinely
        # better point.
        if float(np.max(hi_box - lo_box)) > eps:
            idx = int(np.argmax(hi_box - lo_box))
            mid = (lo_box[idx] + hi_box[idx]) / 2.0

            lo1, hi1 = lo_box.copy(), hi_box.copy(); hi1[idx] = mid
            lo2, hi2 = lo_box.copy(), hi_box.copy(); lo2[idx] = mid
            stack.append((lo1, hi1))
            stack.append((lo2, hi2))

    status = "done" if not stack else "budget exhausted"
    if verbose:
        print(f"\n[maximin_solver] {status}  f_best={U:.6g}  boxes={n_boxes}")

    return x_best, U, n_boxes


# ─────────────────────────────────────────────────────────── tests

if __name__ == '__main__':
    print("=== Constrained: min x1^2+x2^2  s.t. x1+x2>=1,  x in [0,1]^2 ===")
    print("    Expected: f*=0.5  x*=(0.5, 0.5)\n")

    f0_fn, grad_f0, fi_fns, grad_fi_fns, syms = parse_pd(
        'x1**2 + x2**2',
        ['-x1 - x2 + 1'],       # -x1-x2+1 <= 0  ↔  x1+x2 >= 1
    )
    x, f, nb = solve_pd(f0_fn, grad_f0, fi_fns, grad_fi_fns,
                         lb=[0, 0], ub=[1, 1],
                         eps=1e-2, max_boxes=500, lr_x=0.02, verbose=True)
    print(f"\n    x* = {x}   f* = {f:.6f}\n")

    print("=== Unconstrained: min (x-0.3)^2,  x in [0,1] ===")
    print("    Expected: f*=0  x*=0.3\n")

    f0_fn2, grad_f02, fi_fns2, grad_fi_fns2, _ = parse_pd('(x - 0.3)**2', [])
    x2, f2, nb2 = solve_pd(f0_fn2, grad_f02, fi_fns2, grad_fi_fns2,
                             lb=[0], ub=[1],
                             eps=1e-2, max_boxes=200, lr_x=0.05, verbose=True)
    print(f"\n    x* = {x2}   f* = {f2:.6f}")
