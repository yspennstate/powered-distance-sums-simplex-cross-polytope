"""Computer checks for "Constant powered-distance sums characterize the regular simplex and the cross-polytope".
Run: python nr_check.py  (numpy, scipy; prints nr_check_out.txt)
(1) twisted cube, d=3,4,5: exact constancy of sum |p-x|^{2k}, k=1,2,3, at exact rational points of a concentric sphere,
    plus non-central-symmetry and distinctness;
(2) raw-distance least squares: d+1 points with lambda=2,4 (and lambda=4 alone) on one sphere -> every zero-residual
    solution is a regular simplex centred at the sphere's centre;
(3) 2d points with lambda=2,4,6 -> every zero-residual solution is a regular cross-polytope;
(4) control: the residual is NOT zero for a perturbed simplex / cross-polytope, and an irregular weighted fourth-power
    simplex with unequal weights passes the weighted fourth-power test (so weights matter)."""
import itertools, random
from fractions import Fraction as F
import numpy as np
from scipy.optimize import least_squares

def sphere_pts(d, R, n, seed):  # exact rational points on the sphere of radius R (inverse stereographic)
    rnd = random.Random(seed); out = []
    for _ in range(n):
        t = [F(rnd.randint(-9, 9), rnd.randint(1, 7)) for _ in range(d - 1)]
        s = sum(x * x for x in t)
        out.append([R * 2 * x / (1 + s) for x in t] + [R * (1 - s) / (1 + s)])
    return out

def sums(X, p, k):
    return sum(sum((pi - xi) ** 2 for pi, xi in zip(p, x)) ** k for x in X)

print("(1) twisted cube")
c, s = F(3, 5), F(4, 5)
for d in (3, 4, 5):
    B = list(itertools.product([-1, 1], repeat=d - 1))
    def Q(v):
        v = list(v); return [c * v[0] - s * v[1], s * v[0] + c * v[1]] + v[2:]
    X = [list(map(F, v)) + [F(1)] for v in B] + [Q(v) + [F(-1)] for v in B]
    distinct = len(set(tuple(x) for x in X)) == 2 ** d
    cs = all(tuple(-a for a in x) in set(tuple(y) for y in X) for x in X)
    ok = True
    for R in (F(1, 2), F(1), F(3), F(7, 2)):
        P = sphere_pts(d, R, 12, d * 100 + int(R * 2))
        for k in (1, 2, 3):
            vals = set(sums(X, p, k) for p in P)
            ok &= len(vals) == 1
    # k=4 should fail (not a 4-design): report
    P = sphere_pts(d, F(1), 12, 7)
    k4 = len(set(sums(X, p, 4) for p in P))
    print("  d=%d sites %d distinct %s centrally-symmetric %s  k=1,2,3 constant on 4 concentric spheres: %s  (k=4 distinct values: %d)" % (d, len(X), distinct, cs, ok, k4))
    # exact certificate: equal norms, sum x = 0, sum x x^T = c I, sum x(x)x(x)x = 0  =>  each sum is a polynomial in |p|^2
    rad = set(sum(a * a for a in x) for x in X)
    m1 = all(sum(x[i] for x in X) == 0 for i in range(d))
    M2 = [[sum(x[i] * x[j] for x in X) for j in range(d)] for i in range(d)]
    m2 = all(M2[i][j] == (M2[0][0] if i == j else 0) for i in range(d) for j in range(d))
    m3 = all(sum(x[i] * x[j] * x[k] for x in X) == 0 for i in range(d) for j in range(d) for k in range(d))
    print("     exact: radii %s, first moment 0 %s, second moment scalar %s (=%s), third moment tensor 0 %s" % (sorted(rad), m1, m2, M2[0][0], m3))

rng = np.random.default_rng(1)
def U(d, m):
    u = rng.normal(size=(m, d)); return u / np.linalg.norm(u, axis=1, keepdims=True)

def fit(N, d, lams, starts, R=1.0, weights=None):
    T = R * U(d, 300)
    def res(z):
        A = z.reshape(N, d); D2 = ((T[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        r = []
        for lam in lams:
            v = (D2 ** (lam // 2)).sum(1) if weights is None else (weights * D2 ** (lam // 2)).sum(1)
            r.append((v - v.mean()) / max(1.0, abs(v.mean())))
        return np.concatenate(r)
    sols = []
    for _ in range(starts):
        z0 = rng.normal(scale=1.5, size=N * d)
        out = least_squares(res, z0, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        sols.append((np.abs(res(out.x)).max(), out.x.reshape(N, d), res))
    return sols

def describe_simplex(A):
    # all quantities relative to the mean distance of the points from the centre of the sphere
    c = A.mean(0); r = np.linalg.norm(A, axis=1); s = max(r.mean(), 1e-300)
    D = np.linalg.norm(A[:, None] - A[None], axis=-1)
    off = D[~np.eye(len(A), dtype=bool)]
    return np.linalg.norm(c) / s, (r.max() - r.min()) / s, (off.max() - off.min()) / s, off.min() / s

def describe_cross(A):
    r = np.linalg.norm(A, axis=1); G = A @ A.T / np.outer(r, r)
    # each site should have exactly one antipode (cos=-1) and the rest orthogonal
    ok = all(np.sum(np.abs(G[i] + 1) < 1e-6) == 1 and np.sum(np.abs(G[i]) < 1e-6) == len(A) - 2 for i in range(len(A)))
    s = max(r.mean(), 1e-300)
    return np.linalg.norm(A.mean(0)) / s, (r.max() - r.min()) / s, ok, np.min(np.linalg.norm(A[:, None] - A[None], axis=-1) / s + 9 * np.eye(len(A)))

print("(2) d+1 points, lambda=2,4 and lambda=4 alone")
for d in (3, 4, 5):
    for lams in ((2, 4), (4,)):
        sols = fit(d + 1, d, lams, 12)
        zero = [A for e, A, _ in sols if e < 1e-9]
        bad = [A for A in zero if (describe_simplex(A)[2] > 1e-5 or describe_simplex(A)[0] > 1e-5) and describe_simplex(A)[3] > 1e-3]
        print("  d=%d lams=%s starts=12 zero-residual %d, of which NOT a centred regular simplex (and distinct): %d" % (d, lams, len(zero), len(bad)))
        for A in bad: print("     BAD", np.round(describe_simplex(A), 7), "resid", min(e for e, B, _ in sols if B is A))

print("(3) 2d points, lambda=2,4,6")
for d in (3, 4):
    sols = fit(2 * d, d, (2, 4, 6), 12)
    zero = [A for e, A, _ in sols if e < 1e-9]
    good = [describe_cross(A) for A in zero]
    bad = [g for g in good if g[3] > 1e-3 and not (g[0] < 1e-5 and g[1] < 1e-5 and g[2])]
    print("  d=%d starts=12 zero-residual %d, distinct and NOT a centred regular cross-polytope: %d" % (d, len(zero), len(bad)))
    for g in bad: print("     BAD", g)
    print("     min residual over starts %.2e, median %.2e" % (min(e for e, _, _ in sols), np.median([e for e, _, _ in sols])))

print("(4) controls")
d = 4
S = np.vstack([np.eye(d), np.ones(d) * (1 - np.sqrt(d + 1)) / d]); S -= S.mean(0)
T = U(d, 300)
def osc(A, lam, w=None):
    D2 = ((T[:, None, :] - A[None]) ** 2).sum(-1); v = (D2 ** (lam // 2) * (1 if w is None else w)).sum(1); return v.max() - v.min()
print("  regular simplex d=4: osc lam2 %.1e lam4 %.1e" % (osc(S, 2), osc(S, 4)))
S2 = S.copy(); S2[0] *= 1.01
print("  one vertex scaled by 1.01: osc lam2 %.1e lam4 %.1e" % (osc(S2, 2), osc(S2, 4)))
C = np.vstack([np.eye(d), -np.eye(d)])
print("  cross-polytope d=4: osc lam2,4,6 %.1e %.1e %.1e; lam8 %.1e" % (osc(C, 2), osc(C, 4), osc(C, 6), osc(C, 8)))
C2 = C.copy(); C2[0] *= 1.01
print("  one vertex scaled: osc lam6 %.1e" % osc(C2, 6))
# weighted fourth-power simplex with unequal radii (A = R^2 = 1, H = 0), the BBS-type construction
r = np.array([0.6, 0.9, 1.0, 1.3, 1.7]); W = 1 + r ** 2
f = lambda t: np.sum(W ** 2 / (W ** 2 + t * r ** 2)) - 1
lo, hi = 1e-9, 1e9
for _ in range(200):
    mid = np.sqrt(lo * hi); lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
t = lo; b = W / np.sqrt(W ** 2 + t * r ** 2); lam = 1 / (W ** 2 + t * r ** 2); K = 1 / lam.sum(); lam *= K; s_ = K / t
E = np.eye(5); V = np.array([np.sqrt(s_ / lam[i]) * (E[i] - b[i] * b) for i in range(5)])
Bq, _ = np.linalg.qr(np.column_stack([b, np.eye(5)[:, :4]])); A5 = V @ Bq[:, 1:]
print("  weighted 4th-power simplex, radii", np.round(np.linalg.norm(A5, axis=1), 6), "weights", np.round(lam, 4))
print("     weighted osc lam4 %.1e (constant), unweighted osc lam4 %.1e, weighted osc lam2 %.1e (not constant)" % (osc(A5, 4, lam), osc(A5, 4), osc(A5, 2, lam)))
