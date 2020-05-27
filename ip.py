#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (C) 2020 Ahirui Otsu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import random
import operator
import itertools
import fractions
from sympy import Matrix, ImmutableMatrix, fraction, sqrt,\
    power, pprint, ceiling, gcd
import simplex_method
import matrices

def _is_integer(A):
    for e in A:
        if e.is_integer:
            continue
        return False
    return True

def _integral_vector_index(v):
    for i, w in enumerate(v):
        if _is_integer(w):
            return i
    return -1

def _find_initial_vertices(A, b, nonneg, mins):
    m, n = A.shape
    a = max(max(ceiling(abs(e)) for e in A), max(ceiling(abs(e)) for e in b))
    c = Matrix(1, n, [random.randint(-a, a) for i in range(n)])
    cnt = 0
    v = []
    while True:
        solver = simplex_method.SimplexMethod(A, b, c, nonneg, mins)
        _, x = solver.maximize()
        if solver.status != simplex_method.OPTIMAL:
            return None, None
        if len(v) == 0 or (len(v) == 1 and x != v[0]):
            v.append(x)
        if len(v) >= 2:
            break
        c = Matrix(1, n, [random.randint(-a, a) for i in range(n)])
    return v, Matrix(v[1] - v[0])

def _find_more_vertices(A, b, nonneg, mins, v, V):
    n = A.shape[1]
    d = V.shape[1]
    while d < n:
        dd = d
        coeffs = [x.T for x in V.T.nullspace()]
        for coeff in coeffs:
            for c in (coeff, -coeff):
                solver = simplex_method.SimplexMethod(A, b, c, nonneg, mins)
                _, x = solver.maximize()
                assert solver.status == simplex_method.OPTIMAL
                V = V.row_join(x - v[0])
                if len(V.T.nullspace()) == n - d - 1:
                    v.append(x)
                    dd += 1
                    break
                V.col_del(-1)
            if dd > d:
                break
        if dd == d:
            break
        d = dd
    return V

# Construct vertices v[0], ..., v[n] of K by maximizing functions.
# Instead of Khachiyan's algorithm I use simplex method.
def _construct_vertices(A, b, nonneg, mins):
    #print('Constructing vertices of K')
    m, n = A.shape
    # Find initial vertices v != 0 by maximizing arbitrary functions.
    v, V = _find_initial_vertices(A, b, nonneg, mins)
    if v is None:
        return None, None
    d = V.shape[1]
    V = _find_more_vertices(A, b, nonneg, mins, v, V)
    #print('Vertices')
    """
    for x in v:
        #pprint(x)
        y = A * x
        for z, w in zip(y, b):
            assert z <= w
    """
    return v, V

# If A0 = 0 <= b, then return True.
# Are there any simple functions that perform this?
def _zero_lt(b):
    for e in b:
        if e < 0:
            return False
    return True

def _reduce_problem(A, b, v, V, n, d):
    W = Matrix.zeros(n, d)
    for i in range(d):
        w = V.col(i)
        multiple = 1
        for e in w:
            if e == 0 or e.is_integer:
                continue
            denom = fraction(e)[1]
            multiple = multiple // gcd(multiple, denom) * denom
        W[:, i] = multiple * w
    U, _ = matrices.hermite_normal_form_row(W)
    #assert U is not None
    detU = U.det()
    #assert detU == 1 or detU == -1
    #print('U = ')
    #pprint(U)
    #assert U * W == _
    #print('UW = ')
    #pprint(_)
    Uinv = U.inv()
    r = U * v[0]
    for i in range(d, n):
        if not r[i].is_integer:
            #print(f'r[{i}] is not an integer')
            #pprint(r)
            return None, None, None, None, None
    v[0] = r[:d, :]
    VV = Matrix.zeros(d)
    for i in range(1, d + 1):
        y = U * v[i]
        v[i] = y[:d, :]
        VV[:, i - 1] = v[i] - v[0]
    #print('Reduced vertices')
    #for x in v:
        #pprint(x)

    b -= A * Uinv[:, d:] * r[d:, :]
    A *= Uinv[:, :d]
    z = Matrix.zeros(1, d)
    m = A.shape[0]
    #print('Reduced A = ')
    #pprint(A)
    #print('Reduced b = ')
    #pprint(b)

    return A, b, Uinv, VV, r

def _is_linearly_independent(v, V, i):
    n = v[0].shape[0]
    if i == 0:
        VV = Matrix.hstack(*(v[i] - v[0] for i in range(1, n + 1)))
        res = VV.det() != 0
    else:
        tmp = V.col(i - 1)
        V[:, i - 1] = v[i] - v[0]
        res = V.det() != 0
        V[:, i - 1] = tmp
    return res

def _update_V(v, V, i):
    n = v[0].shape[0]
    if i == 0:
        V = Matrix.hstack(*(v[i] - v[0] for i in range(1, n + 1)))
    else:
        V[:, i - 1] = v[i] - v[0]
    return V

def _maximize_g(A, b, nonneg, mins, c, coeffs, v, V):
    solver = simplex_method.SimplexMethod(A, b, c, nonneg, mins)
    _, x = solver.maximize()
    n = V.shape[0]
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                continue
            if 2 * abs(coeffs[i].dot(x - v[j])) > \
               3 * abs(coeffs[i].dot(v[i] - v[j])):
                tmp = v[i]
                v[i] = x
                if _is_linearly_independent(v, V, i):
                    V = _update_V(v, V, i)
                    #print('Vertex replaced')
                    return True, V
                v[i] = tmp
    return False, V

def _increase_volume(A, b, nonneg, mins, v, V):
    #print('Increasing volume(v[0], ..., v[n])')
    n = A.shape[1]
    #x = [Matrix(1, n, [j for j in range(n + 1) if j != i])
    #     for i in range(n + 1)]
    x = Matrix([[1 for i in range(n)]])
    found = True
    while found:
        coeffs = []
        for i in range(n + 1):
            X = Matrix.hstack(*(w - v[i] for j, w in enumerate(v) if j != i))
            assert X.det() != 0
            coeffs.append(x * X.inv())

        for i in range(n + 1):
            for j in range(n + 1):
                if j != i:
                    assert coeffs[i].dot(v[i]) != coeffs[i].dot(v[j])

        found = False
        for coeff in coeffs:
            for c in (coeff, -coeff):
                found, V = _maximize_g(A, b, nonneg, mins, c, coeffs, v, V)
                if found:
                    break
            if found:
                break
    return V

def _get_regular_simplex_vertices(n):
    e = Matrix.eye(n)
    v = Matrix.zeros(n, n + 1)
    v[:, 0] = e[:, 0]
    v[:, 1] = -e[:, 0]
    g = Matrix.zeros(n, 1)
    for i in range(2, n + 1):
        v[:, i] = sqrt(4 - v[:, i - 1].norm() ** 2) * e[:, i - 1]
        g = v[:, i] / (i + 1)
        for j in range(i + 1):
            v[:, j] -= g
    return v

# Construct a nonsingular endomorphism in order to get a basis for L
def _construct_endomorphism(A, b, v, V):
    #print('Constructing a nonsingular endomorphism M')
    n = len(v) - 1
    S = _get_regular_simplex_vertices(n)
    T = Matrix.zeros(n)
    s0 = S.col(0)
    T = Matrix.hstack(*(S.col(i) - s0 for i in range(1, n + 1)))
    M = T * T.inv()
    #print('M constructed')
    return M.inv()

def _gram_schmidt(b, B, u, mu):
    n = len(b)
    for i in range(n):
        u[i] = b[i]
        for j in range(i):
            mu[i][j] = b[i].dot(u[j]) / B[j]
            u[i] -= mu[i][j] * u[j]
        B[i] = u[i].dot(u[i])

def _reduce_mu(b, mu, k, l):
    if 2 * abs(mu[k][l]) > 1:
        r = round(mu[k][l])
        b[k] -= r * b[l]
        for j in range(l):
            mu[k][j] -= r * mu[l][j]
        mu[k][l] -= r

# Transform a basis for L into a reduced one.
def _transform_basis(M):
    #print('Transforming the basis for L into a reduced one')
    if M.shape[0] == 1:
        return [M.col(0)]
    n = M.shape[0]
    b = [M.col(i) for i in range(n)]
    B = [0] * n
    z = Matrix.zeros(n, 1)
    u = [z for i in range(n)]
    mu = [[0] * (n - 1) for i in range(n)]
    _gram_schmidt(b, B, u, mu)
    k = 1
    X = Matrix.eye(2)
    Y = Matrix([[0, 1], [1, 0]])
    while True:
        _reduce_mu(b, mu, k, k - 1)
        if 4 * B[k] < (3 - 4 * mu[k][k - 1] ** 2) * B[k - 1]:
            mux = mu[k][k - 1]
            Bx = B[k] + mux ** 2 * B[k - 1]
            mu[k][k - 1] = mux * B[k - 1] / Bx
            B[k] = B[k - 1] * B[k] / Bx
            B[k - 1] = Bx
            b[k - 1], b[k] = b[k], b[k - 1]
            for j in range(k - 1):
                mu[k - 1][j], mu[k][j] = mu[k][j], mu[k - 1][j]
            for i in range(k + 1, n):
                X[0, 1] = mu[k][k - 1]
                Y[1, 1] = -mux
                Z = X * Y * Matrix([mu[i][k - 1], mu[i][k]])
                mu[i][k - 1] = Z[0]
                mu[i][k] = Z[1]
            if k > 1:
                k -= 1
            continue
        for l in range(k - 2, -1, -1):
            _reduce_mu(b, mu, k, l)
        if k == n - 1:
            break
        k += 1

    # Sort b[i] so that |b[i - 1]| <= |b[i]|
    bn = [(e, e.dot(e)) for e in b]
    bn = sorted(bn, key=operator.itemgetter(1))
    b = [e[0] for e in bn]
    #print('Basis transformed')
    #pprint(b)
    return b

# TODO: よくわからんのでいつか直す
def _find_vector_in_L(L, A, Minv, b, nonneg, mins):
    #print('Finding a vector in L')
    n = len(L)
    k = [[0] for i in range(n)]
    p = 1
    # the number of the hyperplanes H + kb[n] (k in Z)
    # c1 = 2 * n^{3/2}
    # c2 = 2^{n(n - 1) / 4}
    # t - 1 < c1 * c2 * sqrt(n)
    t = ceiling(2 * n * sqrt(n) * power.Pow(2, fractions.Fraction(n * (n - 1), 4))) + 1
    ntries = 5 * t // 7 + 1
    for i in range(n - 1, -1, -1):
        ip1 = i + 1
        for j in range(1, ntries):
            k[i].append(j)
            k[i].append(-j)
        p *= len(k[i])
    print(f'Checking {p - 1} vectors')
    MinvL = Minv * Matrix.hstack(*L)
    step = 1000000
    for i, c in enumerate(itertools.product(*k)):
        if i == 0:
            continue
        if i % step == 0:
            print(f'{i}..', end='', flush=True)
        x = MinvL * Matrix(n, 1, c)
        satisfied = True
        for j, (xj, m) in enumerate(zip(x, mins)):
            if xj < m:
                satisfied = False
                break
        if not satisfied:
            continue
        y = A * x
        for yj, bj in zip(y, b):
            if yj > bj:
                satisfied = False
                break
        if satisfied:
            print()
            print(f'FOUND {i + 1}/{p - 1}')
            return x
    print()
    print('Not found')
    return None

def _satisfies(A, b, x, is_nonneg):
    for y, z in zip(A * x, b):
        if y > z:
            return False
    for i, e in enumerate(x):
        if is_nonneg[i] and e < 0:
            return False
    return True

def get_vector(A, b, nonneg, mins):
    #pprint('Integer programming Ax <= b')
    #print('A = ')
    #pprint(A)
    #print('b = ')
    #pprint(b)

    m, n = A.shape
    n_zeros = mins.count(0)
    if n_zeros == n and _zero_lt(b):
        #print('x = 0 satisfies the inequality.')
        return Matrix.zeros(n, 1)
    reduced_1 = False
    is_nonneg = [False] * n
    for e in nonneg:
        is_nonneg[e] = True
    if m < n:
        _, U1 = matrices.hermite_normal_form_col(A)
        A = H[:, :m]
        b = b[:, :m]
        m, n = A.shape
        reduced_1 = True
        nonneg = []
        mins = [0] * m

    v, V = _construct_vertices(A, b, nonneg, mins)
    if v is None:
        return None
    int_idx = _integral_vector_index(v)
    if int_idx != -1:
        x = v[int_idx]
        if reduced_1:
            x = U1 * x
        if _satisfies(A, b, x, is_nonneg):
            print('Integral vertex found')
            return x
    d = V.shape[1]
    reduced_2 = d < n
    if reduced_2:
        A, b, U2inv, V, r = _reduce_problem(A, b, v, V, n, d)
        nonneg = []
        mins = [0] * d
        if A is None:
            return None

    if reduced_2 and _zero_lt(b):
        print('Integral vertex found')
        x = U2inv[:, d:] * r[d:, :]
        if reduced_1:
            x = U1 * x
        return x

    V = _increase_volume(A, b, nonneg, mins, v, V)
    M = _construct_endomorphism(A, b, v, V)
    L = _transform_basis(M)
    Minv = M.inv()
    x = _find_vector_in_L(L, A, Minv, b, nonneg, mins)
    if x is None:
        return None
    if reduced_2:
        x = U2inv[:, :d] * x + U2inv[:, d:] * r[d:, :]
    if reduced_1:
        x = U1 * x
    return x

def main():
    A = Matrix([[3, 6],
                [3, 1],
                [-1, -2]])
    b = Matrix([27, 10, -7])
    nonneg = [0, 1]
    mins = [0, 0]
    x = get_vector(A, b, nonneg, mins)
    pprint(x)

if __name__ == '__main__':
    main()
