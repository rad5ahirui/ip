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

from sympy import Matrix, pprint, ceiling

def _extgcd(a, b):
    swapped = False
    if abs(a) < abs(b):
        swapped = True
        a, b = b, a
    s = a
    sx = 1
    sy = 0
    t = bp
    tx = 0
    ty = 1
    while t != 0 and s % t != 0:
        tmp = s // t
        u = s - t * tmp
        ux = sx - tx * tmp
        uy = sy - ty * tmp
        s = t
        sx = tx
        sy = ty
        t = u
        tx = ux
        ty = uy
    if t < 0:
        t = -t
        tx = -tx
        ty = -ty

    return (t, ty, tx) if swapped else (t, tx, ty)

def _reduce_off_diagonal_col(k, A, K):
    if A[k, k] < 0:
        A[:, k] = -A.col(k)
        K[:, k] = -K.col(k)
    A_k_k = A[k, k]
    A_k = A.col(k)
    K_k_k = K[k, k]
    K_k = K.col(k)
    for z in range(k - 1):
        r = ceiling(A[k, z] / A_k_k)
        A[:, z] = A.col(z) - r * A_k
        K[:, z] = K.col(z) - r * K_k

def hermite_normal_form_col(A):
    A = A[:, :]
    m, n = A.shape
    # Step 1
    K = Matrix.eye(n)
    # Step 2
    if A[0, 0] == 0:
        singular = True
        for j in range(1, n):
            if A[0, j] != 0:
                singular = False
                A.col_swap(0, j)
                K.col_swap(0, j)
                break
            if singular:
                return None, None
    r = min(m, n)
    rm1 = r - 1
    nm1 = n - 1
    for i in range(1, rm1):
        ip1 = i + 1
        minor = A[:ip1, :ip1]
        j = i
        while j < nm1 and minor.det() == 0:
            j += 1
            minor[:, i] = A.col(j)[:ip1, :]
        if minor.det() == 0:
            return None, None
        if j > i:
            A.col_swap(i, j)
            K.col_swap(i, j)
    # Step 3
    # Step 4
    for i in range(rm1):
        ip1 = i + 1
        for j in range(min(ip1, n)):
            # Step 4.1
            A_j_ip1 = A[j, ip1]
            if A_j_ip1 == 0:
                continue
            A_j_j = A[j, j]
            r, p, q = _extgcd(A_j_j, A_j_ip1)
            # Step 4.2
            D = Matrix([[p, -A_j_ip1 / r],
                        [q, A_j_j / r]])
            X = Matrix.hstack(*(A.col(j), A.col(ip1))) * D
            A[:, j] = X.col(0)
            A[:, ip1] = X.col(1)
            Y = Matrix.hstack(*(K.col(j), K.col(ip1))) * D
            K[:, j] = Y.col(0)
            K[:, ip1] = Y.col(1)
            # Step 4.3
            if 0 < j < rm1:
                _reduce_off_diagonal_col(j, A, K)
        # Step 5
        if ip1 < rm1:
            _reduce_off_diagonal_col(ip1, A, K)
        # Step 6
    return A, K

def _reduce_off_diagonal_row(k, A, U):
    if A[k, k] < 0:
        A[k, :] = -A.row(k)
        U[k, :] = -U.row(k)
    A_k_k = A[k, k]
    A_k = A.row(k)
    U_k_k = U[k, k]
    U_k = U.row(k)
    for z in range(k - 1):
        r = ceiling(A[z, k] / A_k_k)
        A[z, :] = A.row(z) - r * A_k
        U[z, :] = U.row(z) - r * U_k

def hermite_normal_form_row(A):
    A = A[:, :]
    m, n = A.shape
    if m == n:
        assert A.det() != 0
    # Step 1
    U = Matrix.eye(m)
    # Step 2
    if A[0, 0] == 0:
        singular = True
        for j in range(1, m):
            if A[j, 0] != 0:
                singular = False
                A.row_swap(0, j)
                U.row_swap(0, j)
                break
        if singular:
            return None, None
    r = min(m, n)
    rm1 = r - 1
    mm1 = m - 1
    for i in range(1, rm1):
        ip1 = i + 1
        minor = A[:ip1, :ip1]
        j = i
        while j < mm1 and minor.det() == 0:
            j += 1
            minor[i, :ip1] = A.row(j)[:, :ip1]
        if minor.det() == 0:
            return None, None
        if j > i:
            A.row_swap(i, j)
            U.row_swap(i, j)

    for i in range(1, r):
        assert A[:i, :i].det() != 0

    # Step 3
    # Step 4
    for i in range(mm1):
        ip1 = i + 1
        for j in range(min(ip1, n)):
            # Step 4.1
            A_ip1_j = A[ip1, j]
            if A_ip1_j == 0:
                continue
            A_j_j = A[j, j]
            #assert A_j_j != 0
            r, p, q = _extgcd(A_j_j, A_ip1_j)
            # Step 4.2
            D = Matrix([[p, q],
                        [-A_ip1_j / r, A_j_j / r]])
            X = D * Matrix.vstack(*(A.row(j), A.row(ip1)))
            A[j, :] = X.row(0)
            A[ip1, :] = X.row(1)
            Y = D * Matrix.vstack(*(U.row(j), U.row(ip1)))
            U[j, :] = Y.row(0)
            U[ip1, :] = Y.row(1)
            # Step 4.3
            if 0 < j < rm1:
                _reduce_off_diagonal_row(j, A, U)
        # Step 5
        if ip1 < rm1:
            _reduce_off_diagonal_row(ip1, A, U)
        # Step 6
    return U, A

def solution_space(A, b):
    m, n = A.shape
    r = min(m, n)
    i = 0
    j = 0
    leading_entries = []
    other_entries = []
    while i < m and j < n:
        # Pivoting
        pivot = -1
        for k in range(i, m):
            if A[k, j] != 0:
                pivot = k
                break
        if pivot == -1:
            other_entries.append(j)
            j += 1
            continue
        if pivot != i:
            A.row_swap(i, pivot)
            b.row_swap(i, pivot)
        b[i] /= A[i, j]
        A[i, :] /= A[i, j]
        for k in range(i + 1, m):
            b[k] = b[k] - A[k, j] * b[i]
            A[k, :] = A.row(k) - A[k, j] * A.row(i)
        leading_entries.append(j)
        i += 1
        j += 1
    rankA = i
    rankAb = rankA
    if rankA < m and max(b[rankA:, :]) != 0:
        rankAb += 1
        if rankAb < m:
            b[rankAb:] = Matrix.zeros(m - rankAb, 1)
        return None
    if rankA > 1:
        for i, pivot in enumerate(leading_entries[1:]):
            i += 1
            for j in range(i):
                b[j] = b[j] - A[j, pivot] * b[i]
                A[j, :] = A.row(j) - A[j, pivot] * A.row(i)
    x = [Matrix.zeros(n, 1)]
    for i, freevar in enumerate(other_entries):
        x.append(Matrix(n, 1, lambda x, y: 1 if x == freevar else 0))
    for i, pivot in enumerate(leading_entries):
        x[0][i] = b[i]
        for j, freevar in enumerate(other_entries):
            x[1 + j][i] = -A[i, freevar]
    return x

def main():
    from sympy import pprint
    A = Matrix([[4, 0, 0],
                [0, -5, -3],
                [0, 4, 7],
                [-4, 0, 0],
                [3, -2, 0]]).T
    U, H = hermite_normal_form_row(A[:, :])
    if H is None:
        print('singular')
    print('A =')
    pprint(A)
    print('[U, H] = hermite_normal_form_row(A)')
    print('U =')
    pprint(U)
    print('H =')
    pprint(H)
    print('UA =')
    pprint(U * A)
    print(f'det(U) = {U.det()}')
    """
    B = Matrix([[1, -1, -6, 1, 2],
                [2, -1, -1, -2, 3],
                [3, -1, 4, -5, 6]])
    b = Matrix([4, 5, 6])
    pprint(B)
    pprint(b)
    x = solution_space(B, b)
    pprint(x)
    """

if __name__ == '__main__':
    main()
