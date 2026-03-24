import numpy as np


def determinant(A):

    A = np.array(A, dtype=float)
    n, m = A.shape
    

    # ── base cases ──────────────────────────────────────────────────
    if n == 1:
        return A[0, 0]

    # ── (ad - bc) ─────────────────────────────────────────────────────
    if n == 2:
    
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    # ── cofactor expansion along the first row (n ≥ 3) ────
    det = 0.0
    for j in range(n):
        # minor: delete row 0 and column j
        minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
        sign  = (-1) ** j
        det  += sign * A[0, j] * determinant(minor)

    return det


def trace(A):
    """
    Compute the trace of a square matrix — the sum of diagonal entries.

    tr(A) = Σᵢ aᵢᵢ

    Properties:
      - tr(A + B) = tr(A) + tr(B)
      - tr(AB)    = tr(BA)
      - tr(Aᵀ)   = tr(A)
      - tr(A)     = sum of eigenvalues of A
    """
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square.")
    return sum(A[i, i] for i in range(n))


# ── demo ─────────────────────────────────────────────────────────────────────
# ── compares to numpy np.trace() and np.linalg.det() ─────────────────────────

if __name__ == "__main__":

    examples = [
        ("2×2", [[3, 8],
                 [4, 6]]),

        ("3×3", [[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 10]]),

        ("4×4", [[2, -1,  0,  3],
                 [1,  4, -2,  1],
                 [0,  2,  5, -1],
                 [3, -1,  2,  6]]),

        ("Singular 3×3", [[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]]),   # det = 0
    ]

    for label, M in examples:
        our_det   = determinant(M)
        numpy_det = np.linalg.det(M)
        our_tr    = trace(M)
        numpy_tr  = np.trace(M)

        print(f"\n{label} matrix:")
        print(np.array(M))
        print(f"  det  — ours: {our_det:>12.4f}   numpy: {numpy_det:>12.4f}")
        print(f"  trace— ours: {our_tr:>12.4f}   numpy: {numpy_tr:>12.4f}")