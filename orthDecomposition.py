import numpy as np

# Gram-Schmidt – take a bunch of linearly independent vectors and spit out
# an orthonormal basis for the same subspace.
#
# How it works:
#   - keep the first vector as-is (just normalize it)
#   - for every next vector, strip out the parts that point along the
#     vectors we already have, then normalize what's left
#   - repeat until done
#
# The "modified" version just re-projects the running residual each time
# instead of doing everything from the original vector. Numerically nicer.

def gram_schmidt(vectors, normalize=True):
    orthogonal = []

    for b_k in vectors:
        u_k = b_k.astype(float).copy()

        for u_j in orthogonal:
            # how much of u_j is hiding inside u_k? subtract it out
            u_k -= (np.dot(u_k, u_j) / np.dot(u_j, u_j)) * u_j

        norm = np.linalg.norm(u_k)
        if norm < 1e-10:
            raise ValueError("vectors are linearly dependent, can't build a basis")

        orthogonal.append(u_k / norm if normalize else u_k)

    return orthogonal


# Given a vector x and some basis vectors (don't need to be orthogonal),
# split x into:
#   x_U      – the piece that lives inside span(basis)
#   x_U_perp – the leftover piece that's perpendicular to everything in span(basis)
#
# x = x_U + x_U_perp   always, exactly.

def orthogonal_decomposition(x, basis):
    onb = gram_schmidt(basis, normalize=True)   # make a proper ONB first

    x = x.astype(float)
    x_U = np.zeros_like(x)
    for u_i in onb:
        x_U += np.dot(x, u_i) * u_i            # project onto each basis vector

    return x_U, x - x_U                        # complement is just what's left


if __name__ == "__main__":

    # --- test 1: basic gram-schmidt on 3 vectors in R^3 ---
    print("Test 1: Gram-Schmidt in R^3\n")

    b1 = np.array([1, 1, 0], dtype=float)
    b2 = np.array([1, 0, 1], dtype=float)
    b3 = np.array([0, 1, 1], dtype=float)

    onb = gram_schmidt([b1, b2, b3])

    for i, u in enumerate(onb, 1):
        print(f"  u{i} = {np.round(u, 4)}")

    # gram matrix should be the identity if everything worked
    G = np.array([[np.dot(onb[i], onb[j]) for j in range(3)] for i in range(3)])
    print(f"\n  Gram matrix (want I):\n{np.round(G, 6)}\n")


    # --- test 2: decompose a vector against a non-orthogonal basis in R^4 ---
    print("Test 2: Orthogonal Decomposition in R^4\n")

    b1 = np.array([1, 1, 0, 0], dtype=float)
    b2 = np.array([1, 0, 1, 0], dtype=float)   # not orthogonal to b1
    x  = np.array([2, 3, 1, 4], dtype=float)

    x_U, x_perp = orthogonal_decomposition(x, [b1, b2])

    print(f"  x       = {x}")
    print(f"  x_U     = {np.round(x_U, 4)}")
    print(f"  x_perp  = {np.round(x_perp, 4)}")

    print(f"\n  reconstruction error  = {np.linalg.norm(x - x_U - x_perp):.2e}  (want 0)")
    print(f"  <x_U, x_perp>         = {abs(np.dot(x_U, x_perp)):.2e}           (want 0)")
    print(f"  <x_perp, b1>          = {abs(np.dot(x_perp, b1)):.2e}           (want 0)")
    print(f"  <x_perp, b2>          = {abs(np.dot(x_perp, b2)):.2e}           (want 0)\n")


    # --- test 3: project a 2d vector onto a line ---
    print("Test 3: Project v onto span{b} in R^2\n")

    b = np.array([1, 2], dtype=float)
    v = np.array([4, 1], dtype=float)

    v_U, v_perp = orthogonal_decomposition(v, [b])

    print(f"  v       = {v}")
    print(f"  v_U     = {np.round(v_U, 4)}")
    print(f"  v_perp  = {np.round(v_perp, 4)}")

    lhs = np.dot(v, v)
    rhs = np.dot(v_U, v_U) + np.dot(v_perp, v_perp)
    print(f"\n  Pythagoras: ||v||^2 = {lhs:.4f},  ||v_U||^2 + ||v_perp||^2 = {rhs:.4f}  ✓")