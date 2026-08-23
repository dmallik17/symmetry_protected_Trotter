# symmetry_protected_Trotter
# Symmetry-Protected Trotter Error

A small numerical study of ordering error in first-order Trotterization of a nearest-neighbor Heisenberg chain.

The Hamiltonian is split into Pauli sectors

\[
H = H_X + H_Y + H_Z,
\qquad
H_P = \sum_i P_i P_{i+1},
\]

and the ordered product

\[
S(\Delta t)
=
e^{-iH_Z\Delta t}
e^{-iH_Y\Delta t}
e^{-iH_X\Delta t}
\]

is compared with the exact evolution \(e^{-iHt}\).

The leading Trotter error is governed by

\[
v_0
=
[H_Y,H_X]
+
[H_Z,H_X]
+
[H_Z,H_Y],
\]

giving the expected \(O(r^{-1})\) operator-norm scaling.

For the isotropic Heisenberg model, the global Hadamard symmetry

\[
C_0 = H^{\otimes n}
\]

satisfies

\[
[C_0,H]=0,
\qquad
C_0^\dagger v_0 C_0=-v_0.
\]

Alternating conjugation therefore cancels the leading commutator contribution. Numerically, the protected evolution exhibits approximately \(O(r^{-2})\) global-error scaling.

For four qubits at \(t=1\), the fitted slopes are approximately

\[
-1.00 \quad \text{(unprotected)}, \qquad
-2.04 \quad \text{(symmetry protected)}.
\]

## Run

```bash
pip install numpy scipy matplotlib pennylane
python trotter_commutator.py
