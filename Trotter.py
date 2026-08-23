import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from scipy.linalg import expm

matplotlib.use("Agg")

N_QUBITS = 4
TIME = 1.0
STEPS = np.array([4, 8, 16, 32, 64])

WIRES = list(range(N_QUBITS))
DIM = 2**N_QUBITS


def nn_sector(pauli):
    return qml.sum(
        *[
            qml.prod(pauli(i), pauli(i + 1))
            for i in range(N_QUBITS - 1)
        ]
    )


def dense(operator):
    return qml.matrix(operator, wire_order=WIRES)


def spectral_norm(matrix):
    return np.linalg.norm(matrix, 2)


def commutator(a, b):
    return a @ b - b @ a


def power_law_slope(steps, errors):
    return np.polyfit(np.log(steps), np.log(errors), 1)[0]


SECTORS = [
    nn_sector(qml.X),
    nn_sector(qml.Y),
    nn_sector(qml.Z),
]

GENERATORS = [dense(sector) for sector in SECTORS]
H_X, H_Y, H_Z = GENERATORS
H = H_X + H_Y + H_Z

V0 = (
    commutator(H_Y, H_X)
    + commutator(H_Z, H_X)
    + commutator(H_Z, H_Y)
)

V0_NORM = spectral_norm(V0)

C0 = dense(
    qml.prod(
        *[qml.Hadamard(i) for i in range(N_QUBITS)]
    )
)

V0_BAR = 0.5 * (V0 + C0.conj().T @ V0 @ C0)


def trotter_step(dt):
    step = np.eye(DIM, dtype=complex)

    for sector in SECTORS:
        step = dense(qml.exp(sector, -1j * dt)) @ step

    return step


def evolve(steps, protected=False):
    step = trotter_step(TIME / steps)
    evolution = np.eye(DIM, dtype=complex)
    symmetry = np.eye(DIM, dtype=complex)

    for _ in range(steps):
        if protected:
            symmetry = C0 @ symmetry
            evolution = (
                symmetry.conj().T
                @ step
                @ symmetry
                @ evolution
            )
        else:
            evolution = step @ evolution

    return evolution


def compute_errors():
    exact = expm(-1j * TIME * H)

    raw_errors = []
    protected_errors = []
    bounds = []

    for steps in STEPS:
        raw_errors.append(
            spectral_norm(exact - evolve(steps))
        )
        protected_errors.append(
            spectral_norm(exact - evolve(steps, protected=True))
        )
        bounds.append(
            V0_NORM * TIME**2 / (2 * steps)
        )

    return (
        np.array(raw_errors),
        np.array(protected_errors),
        np.array(bounds),
    )


def print_results(raw_errors, protected_errors, bounds):
    print(f"{N_QUBITS}-qubit Heisenberg chain, t = {TIME}\n")
    print(f"||v0||                    = {V0_NORM:.6f}")
    print(
        f"||[C0, H]||               = "
        f"{spectral_norm(commutator(C0, H)):.2e}"
    )
    print(
        f"||v0bar||                 = "
        f"{spectral_norm(V0_BAR):.2e}\n"
    )

    print(
        f"{'r':>4} "
        f"{'raw error':>12} "
        f"{'bound':>12} "
        f"{'ratio':>8} "
        f"{'protected':>12}"
    )

    for steps, raw, bound, protected in zip(
        STEPS,
        raw_errors,
        bounds,
        protected_errors,
    ):
        print(
            f"{steps:>4} "
            f"{raw:>12.6f} "
            f"{bound:>12.6f} "
            f"{raw / bound:>8.3f} "
            f"{protected:>12.6f}"
        )

    raw_slope = power_law_slope(STEPS, raw_errors)
    protected_slope = power_law_slope(
        STEPS,
        protected_errors,
    )

    print(f"\nraw slope       = {raw_slope:+.3f}")
    print(f"protected slope = {protected_slope:+.3f}")


def plot_results(raw_errors, protected_errors, bounds):
    raw_slope = power_law_slope(STEPS, raw_errors)
    protected_slope = power_law_slope(
        STEPS,
        protected_errors,
    )

    fig, ax = plt.subplots(figsize=(6, 4.2))

    ax.loglog(
        STEPS,
        raw_errors,
        "o-",
        label=f"ordered product ({raw_slope:+.2f})",
    )
    ax.loglog(
        STEPS,
        bounds,
        "k--",
        linewidth=1,
        label=r"$\|v_0\|t^2/(2r)$",
    )
    ax.loglog(
        STEPS,
        protected_errors,
        "s-",
        label=f"symmetry protected ({protected_slope:+.2f})",
    )

    ax.set_xlabel("Trotter steps $r$")
    ax.set_ylabel(r"$\|e^{-iHt}-U_r\|_2$")
    ax.set_title(
        f"Ordering error vs. exact evolution ({N_QUBITS} qubits)"
    )
    ax.set_xticks(STEPS)
    ax.set_xticklabels(STEPS.astype(str))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig("scaling.png", dpi=160)


def main():
    raw_errors, protected_errors, bounds = compute_errors()
    print_results(raw_errors, protected_errors, bounds)
    plot_results(raw_errors, protected_errors, bounds)


if __name__ == "__main__":
    main()
