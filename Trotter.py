            frame = np.linalg.matrix_power(C, k % 3)
            propagator = frame.conj().T @ s @ frame @ propagator
        else:
            propagator = s @ propagator
    return propagator


def main():
    exact = expm(-1j*TIME*H)
    raw = np.array([np.linalg.norm(exact - evolve(int(r), False), 2) for r in STEPS])
    protected = np.array([np.linalg.norm(exact - evolve(int(r), True), 2) for r in STEPS])
    print('4-qubit XX+YY open spin chain; time 1')
    print('||C^dagger H C - H|| =', np.linalg.norm(C.conj().T@H@C-H, 2))
    print('||V0|| =', np.linalg.norm(V0, 2))
    print('||three-frame average V0|| =', np.linalg.norm(V0_average, 2))
    print(' r        raw           protected')
    for r, e0, e1 in zip(STEPS, raw, protected):
        print(f'{r:3d}  {e0:14.9f}  {e1:14.9f}')
    slope_raw = np.polyfit(np.log(STEPS), np.log(raw), 1)[0]
    slope_protected = np.polyfit(np.log(STEPS), np.log(protected), 1)[0]
    print(f'slopes: raw={slope_raw:.5f}, protected={slope_protected:.5f}')

    # Palindromic / self-adjoint compositions have M(-h)M(h) = I.
    def macro(h):
        s = step(h)
        result = np.eye(DIM, dtype=complex)
        for k in range(3):
            frame = np.linalg.matrix_power(C, k)
            result = (frame.conj().T @ s @ frame) @ result
        return result

    test_h = 0.1
    reversal_defect = np.linalg.norm(macro(-test_h) @ macro(test_h) - np.eye(DIM), 2)
    print(f'||M(-h)M(h)-I|| for h={test_h}: {reversal_defect:.9g}')

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.loglog(STEPS, raw, 'o-', label=f'ordinary (slope {slope_raw:.2f})')
    ax.loglog(STEPS, protected, 's-', label=f'threefold-protected (slope {slope_protected:.2f})')
    ax.set_xlabel('Number of steps r')
    ax.set_ylabel('Spectral-norm error')
    ax.set_title('XX+YY chain: threefold symmetry protection')
    ax.set_xticks(STEPS)
    ax.set_xticklabels([str(r) for r in STEPS])
    ax.minorticks_off()
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig('/mnt/data/threefold_xy_scaling.png', dpi=180)
    print('Plot: /mnt/data/threefold_xy_scaling.png')


if __name__ == '__main__':
    main()
