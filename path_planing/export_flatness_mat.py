"""Apply differential flatness to fig8 and loop PMM trajectories; export .mat for MATLAB animation."""
import os
import numpy as np
from scipy.io import savemat
from reference_conversion import flat_map_trajectory

script = os.path.dirname(os.path.abspath(__file__))

CIRCUITS = [
    ('fig8', ''),         # default suffix -> xref_optimo_3D_PMM.npy
    ('loop', '_loop'),
]

for name, suf in CIRCUITS:
    X = np.load(os.path.join(script, f'xref_optimo_3D_PMM{suf}.npy'))
    U = np.load(os.path.join(script, f'uref_optimo_3D_PMM{suf}.npy'))
    t = np.load(os.path.join(script, f'tref_optimo_3D_PMM{suf}.npy'))

    ref = flat_map_trajectory(X, U, t)

    # Pre-compute body-axes in world frame at every sample:
    #   R(q) columns = [x_b | y_b | z_b]
    q = ref['q']
    N = q.shape[1]
    xb = np.zeros((3, N))
    yb = np.zeros((3, N))
    zb = np.zeros((3, N))
    for k in range(N):
        qw, qx, qy, qz = q[:, k]
        # Rotation from quaternion (Hamilton, qw first)
        R = np.array([
            [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),     1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
        ])
        xb[:, k] = R[:, 0]
        yb[:, k] = R[:, 1]
        zb[:, k] = R[:, 2]

    out = {
        't':     ref['t'],
        'p':     ref['p'],
        'v':     ref['v'],
        'q':     ref['q'],
        'q_att': ref['q_att'],
        'omega': ref['omega'],
        'T':     ref['T'],
        'yaw':   ref['yaw'],
        'xb':    xb,
        'yb':    yb,
        'zb':    zb,
    }
    fname = os.path.join(script, f'flatness_ref_{name}.mat')
    savemat(fname, out)
    print(f"Saved {fname}  |  N={N} samples, T={t[-1]:.2f}s")
    # sanity
    print(f"  q norms min/max: {np.linalg.norm(ref['q'], axis=0).min():.4f} / {np.linalg.norm(ref['q'], axis=0).max():.4f}")
    print(f"  zb[2] range: {zb[2].min():.3f} .. {zb[2].max():.3f}  (<0 => INVERTED)")
    print(f"  T range: {ref['T'].min():.2f} .. {ref['T'].max():.2f} N")
    print(f"  |omega| max: {np.abs(ref['omega']).max():.2f} rad/s")
