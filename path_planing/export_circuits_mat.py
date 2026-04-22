"""Export the three PMM circuits to .mat files for MATLAB plotting."""
import os
import numpy as np
from scipy.io import savemat

D = os.path.dirname(os.path.abspath(__file__))

circuits = [
    dict(xref='xref_optimo_3D_PMM_sprint.npy',
         tref='tref_optimo_3D_PMM_sprint.npy',
         gates='gates_sprint.npz',
         out='circuit_sprint.mat'),
    dict(xref='xref_optimo_3D_PMM.npy',
         tref='tref_optimo_3D_PMM.npy',
         gates='gates.npz',
         out='circuit_fig8.mat'),
    dict(xref='xref_optimo_3D_PMM_helix.npy',
         tref='tref_optimo_3D_PMM_helix.npy',
         gates='gates_helix.npz',
         out='circuit_helix.mat'),
]

for c in circuits:
    X   = np.load(os.path.join(D, c['xref']))
    t   = np.load(os.path.join(D, c['tref']))
    cfg = np.load(os.path.join(D, c['gates']))
    vel = np.linalg.norm(X[3:6, :], axis=0)
    savemat(os.path.join(D, c['out']), {
        'px': X[0,:], 'py': X[1,:], 'pz': X[2,:],
        'vx': X[3,:], 'vy': X[4,:], 'vz': X[5,:],
        'vel': vel,
        't':  t,
        'T_opt':  float(t[-1]),
        'v_peak': float(vel.max()),
        'gate_positions': cfg['gate_positions'],
        'gate_normals':   cfg['gate_normals'],
        'gate_radius':    float(cfg['gate_radius']),
    })
    print(f"✓  {c['out']}")
