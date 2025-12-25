import numpy as np

# =========================
# Quadrotor parameters
# =========================
l = 0.2        # arm length [m]
kappa = 0.0001 # yaw torque coefficient

# =========================
# Desired wrench
# =========================
T = 1.13016
tau_x = -0.0148325
tau_y = -0.0088076
tau_z = 0.0071

wrench = np.array([T, tau_x, tau_y, tau_z])

# =========================
# Allocation matrix (X config)
# =========================
a = l / np.sqrt(2)

A = np.array([
    [ 1.0,  1.0,  1.0,  1.0],
    [ a,   -a,   -a,    a ],
    [ a,    a,   -a,   -a ],
    [-kappa, kappa, -kappa, kappa]
])
# Order must match Flightmare motor indexing
motor_pos = np.array([
    [ l/np.sqrt(2),  l/np.sqrt(2), 0.0],   # Motor 1
    [-l/np.sqrt(2),  l/np.sqrt(2), 0.0],   # Motor 2
    [-l/np.sqrt(2), -l/np.sqrt(2), 0.0],   # Motor 3
    [ l/np.sqrt(2), -l/np.sqrt(2), 0.0],   # Motor 4
])

# Torque = r × f, thrust along +Z
t_BM = np.zeros((3, 4))
for i in range(4):
    r = motor_pos[i]
    f = np.array([0, 0, 1])   # thrust direction
    t_BM[:, i] = np.cross(r, f)

# Allocation matrix (EXACT Flightmare structure)
A = np.vstack([
    np.ones(4),
    t_BM[:2, :],   # topRows<2>()
    kappa * np.array([1, -1, 1, -1])
])



# =========================
# Solve for motor thrusts
# =========================
# motor_thrusts = np.linalg.solve(A, wrench)
A_inv = np.linalg.inv(A)

print("Allocation matrix A:\n", A_inv)

A_inv = np.array( [[0.25,  1.76777, -1.76777,   15.625],
    [0.25, -1.76777, -1.76777,  -15.625],
    [0.25, -1.76777,  1.76777,  15.625],
    [0.25,  1.76777,  1.76777, -15.625]]
)



L = 0.5
K = 5

M = np.array([[ 1.0/4, -1.0/(4*L), 1.0/(4*L), -1.0/(4*K)],
    [1.0/4, 1.0/(4*L), -1.0/(4*L), -1.0/(4*K)],
    [1.0/4, 1.0/(4*L), 1.0/(4*L), 1.0/(4*K)],
    [1.0/4, -1.0/(4*L), -1.0/(4*L), 1.0/(4*K)]])

m_inv = np.linalg.inv(M)

motor_thrusts = M @ wrench
# =========================
# Print results
# =========================
for i, f in enumerate(motor_thrusts, start=1):
    print(f"Motor {i} thrust: {f:.4f} N")
