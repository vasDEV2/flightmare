import numpy as np

arm_l = 0.2
kappa = 0.016

t_BM = (arm_l * np.sqrt(0.5) *
            np.array([
                [ 1, -1, -1,  1],
                [-1, -1,  1,  1],
                [ 0,  0,  0,  0]
            ])
        )

allocation_matrix = np.vstack([ np.ones(4),
                                t_BM[:2, :],
                                kappa * np.array([1, -1, 1, -1])
                            ])

inverse_allocat = np.linalg.inv(allocation_matrix)


thrust = np.array([17.0972, 0.0, 0.0, 0.05])

all = inverse_allocat @ thrust

print(all)