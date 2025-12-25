from scipy.spatial.transform import Rotation as R
import numpy as np

q = np.array([0, 0, 0, 1])

r = R.from_quat(q)

ang = r.as_euler("zyx")

print(ang)

eu = [3.14, 0, 0]

r = R.from_euler("zyx", eu)

ang = r.as_quat()

print(ang)