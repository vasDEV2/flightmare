import onnxruntime as ort
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
# Load the ONNX model
sess = ort.InferenceSession("/home/vasudevan/Desktop/flightmare/flightrl/examples/a_new_hope.onnx")

# Get input name (usually 'input/Ob:0')
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape

print(input_name)

# # Create dummy input (Match the shape of your observation)
# # Note: SB1 usually expects [Batch_Size, Obs_Dim]
# # obs = np.random.randn(1, *input_shape[1:]).astype(np.float32)
# obs = np.array([[ 0.20710441,  0.17971317,  0.17334747,  0.31125432,  0.01186327,  0.01035172,
#   -0.10929969, -0.04562206,  0.02839059, -0.04129951,  0.01239144, -0.022663  ]]).astype(np.float32)

# obs = np.array([[-5.0002441e+00, -5.0000086e+00,  4.7486296e+00,  9.0159290e-04,
#    7.3797055e-02, -1.2190812e-02,  2.3667186e-02,  1.4997749e-03,
#    2.8817841e-01, -1.1536789e+00,  1.0295414e+00,  9.8818228e-02,]]
# ).astype(np.float32)

# obs = np.array([[0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.float32)
# euler = [0, 0, 0]

# angst = np.array([[8.8564354e-01, 4.6436411e-01,  1.1299541e-03,], [-4.6436414e-01,  8.8564420e-01,
#   -2.6648317e-04], [-1.1244826e-03, -2.8870156e-04, 9.9999934e-01]])

# r = R.from_matrix(angst)
# angu = r.as_euler("zyx")

# print(f"ANGU: {angu}")

# r = R.from_euler("zyx", euler)

# rotation_matrix = r.as_matrix()

# rotation_matrix = rotation_matrix.reshape(-1)
# # rotation_matrix = np.expand_dims(rotation_matrix, axis=0)
# # print(rotation_matri)
# # obs = np.zeros((1, 18))
# # # obs = np.array([[ 3.4256729e-01, -1.5300065e-01,  3.9916039e-03,  8.8564354e-01,
# # #    4.6436411e-01,  1.1299541e-03, -4.6436414e-01,  8.8564420e-01,
# # #   -2.6648317e-04, -1.1244826e-03, -2.8870156e-04,  9.9999934e-01,
# # #    1.6551508e-02,  2.9529743e-03, -2.1576297e-03, -3.9716638e-04,
# # #   -8.4366265e-04,  1.2204273e-04,]])
# # obs = np.array([[ 2.5225699e-01, -1.4624867e-01,  2.5234222e-03,  8.8733983e-01,
# #    4.6099454e-01, -1.0603435e-02, -4.6102944e-01,  8.8738453e-01,
# #   -9.7770966e-04,  8.9586079e-03,  5.7560480e-03,  9.9994332e-01,
# #   -1.3116394e-01, -2.3874214e-02,  1.5861075e-02,  1.1019330e-02,
# #    2.1925174e-04, -5.1474068e-03,]]
# # )
# # # obs[:, 3:12] = rotation_matrix
# # obs = obs.astype(np.float32)
# # obs[:, 1] = 0.0
# # obs[:, 0] = 0.0
# # Run inference 

# obs = np.zeros((1, 18))

# # obs[:, 0] = 3.0
# obs[:, 2] = 4.9 
# obs[:, 3:12] = rotation_matrix
# obs = obs.astype(np.float32)
# # print(obs)
# e = [ 0.1, 2.25231, -3.19519942e-04]
# r = R.from_euler("zyx", e)
# mt = r.as_matrix()
# quats = r.as_quat()
# print("yo", quats)
# mt = np.reshape(mt, (9,), "F")
# obs = np.array([[3.0009408e-02,  5.6159822e-03,  4.0063171e+00, -1.0267395e-01,
#   -9.9471503e-01, -1.7978862e-04,  9.9471498e-01, -1.0267400e-01,
#    3.4033493e-04, -3.5699591e-04, -1.4389491e-04,  9.9999994e-01,
#    1.6636505e-03,  4.0779137e-03,  2.5733274e-03, -1.3716160e-05,
#    2.5522301e-04,  4.7263943e-04]]).astype(np.float32)

# obs[0, 3:12] = mt

# print(obs)

act = 9.81*2*np.ones(4)
std = 9.81*2*0.1

t = 9.81

obs = np.zeros((1, 18))
# rt = [0, 0, 0, 1]

# rr = R.from_quat(rt)
# k = (rr.as_euler("zyx"))
# print(f"eu : {k}")


e = [1.67, 0.0, 0.0]

r = R.from_euler("zyx", e)
print(f"quats: {r.as_quat()}")
rot = r.as_matrix()

mmk = np.array([-9.8691225e-02,
   9.9511784e-01, -2.4673686e-04, -9.9511784e-01, -9.8691344e-02,
  -2.4357777e-04, -2.6673946e-04,  2.2149325e-04,  9.9999994e-01
])
print(mmk.shape)
mm = np.reshape(mmk, (3, 3), 'F')
rrr = R.from_matrix(mm)
print(rrr.as_euler("zyx"))
print(rrr.as_quat())

rot = np.reshape(rot, (9,), 'F')

obs[:, 3:12] = mmk
obs[:, 2] = 0.489
obs = obs.astype(np.float32)

# obs = np.array( [[-0.00881353, -0.00165005,  0.4031011,  0.76482695,  0.64374286,  0.02520664,
#   -0.64420325,  0.76380676,  0.04002332,  0.00651169, -0.04684911,  0.99888074,
#   -0.00128235,  0.00889989, -0.00256583,  0.02129127,  0.06160183,  0.01381842]]).astype(np.float32)

obs = np.array([[-1.5291860e-03,  5.0087718e-05,  4.8961139e-01, -9.8691225e-02,
   9.9511784e-01, -2.4673686e-04, -9.9511784e-01, -9.8691344e-02,
  -2.4357777e-04, -2.6673946e-04,  2.2149325e-04,  9.9999994e-01,
   7.6458886e-02, -2.5040468e-03,  7.7533566e-02, -4.1517960e-02,
   5.2739505e-02, -5.2312765e-02]]).astype(np.float32)

obs[:, 0] = 0.0
obs[:, 1] = 0.0
# obs[:, 12] = 0.0
obs[:, 13] = 0.0
obs[:, 14] = 0.0
# obs[:, 15] = 0.0
# obs[:, 15] = 0.0

# rm = obs[0, 3:12]

# rm = np.sresh 

result = sess.run(None, {input_name: obs})
result = torch.tensor(result)
result = torch.tanh(result) 
print(f"OBSERVATION = {obs}")
# 'result' is a list containing the outputs (actions)
print("Action:", result[0]) 

result[0, 0, 0] = result[0, 0, 0] + t
print(result)
result[0, 0, 0] = result[0, 0, 0]/(1.35*9.81)



print(f"Thrust: {result}")