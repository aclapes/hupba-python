import numpy as np

def camera_matrix_to_intrinsics(K):
  '''

  If camera matrix is a 3x3 matrix where:
        | fx  0  cx |
    K = | 0  fy  cy |
        | 0   0   1 |

  Then the function returns:

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

  :param K: 3x3 camera matrix
  :return: (fx, fy), (cx, cy)
  '''
  return (K[0,0], K[1,1]), (K[0,2], K[1,2])


def align_to_depth(D, Kd, Ko, scale_d, R, t):
  '''
  Algin some other modality to depth.
  :param D: depth frame
  :param Kd: depth's modality camera matrix
  :param Ko: other's modality camera matrix
  :param scale_d: a scaling factor to convert depth values into meters
  :param R: other-to-depth rotation matrix
  :param t: other-to-depth translation vector
  :return: map_x and map_y that can be used in OpenCV's cv2.remap(...)
  '''
  (fx_d, fy_d), (cx_d, cy_d) = camera_matrix_to_intrinsics(Kd.astype(np.float32))
  (fx_o, fy_o), (cx_o, cy_o) = camera_matrix_to_intrinsics(Ko.astype(np.float32))

  i = np.repeat(np.linspace(0, D.shape[0]-1, D.shape[0], dtype=np.float32), D.shape[1])
  j = np.tile(np.linspace(0, D.shape[1]-1, D.shape[1], dtype=np.float32), D.shape[0])
  d = np.reshape(D, [np.prod(D.shape),]).astype(np.float32)

  z = d * scale_d
  x = ((j - cx_d) * z) / fx_d
  y = ((i - cy_d) * z) / fy_d

  P = np.concatenate([x[np.newaxis,:], y[np.newaxis,:], z[np.newaxis,:]], axis=0)

  P_t = np.matmul(R.astype(np.float32), P) + t.astype(np.float32)

  map_x = np.reshape(P_t[0, :] * fx_o / P_t[2, :] + cx_o, D.shape)
  map_y = np.reshape(P_t[1, :] * fy_o / P_t[2, :] + cy_o, D.shape)

  return map_x, map_y