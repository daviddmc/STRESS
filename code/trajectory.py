import numpy as np
import scipy.io as sio
import os
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

#kp = ['ankle_l', 'ankle_r', 'knee_l', 'knee_r', 'bladder', 'elbow_l', 'elbow_r',
# 'eye_l', 'eye_r', 'hip_l', 'hip_r', 'shoulder_l', 'shoulder_r', 'wrist_l', 'wrist_r']

def get_trajectory(folder='../trajectory'):

    traj = []
    
    for f in os.listdir(folder):
        joint_coord = sio.loadmat(os.path.join(folder, f))['joint_coord'].astype(np.float32)
        
        joint_coord = joint_coord[np.all(joint_coord > 0, (1, 2))]
    
        eye_l = joint_coord[..., 7]
        eye_r = joint_coord[..., 8]
        neck = (joint_coord[..., 11] + joint_coord[..., 12]) / 2
        
        origin = (eye_l + eye_r + neck) / 3
        
        x_vec = eye_l - eye_r
        x_vec = x_vec / np.linalg.norm(x_vec, ord=2, axis=-1, keepdims=True)
        
        neck_eye_l = neck - eye_l
        y_vec = np.cross(x_vec, neck_eye_l)
        y_vec = y_vec / np.linalg.norm(y_vec, ord=2, axis=-1, keepdims=True)
        
        z_vec = np.cross(x_vec, y_vec)
        z_vec = z_vec / np.linalg.norm(z_vec, ord=2, axis=-1, keepdims=True)
        
        R = np.stack([x_vec, y_vec, z_vec], -1)
        R = R @ R[0].T[None]
        R = Rotation.from_matrix(R).as_euler('xyz')
        t = origin - origin[[0]]
        Rt = np.concatenate([R, t], -1)
        Rt = Rt[::2]  ####
        Rt = gaussian_filter1d(Rt, 0.5, 0) ######
        
        interp_func = interp1d(np.arange(Rt.shape[0]), Rt, kind='cubic', axis=0, fill_value="extrapolate", assume_sorted=True)
        
        traj.append((interp_func, Rt.shape[0]-1))
        
    return traj
    
if __name__ == '__main__':
    pass
