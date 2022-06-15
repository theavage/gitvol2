import numpy as np

w1 = np.load('/zhome/df/9/164401/gitvol2/Colin/project2/stylegan2directions/eyes_open.npy')
w1 = np.expand_dims(w1, 0)

w2 = np.load('/zhome/df/9/164401/gitvol2/Colin/project2/stylegan2directions/mouth_open.npy')
w2 = np.expand_dims(w2, 0)

z = np.load('/zhome/df/9/164401/gitvol2/Colin/project2/interpolation_variables/projected_nico_w.npz')['w']

magnitude1 = - 40
magnitude2 = 50

z_new = z + w1 * magnitude1 + w2 * magnitude2

savename = '/zhome/df/9/164401/gitvol2/Colin/project2/interpolation_variables/eyes_open.npz'
np.savez(savename, w = z_new.astype(np.float32))