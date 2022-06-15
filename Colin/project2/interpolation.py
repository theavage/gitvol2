import numpy as np

z1 = np.load('/zhome/df/9/164401/gitvol2/Colin/project2/interpolation_variables/projected_nico_w.npz')['w']
z2 = np.load('/zhome/df/9/164401/gitvol2/Colin/project2/interpolation_variables/projected_colin_w.npz')['w']

step = 0.01
steps = np.arange(0, 1 + step, step)
s = 0.75

result = s * z1 + (1 - s) * z2
savename = '/zhome/df/9/164401/gitvol2/Colin/project2/interpolation_variables/inter075.npz'
np.savez(savename, w = result)