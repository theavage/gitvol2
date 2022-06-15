import numpy as np
from sklearn import svm
import glob

X = np.empty((0, 18 * 512))
for filename in glob.glob('/zhome/df/9/164401/gitvol2/Colin/project2/latentvariables/own/*.npz'):
    temp = np.load(filename)['w']
    temp = temp.flatten()
    X = np.vstack([X, np.atleast_2d(temp)])

y = np.squeeze(np.hstack([np.ones((1, 10)), np.zeros((1, 10))]))

model = svm.LinearSVC()
model.fit(X, y)
latent_direction = model.coef_

w = np.load('/zhome/df/9/164401/gitvol2/Colin/project2/latentvariables/nico.npz')['w']

magnitude = 3000
w_new = w + latent_direction.reshape(w.shape) * magnitude

np.savez('/zhome/df/9/164401/gitvol2/Colin/project2/latentvariables/glasses.npz', w = w_new)