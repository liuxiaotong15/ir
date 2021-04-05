from ase.db import connect
from sklearn.decomposition import PCA
import random
import numpy as np

db = connect('qm9_ir_spectrum.db')

rows_lst = list(db.select())
print(len(rows_lst))

ir_data = []
for row in rows_lst:
    dt = np.array(row.data.ir_spectrum[1])
    ir_data.append(dt/np.sum(dt))

random.shuffle(ir_data)

ir_data = np.array(ir_data)
print(ir_data.shape)

min_loss = 1e50
min_comp = 0

orig_l1 = np.sum(np.abs(ir_data[2000:]))/ir_data.shape[1]
print(orig_l1)

for comp in range(5, 2000):
    # PCA by Scikit-learn
    pca = PCA(n_components=comp) # n_components can be integer or float in (0,1)
    pca.fit(ir_data[0:2000]) # fit the model
    
    # print(pca.components_)
    # new_mat = pca.fit_transform(ir_data[2000:])
    new_mat = pca.transform(ir_data[2000:])
    
    inv_pca = np.matrix(pca.components_)
    
    # print("mean in pca:", pca.mean_)
    # print('\nMethod 3: PCA by Scikit-learn:')
    # print('After PCA transformation, data becomes:')
    # print(pca.fit_transform(ir_data)[0:5]) # transformed data
    # print(new_mat * inv_pca + pca.mean_)
    rcov_mat = new_mat * inv_pca + pca.mean_
    loss = np.sum(np.abs(ir_data[2000:] - rcov_mat))
    latent_l1 = np.sum(np.abs(new_mat))/comp # bigger is better, because we don't need 0 in latent space

    # pca.explained_variance is the eigen value of the matrix
    print(comp, loss, pca.explained_variance_ratio_.sum(), latent_l1)
