import torch
from ase.build import bulk
from ase import Atom, Atoms
import random, pickle
import numpy as np
from ase.formula import Formula

seed = 1234
random.seed(seed)
np.random.seed(seed)

train_ratio = 0.8
vali_ratio = 0.1

from ase.db import connect
db_ir = connect('qm9_ir_spectrum.db')
db_qm9 = connect('qm9.db')

rows_ir = list(db_ir.select())
rows_qm9 = list(db_qm9.select())
# row = db_ir.get(1)
# tgt_len = len(row.data.ir_spectrum[1])

print('db load finish...')

with open('qm9_id_boc.lst', 'rb') as fp:
    c = pickle.load(fp)

print('pickle load finish...')

extra_adding = 0 # due to the exception in ir.py

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_default_tensor_type(torch.DoubleTensor)

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_node, output_node):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_node, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_node)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x


def extract_descriptor(rows_input):
    global extra_adding
    print('extract descriptor start')
    bocs, targets = [], []
    for row in rows_input:
        #### check if molecular formula in ir.db is the same with qm9.db #####
        sym1 = 'C'
        sym2 = 'B'
        while(Formula(str(sym1)) != Formula(str(sym2))):
            if str(sym1) != 'C' and str(sym2) != 'B':
                print(row.id, sym1, sym2, 'An mismatch occur, extra id:', extra_adding)
                extra_adding += 1
            sym1 = row.toatoms().symbols
            sym2 = rows_qm9[row.id-1+extra_adding].toatoms().symbols

        #### read boc from pre_calculated boc.lst #########
        bocs.append(c[row.id-1+extra_adding][1])

        #### read ir targets from ir.db ##############
        targets.append(row.data.ir_spectrum[1])
    
    #### shuffle ####
    shfl = list(zip(bocs, targets))
    random.shuffle(shfl)
    bocs, targets = zip(*shfl)
    return bocs, targets

if __name__ == '__main__':
    # training dataset
    boc_lst, tgt_lst = extract_descriptor(rows_ir[:int(train_ratio * len(rows_ir))])
    # vali dataset 
    vali_boc_lst, vali_tgt_lst = extract_descriptor(rows_ir[int(train_ratio * len(rows_ir)):int((train_ratio + vali_ratio) * len(rows_ir))])
    # train
    model = Model(boc_lst[0].shape[0], tgt_lst[0].shape[0])

    criterion = torch.nn.MSELoss() # Defined loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Defined optimizer
    
    x_data = torch.from_numpy(np.array(boc_lst)).to('cpu')
    y_data = torch.from_numpy(np.array(tgt_lst)).to('cpu')
    x_data_vali = torch.from_numpy(np.array(vali_boc_lst)).to('cpu')
    y_data_vali = torch.from_numpy(np.array(vali_tgt_lst)).to('cpu')

    min_loss = 999
    # Training: forward, loss, backward, step
    # Training loop
    for epoch in range(1000):
        # Forward pass
        y_pred = model(x_data.double())
    
        # Compute loss
        loss = criterion(y_pred, y_data)

        # Forward pass vali
        y_pred_vali = model(x_data_vali.double())
    
        # Compute loss vali
        loss_vali = criterion(y_pred_vali, y_data_vali)

        print(epoch, loss.item(), loss_vali.item())

        if loss_vali.item() < min_loss:
            min_loss = loss_vali.item()
            torch.save(model.state_dict(), 'best_model.dict')
            print('model saved')
        # Zero gradients
        optimizer.zero_grad()
        # perform backward pass
        loss.backward()
        # update weights
        optimizer.step()
    model.load_state_dict(torch.load('best_model.dict'))
    # inference dataset
    boc_lst, tgt_lst = extract_descriptor(rows_ir[int((train_ratio+vali_ratio) * len(rows_ir)):])
    # inference
    x_data = torch.from_numpy(np.array(boc_lst)).to('cpu')
    y_data = torch.from_numpy(np.array(tgt_lst)).to('cpu')

    y_pred = model(x_data.double())
    print(y_data.shape, y_pred.shape)
    # check error
    err_sum = 0
    for i in range(y_data.shape[0]):
        err_sum += abs(y_data.detach().numpy()[i][0] - y_pred.detach().numpy()[i][0])
        print(y_data.shape[0], y_data.detach().numpy()[i][0], y_pred.detach().numpy()[i][0])
    print(err_sum/y_data.shape[0])

