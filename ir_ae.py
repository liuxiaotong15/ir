import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os


from ase.db import connect

db = connect('qm9_ir_spectrum.db')

rows_lst = list(db.select())
print(len(rows_lst))

ir_data = []
for row in rows_lst:
    ir_data.append(row.data.ir_spectrum[1])

# row = db.get(1)
# print(row.toatoms().symbols)
# 
# print(row.data.ir_spectrum[0])
# print(len(row.data.ir_spectrum[1]))
# 
# for i,j  in zip(row.data.ir_spectrum[0], row.data.ir_spectrum[1]):
#     if j > 0.1:
#         print(i, j)

 
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

ir_data_tensor = torch.tensor(ir_data)
print(ir_data_tensor[:batch_size].shape[0])
max_ir = torch.max(ir_data_tensor, 1)[0]

print(ir_data_tensor.shape)
print(max_ir.view(-1,1).shape)

print(torch.div(ir_data_tensor, max_ir.view(-1, 1)))

print(torch.max(torch.div(ir_data_tensor, max_ir.view(-1, 1)), 1)[0])
ir_data_tensor = torch.div(ir_data_tensor, max_ir.view(-1, 1))

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
 
class autoencoder_ir(nn.Module):
    def __init__(self, input_size):
        super(autoencoder_ir, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            # nn.ReLU(True),
            # nn.Linear(2048, 2048),
            # nn.ReLU(True),
            # nn.Linear(2048, 1024),
        )
        self.decoder = nn.Sequential(
            # nn.Linear(1024, 2048),
            # nn.ReLU(True),
            # nn.Linear(2048, 2048),
            # nn.ReLU(True),
            nn.Linear(2048, input_size),
            # nn.functional.softmax(True)
            nn.ReLU(True),
            # nn.Sigmoid(),
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder_ir(ir_data_tensor.shape[1]).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

l1_lambda = 1e-2

for epoch in range(num_epochs):
    for i in range(ir_data_tensor.shape[0]//batch_size):
        batch_data = ir_data_tensor[i*batch_size:(i+1)*batch_size]
        batch_data = Variable(batch_data).cuda()
        # ===================forward=====================
        output = model(batch_data.float())
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(abs(param))
        loss = criterion(output, batch_data.float())
        loss_total = loss + l1_lambda * regularization_loss
        # ===================backward====================
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
    # ===================log========================
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(abs(param))
    
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data))
    print(torch.max(batch_data, 1)[0])
    print(torch.max(output, 1)[0])
          # .format(epoch+1, num_epochs, loss.data[0]))
    # if epoch % 10 == 0:
    #     pic = to_img(output.cpu().data)
    #     save_image(pic, './dc_img/image_{}.png'.format(epoch))
 
torch.save(model.state_dict(), './conv_autoencoder.pth')
