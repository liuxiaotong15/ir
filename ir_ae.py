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


if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')
 
 
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 28, 28)
    return x
 
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

ir_data_tensor = torch.tensor(ir_data)
print(ir_data_tensor[:batch_size].shape[0])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

# dataset = MNIST('./data', transform=img_transform)
# dataset = MNIST('./data', transform=transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

 
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
 
class autoencoder_ir(nn.Module):
    def __init__(self, input_size):
        super(autoencoder_ir, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Linear(2048, input_size),
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder_ir(ir_data_tensor.shape[1]).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    for i in range(ir_data_tensor.shape[0]//batch_size):
        batch_data = ir_data_tensor[i*batch_size:(i+1)*batch_size]
        batch_data = Variable(batch_data).cuda()
        # ===================forward=====================
        output = model(batch_data.float())
        loss = criterion(output, batch_data.float())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data))
          # .format(epoch+1, num_epochs, loss.data[0]))
    # if epoch % 10 == 0:
    #     pic = to_img(output.cpu().data)
    #     save_image(pic, './dc_img/image_{}.png'.format(epoch))
 
torch.save(model.state_dict(), './conv_autoencoder.pth')
