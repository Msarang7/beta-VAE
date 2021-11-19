from data import data
from betaVAE import betaVAE
import torch
from utils import train

########################################
# parameters

torch.manual_seed(7)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("gpu detected for training")
else :
    device = torch.device('cpu')
    print("cpu detected for training")

latent_dims = 10
batch_size = 32
beta = 4
nc = 3
epochs = 50
lr = 1e-4
img_h = 64
img_w = 64

###################################################
# data and model


dataset_name = '3dchairs' # dsprties, celebA, chairs
data_train = data(dataset_name, batch_size, img_h, img_w)

batch = iter(data_train).next()
print(batch.shape)

model = betaVAE(latent_dims, nc, beta, device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-5)
model.to(device)

##########################################################

for epoch in range(epochs):

    train_loss = train(model, data_train, optimizer, device, batch_size, epoch)
    print("epoch : " + str(epoch) + " , " + "loss : " + str(train_loss))






