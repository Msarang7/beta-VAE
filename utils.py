import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def train(betaVAE, dataloader, optimizer, device, batch_size, epoch):

    betaVAE.train()
    loss_avg_epoch = 0
    i = 0 # batch index
    for x in dataloader:

        x = x.to(device)
        x_hat = betaVAE(x)
        loss = ((x_hat - x)**2).sum() + betaVAE.encoder.beta_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("epoch : " + str(epoch)  + ", avergae loss of batch " + str(i) + "/" + str(len(dataloader))  + " loss : "  + str(float(loss.item() /  batch_size)) )
        loss_avg_epoch += loss.item()
        i = i+1

    return loss_avg_epoch / len(dataloader.dataset)

def eval(betaVAE, dataloader, device, batch_size):

    betaVAE.eval()
    val_avg_loss = 0

    with torch.no_grad():

        for x in datalaoder : # x comprises a single batch
            x = x.to(device)
            x_hat = betaVAE(x)
            loss = ((x - x_hat)**2).sum() + betaVAE.encoder.beta_kl
            val_avg_loss += loss
        return val_avg_loss / len(dataloader.dataset)








