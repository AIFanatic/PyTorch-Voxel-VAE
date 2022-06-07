import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from model import VAE
from ShapeNetChairs import ShapeNetChairs

learning_rate = 0.005
momentum = 0.9
batch_size = 10
epoch_num = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_train = ShapeNetChairs('datasets/shapenet10_chairs_nr.tar')
train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

for epoch in range(epoch_num):
    model.train(True)
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_dataloader)):
        inputs = data.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        
        loss = model.loss(inputs, outputs)

        loss.backward()
        optimizer.step()

        running_loss = loss.item()
    
    print("Epoch", epoch, "Loss", running_loss)

print('Finished Training')

torch.save(model.state_dict(), "./models/vae.pt")