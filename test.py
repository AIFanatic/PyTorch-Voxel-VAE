import os
import torch
from torch.utils.data import DataLoader

from model import VAE
from ShapeNetChairs import ShapeNetChairs
from utils.save_volume import save_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
checkpoint = torch.load("./models/vae.pt")
model.load_state_dict(checkpoint)
model.eval()

data_train = ShapeNetChairs('datasets/shapenet10_chairs_nr.tar')
train_dataloader = DataLoader(data_train, batch_size=1, shuffle=False)

if not os.path.exists('reconstructions'):
    os.makedirs('reconstructions')

for i, data in enumerate(train_dataloader):
    sample = data.to(device)

    reconstructions = model(sample)

    reconstructions[reconstructions > 0] = 1
    reconstructions[reconstructions < 0] = 0

    reconstructions = reconstructions.detach().cpu()

    save_output(reconstructions[0][0], 32, 'reconstructions', i)

    print("Saved", i)

    if i != 0 and i % 100 == 0:
        break