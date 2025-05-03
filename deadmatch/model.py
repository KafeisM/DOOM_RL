import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        # Capas convolucionales
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Calcular el tamaño tras las convoluciones
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv3(self.conv2(self.conv1(dummy)))
            conv_out_size = conv_out.flatten(1).shape[1]
        # Capas totalmente conectadas
        self.fc1    = nn.Linear(conv_out_size, 512)
        self.fc_out = nn.Linear(512, num_actions)

    def forward(self, x):
        # x: (batch, canales, alto, ancho)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Usamos reshape para aplanar (manteniendo batch)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc_out(x)
