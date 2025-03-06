import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the Generator Network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )
    
    def forward(self, z):
        return self.model(z)

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Hyperparameters
input_dim = 100
output_dim = 8192  # Length of time-series
batch_size = 24
lr_G = 2e-4
lr_D = 1e-5
num_epochs = 10000

# Instantiate models
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training Loop
for epoch in range(num_epochs):
    # Train Discriminator
    real_data = torch.randn((batch_size, output_dim))  # Simulating real financial data
    z = torch.randn((batch_size, input_dim))
    fake_data = generator(z).detach()
    
    optimizer_D.zero_grad()
    
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)
    
    real_loss = criterion(discriminator(real_data), real_labels)
    fake_loss = criterion(discriminator(fake_data), fake_labels)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()
    
    # Train Generator
    optimizer_G.zero_grad()
    fake_data = generator(z)
    g_loss = criterion(discriminator(fake_data), real_labels)
    g_loss.backward()
    optimizer_G.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}')

# Generate Synthetic Time-Series Data
z = torch.randn((1, input_dim))
generated_series = generator(z).detach().numpy().flatten()

# Plot the Generated Time-Series
time = np.arange(output_dim)
plt.figure(figsize=(10, 5))
plt.plot(time, generated_series)
plt.xlabel('Time')
plt.ylabel('Generated Price Return')
plt.title('Generated Financial Time-Series')
plt.show()
