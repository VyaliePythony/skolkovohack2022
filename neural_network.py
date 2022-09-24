import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(625, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.fc4 = nn.Linear(10, 1)
    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return F.log_softmax(x)
model = Net()
optimizer = optim.SGD(net.parameters(), lr=lr=1e-3, momentum=0.9)
criterion = nn.CrossEntropyLoss()
from torch.utils.data import DataLoader

batch_size = 128
epochs = 50
history = []
train_loader = DataLoader(train, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, drop_last=True)
for i in range(epochs):
  for x_batch, y_batch in train_loader:
    x_batch = x_batch.view(x_batch.shape[0], -1).to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)

    loss = criterion(logits, y_batch)
    history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

  print(f'{i+1},\t loss: {history[-1]}')
plt.figure(figsize=(10, 7))

plt.plot(history)

plt.title('Loss by batch iterations')
plt.ylabel('Entropy Loss')
plt.xlabel('batches')

plt.show()
acc = 0
batches = 0

for x_batch, y_batch in test_loader:
  batches += 1
  x_batch = x_batch.view(x_batch.shape[0], -1).to(device)
  y_batch = y_batch.to(device)

  preds = torch.argmax(model(x_batch), dim=1)
  acc += (preds==y_batch).cpu().numpy().mean()

print(f'Test accuracy {acc / batches:.3}')
from sklearn.metrics import accuracy_score
