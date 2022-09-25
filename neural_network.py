import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score
train = np.load('train.npy')
y = train[:,625].reshape(40570,1)
x = train[:,:625]
x = torch.from_numpy(x)
y = torch.from_numpy(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    shuffle=True, random_state=42)
model = nn.Sequential(
  nn.Linear(625, 200),
  nn.ReLU(), 
  nn.Linear(200, 200), 
  nn.ReLU(),
  nn.Dropout(),
  nn.Linear(200, 60),
  nn.ReLU(),
  nn.Linear(60, 10), 
  nn.ReLU(),
  nn.Linear(10, 1),
  nn.Sigmoid()
)
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
criterion = nn.MSELoss()
from torch.utils.data import DataLoader,TensorDataset

batch_size = 300
epochs = 50
history = []
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

for i in range(100):
    for x_batch,y_batch in train_loader:
    
        logits = model(x_batch.float())
        #y_batch = y_batch.type(torch.LongTensor)
        loss = criterion(logits, y_batch.float())
        history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print(f'{i+1},\t loss: {history[-1]}')

plt.figure(figsize=(10, 7))

plt.plot(history)

plt.title('Loss ')
plt.ylabel('MSELoss')
plt.xlabel('batches')

plt.show()
from sklearn.metrics import accuracy_score
acc = 0
batches = 0

for x_batch, y_batch in test_loader:
  batches += 1
  preds = model(x_batch.float())
  preds = torch.round(preds)
  y_batch = torch.round(y_batch)
  acc += (preds==y_batch).numpy().mean()

print(f'Test accuracy {acc / batches:.3}')
