from operator import mod
from statistics import mode
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
# dataset markup module
import data

#dataset path
data_path = 'C:\\Users\\zer0nu11\\Documents\\grad\\skolkovohack2022\\data\\for_hack_2022\\'

data.set_path(data_path)

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

def load_model():
    model = torch.load(data_path+'model.pth')
    model.eval()

# train on dataset
def train(full=False,save=False):
    if full:
        data.load_data()
        jobs = data.get_embedding(data.get_soup_job(data.data_jobs))
        candidates = data.get_embedding(data.get_soup_candidate(data.data_candidates))
        train = data.get_train_data(jobs,candidates,data.status)
        train = data.make_train_array(train)
        if save:
            np.save(data_path+'train.npy', train)
            return 0
    else:
        train = np.load(data_path+'train.npy')

    y = train[:,625].reshape(train.shape[0],1) # last columns is metric status
    x = train[:,:-1] # 312+312+1 | vec1,vec2,region
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        shuffle=True, random_state=42)
    # TRAIN MODEL
    batch_size = 300
    epochs = 100
    history = []
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # learn on train dataset
    for i in range(epochs):
        for x_batch,y_batch in train_loader:

            logits = model(x_batch.float())
            #y_batch = y_batch.type(torch.LongTensor)
            loss = criterion(logits, y_batch.float())
            history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        print(f'{i+1},\t loss: {history[-1]}')

    torch.save(model, data_path+'model.pth')

    plt.figure(figsize=(10, 7))

    plt.plot(history)

    plt.title('Loss ')
    plt.ylabel('MSELoss')
    plt.xlabel('batches')

    plt.show()
    from sklearn.metrics import accuracy_score
    acc = 0
    batches = 0

    # predict on test dataset
    for x_batch, y_batch in test_loader:
        batches += 1
        preds = model(x_batch.float())
        preds = torch.round(preds)
        y_batch = torch.round(y_batch)
        acc += (preds==y_batch).numpy().mean()

    print(f'Test accuracy {acc / batches:.3}')

# test on dataset
def predict_test():
    data.load_data(raw=True)
    res = predict(data.test_jobs,data.test_candidates,data.test_candidates_workplaces,data.test_candidates_education)
    # MODEL PREDICT ON TEST
    
    # SET DEPENDENCIES OF PREDICTIONS BY IDENTITIES
    return res

# predict on dataframes
def predict(jobs,candidates,candidates_workplaces,candidates_education):
    identities,test = data.pair_to_vec(jobs,candidates,candidates_workplaces,candidates_education)

    # MODEL PREDICT ON TEST
    # predictions = []
    # for row in test:
    #     predictions.append(model( torch.from_numpy(row).float() ))
    predictions = model( torch.from_numpy(test).float() )
    identities['predict'] = predictions.detach().numpy()
    return identities.sort_values(by=['predict'],ascending=False)
    # SET DEPENDENCIES OF PREDICTIONS BY IDENTITIES

# predict on JSON
def predict_json(json):
    res = json2pdFrame(json)
    """
    1. JSON -> dataframes
    2. predict()
    """
    res = predict(res)
    return 0

def json2pdFrame(json):
    job = 0
    candidate = 0
    return 0

def decode_prediction(predicts):
    data.load_data(raw=True)
    jobs, candidates = data.pair_to_vec(data.test_jobs,data.test_candidates,data.test_candidates_workplaces,data.test_candidates_education, soup=True)

    res=[]

    for i in range(predicts.shape[0]):
        job_soup = jobs[jobs.id==predicts.job_id[i]].Soup
        candidates_soup = candidates[candidates.id==predicts.cand_id[i]].Soup
        metric = predicts.predict[i]
        res.append([metric,job_soup,candidates_soup])

    return pd.DataFrame(data=res,columns=['metric','job','candidate'])

def save_predictions(predicts):
    jobs = predicts.job_id.unique().tolist()
    for i in jobs:
        i_predicts = predicts[predicts.job_id == i]
        sort_pred = i_predicts.sort_values(by=['predict'],ascending=False)
        tmp = sort_pred[['cand_id','predict']]
        tmp.to_csv(data_path+f'result_job_{i}.csv',header=False)
