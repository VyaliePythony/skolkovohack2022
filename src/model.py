import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd

import data

data_path = 'C:\\Users\\zer0nu11\\Documents\\grad\\skolkovohack2022\\data\\for_hack_2022\\'

data.set_path(data_path)

dataset = data.load_data()
df = dataset['train']['job']
# print(df)