import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
data_candidates = pd.read_csv('data_candidates.csv',sep=';')
data_candidates = pd.concat([pd.DataFrame([data_candidates.columns.values], columns=['CandidateId', 'Position', 'Sex', 'Citizenship', 'Age', 'Salary', 'Langs', 'DriverLicense', 'Subway', 
                                                                                     'Skills', 'Employment', 'Shedule','CandidateRegion','DateCreated', 'JobId', 'CandidateStatusId', 'Status']), data_candidates], ignore_index=True)
