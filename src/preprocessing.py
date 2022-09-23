import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
data_candidates = pd.read_csv('data_candidates.csv',sep=';')
data_candidates.rename(columns={'7435': "CandidateId", 'Водитель-экспедитор': "Position", '2': "Sex",
                                'Россия': "Citizenship" ,'21' : 'Age', '0': 'Salary', 'NULL': ' Langs',
                                'NULL.1' : 'DriverLicense','NULL.2' : 'Subway' , 'NULL.3': 'Skills','Full' : 'Employment','Full.1' : 'Shedule',
                               'Санкт-Петербург' : 'CandidateRegion','2014-01-15 00:00:00.0000000' : 'DateCreated','163' : 'JobId','1425' : 'CandidateStatusId','Отклонен' : 'Status'
                                })