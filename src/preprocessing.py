import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
jobs_labels = ['JobId','Status','Name','Region','Description']
candidates_labels = ['CandidateId', 'Position', 'Sex', 'Citizenship', 'Age', 'Salary',
       'Langs', 'DriverLicense', 'Subway', 'Skills', 'Employment', 'Schedule',
       'CandidateRegion','DateCreated','JobId','CandidateStatusId','Status']
candidates_education_labels = ['CandidateId', 'University', 'Faculty', 'GraduateYear']
candidates_workplaces_labels = ['CandidateId', 'Position', 'FromYear', 'FromMonth', 'ToYear',
       'ToMonth']

data_jobs = pd.read_csv('data_jobs.csv',sep=';',names=jobs_labels)
data_candidates_workplaces = pd.read_csv('data_candidates_work_places.csv',sep=';',names=candidates_workplaces_labels)
data_candidates_education = pd.read_csv('data_candidates_education.csv',sep=';',names=candidates_education_labels)
data_candidates = pd.read_csv('data_candidates.csv',sep=';',names=candidates_labels)
data_candidates = data_candidates.fillna(0)
#токенизируем позиции которые не для эмбединга(выходной формат вместо слова - цифра)
position = data_candidates.Position.unique()
Citizenship = data_candidates.Citizenship.unique()
Employmten = data_candidates.Employment.unique()
Schedule = data_candidates.Schedule.unique()
CandidateRegion = data_candidates.CandidateRegion.unique()
for i in tqdm(range(len(data_candidates))):
    a = list(position).index(data_candidates['Position'][i])
    data_candidates['Position'][i] = a
    a = list(Citizenship).index(data_candidates['Citizenship'][i])
    data_candidates['Citizenship'][i] = a   
    a = list(Employmten).index(data_candidates['Employment'][i])
    data_candidates['Employment'][i] = a
    a = list(Schedule).index(data_candidates['Schedule'][i])
    data_candidates['Schedule'][i] = a
    a = list(CandidateRegion).index(data_candidates['CandidateRegion'][i])
    data_candidates['CandidateRegion'][i] = a
#токенизируем для эмбединга слобец Langs(выход -вектор длинной с количества языков, где в каждом элементе показываетя степень владения языком(0-5) 
languages = []
level = []
for i in tqdm(range(len(data_candidates))):
    if data_candidates['Langs'][i] != 0 and data_candidates['Langs'][i] != "||":
        a = data_candidates['Langs'][i].split("||")
        for i in range(len(a)-1):
            ll = a[i].split(':')
            if ll[0] not in languages:
                languages.append(ll[0])
            if ll[1] not in level:
                level.append(ll[1]) 
 for i in tqdm(range(len(data_candidates))):
    vec = [0]*len(languages)
    if data_candidates['Langs'][i] != 0 and data_candidates['Langs'][i] != "||":
        a = data_candidates['Langs'][i].split("||")
        for j in range(len(a)-1):
            ll = a[j].split(':')
            lang = languages.index(ll[0])
            lev = level.index(ll[1])
            vec[lang] = lev+1
        data_candidates['Langs'][i] = vec
#токенизируем для эмбединнга столбец Skills(выход - вектор скилов, где будут 1 в ячейках где есть такие скиллы 0 -если нет)        
skills=[]
for i in tqdm(range(len(data_candidates))):
    if data_candidates['Skills'][i] != 0 and data_candidates['Skills'][i] != "||":
        a = data_candidates['Skills'][i].split("||")
        for j in range(len(a)-1):
            if a[j] not in skills:
                skills.append(a[j])

for i in tqdm(range(len(data_candidates))):
    vec = [0]*len(skills)
    if data_candidates['Skills'][i] != 0 and data_candidates['Skills'][i] != "||":
        a = data_candidates['Skills'][i].split("||")
        for j in range(len(a)-1):
            skill = skills.index(a[j])
            vec[skill] = 1
        data_candidates['Skills'][i] = vec
data_candidates.pop('DateCreated')
data_candidates_correct.pop('DriverLicense')
data_candidates_correct.pop('Subway')
