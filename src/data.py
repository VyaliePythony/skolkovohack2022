import numpy as np
import pandas as pd
import os
import re
from string import punctuation
from tqdm import tqdm

def set_path(path):
    global data_path, test_path
    data_path = path
    test_path = data_path+'test\\'

def preprocess_signs(text):
    text = text.lower()
    text = re.sub(r'<.*?>', " ", text)
    text = re.sub(r'[\_+\*+\#+\№\"\-+\+\=+\?+\&\^\.+\;\,+\>+\(\)\/+\:\\+]', " ", text)
    text = re.sub(r'[ ]{2,}',' ',text)
    text = text.strip()
    # tokens = mystem.lemmatize(text)
    # tokens = [snowball.stem(token) for token in tokens if token not in russian_stopwords\
    #             and token != " " \
    #             and token.strip() not in punctuation ]
    # text = " ".join(tokens)
    return text

def load_data():
    jobs_labels = ['JobId','Status','Name','Region','Description']
    test_jobs_labels = ['JobId','Status','Name','Region','Description','nan1','nan2','nan3']
    candidates_labels = ['CandidateId', 'Position', 'Sex', 'Citizenship', 'Age', 'Salary',
        'Langs', 'DriverLicense', 'Subway', 'Skills', 'Employment', 'Schedule',
        'CandidateRegion','DateCreated','JobId','CandidateStatusId','Status']
    candidates_education_labels = ['CandidateId', 'University', 'Faculty', 'GraduateYear']
    candidates_workplaces_labels = ['CandidateId', 'Position', 'FromYear', 'FromMonth', 'ToYear',
        'ToMonth']

    data_jobs = pd.read_csv(data_path+'data_jobs.csv',sep=';',names=jobs_labels)
    data_candidates_workplaces = pd.read_csv(data_path+'data_candidates_work_places.csv',sep=';',names=candidates_workplaces_labels)
    data_candidates_education = pd.read_csv(data_path+'data_candidates_education.csv',sep=';',names=candidates_education_labels)
    data_candidates = pd.read_csv(data_path+'data_candidates.csv',sep=';',names=candidates_labels)
    test_jobs = pd.read_csv(test_path+'test_jobs.csv',sep=';',names=test_jobs_labels)
    test_candidates_workplaces = pd.read_csv(test_path+'test_candidates_workplaces.csv',sep=';')
    test_candidates_education = pd.read_csv(test_path+'test_candidates_education.csv',sep=';')
    test_candidates = pd.read_csv(test_path+'test_candidates.csv',sep=';')


    # missing data
    data_jobs = data_jobs.fillna('').drop(['Status'],axis=1)
    data_candidates_workplaces = data_candidates_workplaces.fillna('').drop(['FromYear','FromMonth','ToYear','ToMonth'],axis=1)
    data_candidates_education = data_candidates_education.fillna('').drop(['GraduateYear','University'],axis=1)
    data_candidates = data_candidates.fillna('').drop(['Sex','Citizenship','Age','Salary','Subway','DateCreated','CandidateStatusId'],axis=1)
    test_jobs = test_jobs.fillna('').drop(['Status','nan1','nan2','nan3'],axis=1)
    test_candidates_workplaces = test_candidates_workplaces.fillna('').drop(['FromYear','FromMonth','ToYear','ToMonth'],axis=1)
    test_candidates_education = test_candidates_education.fillna('').drop(['GraduateYear','University'],axis=1)
    test_candidates = test_candidates.fillna('').drop(['Sex','Citizenship','Age','Salary','Subway'],axis=1)

    # preprocess
    data_jobs[['Name','Region','Description']] = data_jobs[['Name','Region','Description']].applymap(preprocess_signs)
    data_candidates_workplaces[['Position']] = data_candidates_workplaces[['Position']].applymap(preprocess_signs)
    data_candidates_education[['Faculty']] = data_candidates_education[['Faculty']].applymap(preprocess_signs)
    data_candidates[['Position','Langs','DriverLicense', \
            'Skills','CandidateRegion','Status']] = \
        data_candidates[['Position','Langs','DriverLicense', \
            'Skills','CandidateRegion','Status']].applymap(preprocess_signs)
    test_jobs[['Name','Region','Description']] = test_jobs[['Name','Region','Description']].applymap(preprocess_signs)
    test_candidates_workplaces[['Position']] = test_candidates_workplaces[['Position']].applymap(preprocess_signs)
    test_candidates_education[['Faculty']] = test_candidates_education[['Faculty']].applymap(preprocess_signs)
    test_candidates[['Position','Langs','DriverLicense', \
            'Skills','CandidateRegion']] = \
        test_candidates[['Position','Langs','DriverLicense', \
            'Skills','CandidateRegion']].applymap(preprocess_signs)

    for i in tqdm(range(len(data_candidates))):
        skills=['']
        if data_candidates['Skills'][i] != 0 and data_candidates['Skills'][i] != "||":
            skills = data_candidates['Skills'][i].split("||")
            a = ''.join(skills)
        data_candidates['Skills'][i] = a

    for i in tqdm(range(len(data_candidates))):
        languages = ['']
        if data_candidates['Langs'][i] != 0 and data_candidates['Langs'][i] != "||":
            a = data_candidates['Langs'][i].split("||")
            for j in range(len(a)-1):
                ll = a[j].split(':')
                languages.append(ll[0]) 
                b = ''.join(languages)
        data_candidates['Langs'][i] = b

    return {
        'train':{
            'job':
                [data_jobs],
            'candidate':
                [data_candidates, data_candidates_education, data_candidates_workplaces] },
        'test':{
            'job':
                [test_jobs],
            'candidate':
                [test_candidates, test_candidates_education, test_candidates_workplaces] } }
