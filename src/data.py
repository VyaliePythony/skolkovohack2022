import numpy as np
import pandas as pd
import os
import re
from string import punctuation
from status_decode import stat

def set_path(path):
    global data_path, test_path
    data_path = path
    test_path = data_path+'test\\'

def preprocess_signs(text):
    text = text.lower()
    text = re.sub(r'<.*?>', " ", text)
    text = re.sub(r'[\_+\*+\#+\№\"\-+\+\=+\?+\&\^\.+\;\,+\>+\(\)\/+]', " ", text)
    text = re.sub(r'[ ]{2,}',' ',text)
    text = text.strip()
    return text

def select_langs(text):
    text = re.sub(r':.*?\|\|', " ", text)
    text = text.strip()
    text = re.sub(r'[\|+]', " ", text)
    text = ' '.join(set(text.split()))
    return text

def remove_slash(text):
    text = re.sub(r'[\:]', "", text)
    return re.sub(r' \|\|', "", text)

def driver_license_fix(text):
    text = ' '.join(set(list(re.sub(r'[А-я ]+', "", text))))
    return ('категории '+text) if text else text

def merge_list(text_list):
    return ' '.join(text_list).strip()

def load_data():
    global data_candidates_workplaces, test_candidates_workplaces, \
        data_candidates_education, test_candidates_education, \
        data_candidates, test_candidates, \
        data_jobs, test_jobs, \
        status

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
    data_candidates = data_candidates.fillna('').drop(['Sex','Citizenship','Age','Salary','Subway','Employment','Schedule','DateCreated','CandidateStatusId'],axis=1)
    test_jobs = test_jobs.fillna('').drop(['Status','nan1','nan2','nan3'],axis=1)
    test_candidates_workplaces = test_candidates_workplaces.fillna('').drop(['FromYear','FromMonth','ToYear','ToMonth'],axis=1)
    test_candidates_education = test_candidates_education.fillna('').drop(['GraduateYear','University'],axis=1)
    test_candidates = test_candidates.fillna('').drop(['Sex','Citizenship','Age','Salary','Subway','Employment','Schedule'],axis=1)

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

    data_candidates[['Langs']] = data_candidates[['Langs']].applymap(select_langs)
    test_candidates[['Langs']] = test_candidates[['Langs']].applymap(select_langs)
    data_candidates[['Skills']] = data_candidates[['Skills']].applymap(remove_slash)
    test_candidates[['Skills']] = test_candidates[['Skills']].applymap(remove_slash)
    data_candidates[['DriverLicense']] = data_candidates[['DriverLicense']].applymap(driver_license_fix)
    test_candidates[['DriverLicense']] = test_candidates[['DriverLicense']].applymap(driver_license_fix)

    data_candidates_pure = data_candidates[['CandidateId','Position','Langs','DriverLicense','Skills','CandidateRegion']]
    status = data_candidates[['CandidateId', 'JobId', 'Status']]

    test_candidates_workplaces = test_candidates_workplaces.groupby(['CandidateId'])['Position'].apply(list).reset_index()
    data_candidates_workplaces = data_candidates_workplaces.groupby(['CandidateId'])['Position'].apply(list).reset_index()
    test_candidates_education = test_candidates_education.groupby(['CandidateId'])['Faculty'].apply(list).reset_index()
    data_candidates_education = data_candidates_education.groupby(['CandidateId'])['Faculty'].apply(list).reset_index()

    test_candidates_workplaces[['Position']] = test_candidates_workplaces[['Position']].applymap(merge_list)
    data_candidates_workplaces[['Position']] = data_candidates_workplaces[['Position']].applymap(merge_list)
    test_candidates_education[['Faculty']] = test_candidates_education[['Faculty']].applymap(merge_list)
    data_candidates_education[['Faculty']] = data_candidates_education[['Faculty']].applymap(merge_list)

    data_candidates = data_candidates_pure.drop_duplicates(subset=['CandidateId'])
    
    data_candidates_tmp = pd.merge(data_candidates_education, data_candidates_workplaces, left_on='CandidateId', right_on='CandidateId')
    data_candidates = pd.merge(data_candidates, data_candidates_tmp, left_on='CandidateId', right_on='CandidateId')
    test_candidates_tmp = pd.merge(test_candidates_education, test_candidates_workplaces, left_on='CandidateId', right_on='CandidateId')
    test_candidates = pd.merge(test_candidates, test_candidates_tmp, left_on='CandidateId', right_on='CandidateId')

    # set metric instead of status + normalize
    status[['Status']] = status[['Status']].applymap(lambda x: stat[x]/10.0)

def get_soup_job(jobs):
    tmp = jobs.copy()
    def soup(row):
        res = ' '.join([row['Name'],row['Description']])
        res = re.sub(r'[ ]{2,}',' ',res)
        return res
    tmp['Soup'] = tmp[['Name','Description']].apply(soup, axis=1)
    return tmp[['JobId','Region','Soup']]

def get_soup_candidate(candidates):
    tmp = candidates.copy()
    def soup(row):
        res = ' '.join([row['Position_x'],row['Langs'],row['DriverLicense'],row['Skills'],row['Faculty'],row['Position_y']])
        res = re.sub(r'[ ]{2,}',' ',res)
        return res
    tmp['Soup'] = tmp[['Position_x','Langs','DriverLicense','Skills','Faculty','Position_y']].apply(soup, axis=1)
    return tmp[['CandidateId','CandidateRegion','Soup']]