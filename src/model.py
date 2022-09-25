import pandas as pd

# dataset markup module
import data

#dataset path
data_path = 'C:\\Users\\zer0nu11\\Documents\\grad\\skolkovohack2022\\data\\for_hack_2022\\'

data.set_path(data_path)

# train on dataset
def train():
    data.load_data()
    jobs = data.get_embedding(data.get_soup_job(data.data_jobs))
    candidates = data.get_embedding(data.get_soup_candidate(data.data_candidates))
    # jobs.to_csv(data.data_path+'jobs_embed.csv')
    # candidates.to_csv(data.data_path+'candidates_embed.csv')
    train = data.get_train_data(jobs,candidates,data.status)
    train = data.make_train_array(train)

    # TRAIN MODEL

# predict on test dataset
def predict_test():
    data.load_data(raw=True)
    identities,test = data.pair_to_vec(data.test_jobs,data.test_candidates,data.test_candidates_workplaces,data.test_candidates_education)

    # MODEL PREDICT ON TEST

    # SET DEPENDENCIES OF PREDICTIONS BY IDENTITIES

# predict on dataframes
def predict(jobs,candidates,candidates_workplaces,candidates_education):
    identities,test = data.pair_to_vec(jobs,candidates,candidates_workplaces,candidates_education)

    # MODEL PREDICT ON TEST

    # SET DEPENDENCIES OF PREDICTIONS BY IDENTITIES

def predict_json(json):
    res = json2pdFrame(json)
    return 0

def json2pdFrame(json):
    job = 0
    candidate = 0
    return 0