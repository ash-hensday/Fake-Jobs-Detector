import pandas as pd
from transformers import AlbertTokenizer
import numpy as np

#Load Dataframes
postings_df = pd.read_csv('postings.csv')
fake_postings_df = pd.read_csv('Fake Postings.csv')
industries_df = pd.read_csv('industries.csv')

# Remove unnecessary columns 
columns_to_remove = [
        'pay_period',
        'job_id',
        'company_id',
        'views',
        'applies',
        'original_listed_time',
        'remote_allowed',
        'job_posting_url',
        'application_url',
        'application_type',
        'expiry',
        'work_type',
        'closed_time',
        'formatted_experience_level',
        'skills_desc',
        'listed_time',
        'posting_domain',
        'sponsored',
        'currency',
        'compensation_type',
        'max_salary',
        'med_salary',
        'min_salary',
        'zip_code',
        'fips'
    ]
postings_df = postings_df.drop(columns=columns_to_remove)
fake_postings_df = fake_postings_df.drop(columns='benefits')

#Add industry information to real postings from separate csv
postings_df['industry'] = industries_df['industry_name']

#Add labels to real postings
postings_df['fraudulent'] = 0







