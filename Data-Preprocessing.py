
import pandas as pd
from transformers import AlbertTokenizer
import numpy as np

#Load Dataframes
postings_df = pd.read_csv(r"C:\Users\ashcr\Downloads\Fake Job Detector\postings.csv")
fake_postings_df = pd.read_csv(r"C:\Users\ashcr\Downloads\Fake Job Detector\Fake Postings.csv")
industries_df = pd.read_csv(r"C:\Users\ashcr\Downloads\Fake Job Detector\industries.csv")
job_industries_df = pd.read_csv(r"C:\Users\ashcr\Downloads\Fake Job Detector\job_industries.csv")

#Adding industries because in this dataset all the jobs have an ID that corresponds to a separate mapping csv 
#which uses a separate lookup csv of industry names and its just too much.
with_mapping_df = postings_df.merge(job_industries_df, 
                                    on='job_id',
                                    how='left')

final_postings_df = with_mapping_df.merge(industries_df,
                                          on='industry_id',
                                          how='left')

# Remove unnecessary columns 
columns_to_remove = [
        'pay_period',
        'industry_id',
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

final_postings_df = final_postings_df.drop(columns=columns_to_remove)
fake_postings_df = fake_postings_df.drop(columns='benefits')


#Add labels to real postings
final_postings_df['fraudulent'] = 0

#Reorder columns 
print(list(final_postings_df.columns.values))
###fake_postings_df['company_profile'] = fake_postings_df['company_profile'].str.replace(r'\s*-\s*Established\s*\d{4}\.*', '', regex=True)
final_postings_df = final_postings_df[['title', 'description', 'company_name', 'location', 'normalized_salary', 'formatted_work_type',
                                       'industry_name', 'fraudulent']]

def calculate_mean_salary(salary_range):
    start, end = salary_range.replace('$', '').split('-')
    mean = (int(start) + int(end)) // 2
    return mean

fake_postings_df['salary_range'] = fake_postings_df['salary_range'].apply(calculate_mean_salary)

#Prepare to load into model
category_template = ['title', 'description', 'company_name', 'location', 'salary', 'work_type', 'industry', 'fraudulent']

def label_with_categories(df, categories):
    return df.apply(lambda row: [f"{cat}:{val}" for cat, val in zip(categories, row)], axis=1)

labeled_final_postings_df = label_with_categories(final_postings_df, category_template)
labeled_fake_postings_df= label_with_categories(fake_postings_df, category_template)

X1 = labeled_final_postings_df.apply(lambda x: ' '.join(x), axis=1).values
X2 = labeled_fake_postings_df.apply(lambda x: ' '.join(x), axis=1).values