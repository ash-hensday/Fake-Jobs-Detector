
import pandas as pd

class Preprocessor:
    def __init__(self, postings_path, fake_postings_path, industries_path, job_industries_path):
        # Load DataFrames during initialization
        self.postings_df = pd.read_csv(postings_path)
        self.fake_postings_df = pd.read_csv(fake_postings_path)
        self.industries_df = pd.read_csv(industries_path)
        self.job_industries_df = pd.read_csv(job_industries_path)

    def preprocess_data(self):
        # Adding industry information to postings_df
        with_mapping_df = self.postings_df.merge(self.job_industries_df, on='job_id', how='left')
        final_postings_df = with_mapping_df.merge(self.industries_df, on='industry_id', how='left')

        # Remove unnecessary columns
        columns_to_remove = [
            'pay_period', 'industry_id', 'job_id', 'company_id', 'views', 'applies',
            'original_listed_time', 'remote_allowed', 'job_posting_url', 'application_url',
            'application_type', 'expiry', 'work_type', 'closed_time', 'formatted_experience_level',
            'skills_desc', 'listed_time', 'posting_domain', 'sponsored', 'currency',
            'compensation_type', 'max_salary', 'med_salary', 'min_salary', 'zip_code', 'fips'
        ]

        final_postings_df = final_postings_df.drop(columns=columns_to_remove)
        self.fake_postings_df = self.fake_postings_df.drop(columns=['benefits', 'requirements', 'fraudulent'])

        # Reorder columns
        self.fake_postings_df['company_profile'] = self.fake_postings_df['company_profile'].str.replace(r'\s*-\s*Established\s*\d{4}\.*', '', regex=True)
        final_postings_df = final_postings_df[['title', 'description', 'company_name', 'location', 'normalized_salary', 'formatted_work_type', 'industry_name']]

        return self.fake_postings_df, final_postings_df

    @staticmethod
    def calculate_mean_salary(salary_range):
        start, end = salary_range.replace('$', '').split('-')
        mean = (int(start) + int(end)) // 2
        return mean
 
    def process_salaries(self):
        self.fake_postings_df['salary_range'] = self.fake_postings_df['salary_range'].apply(self.calculate_mean_salary)

    def prepare_data_for_model(self):
        fake_postings_df, final_postings_df = self.preprocess_data()
        self.process_salaries()

        final_postings_df = final_postings_df.fillna(' ').astype(str)
        X1 = final_postings_df.apply(' '.join, axis=1).tolist()

        fake_postings_df = fake_postings_df.fillna(' ').astype(str)
        X2 = fake_postings_df.apply(' '.join, axis=1).tolist()

        y1 = [0] * len(X1)
        y2 = [1] * len(X2)

        X = list(X1) + list(X2)
        y = y1 + y2

        return X, y
