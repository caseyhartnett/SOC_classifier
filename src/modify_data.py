"""Modify extracted data to prepare for ML."""
import os
import pandas as pd

CURR_PATH = os.path.dirname(os.path.realpath(__file__))

def int_round(n: int, sigfigs: int) -> int:
    """
    Round integer using sigfigs.

    :param n: integer to be rounded
    :param sigfigs: number of sigfigs for output
    :return: the integer rounded to the sigfig
    """
    n = str(n)
    return n[:sigfigs] + ('0' * (len(n) - (sigfigs)))


def modify_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the data to extract needed pieces of data.

    :param data: a pandas DataFrame with column soc_code
    :return: a pandas DataFrame with additional columns
    """
    # break down soc codes to useful pieces
    # https://www.bls.gov/soc/2018/soc_2018_class_and_coding_structure.pdf
    data['soc_code_split'] = data['soc_code'].str.split('-')
    data['major_group'] = data['soc_code_split'].apply(lambda x: int(x[0]))
    data['occ_number'] = data['soc_code_split'].apply(lambda x: int(float(x[1])))
    data['minor_group'] = data['soc_code_split'].apply(lambda x: int_round(int(float(x[1])), 1))

    # mapping provided from above pdf
    data['high_level_groups'] = data.major_group.map({
        11: 1, 12: 1, 13: 1,
        15: 2, 16: 2, 17: 2, 18: 2, 19: 2,
        21: 3, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3,
        29: 4,
        31: 5, 32: 5, 33: 5, 34: 5, 35: 5, 36: 5, 37: 5, 38: 5, 39: 5,
        41: 6,
        43: 7,
        45: 8,
        47: 9,
        49: 10,
        51: 11,
        53: 12,
        55: 13})
    return data

def apply_modifications(job_sample_data: pd.DataFrame, job_title_data:pd.DataFrame) -> pd.DataFrame:
    """
    Merge data sources, change columnn names and apply modifications to data.

    :param job_sample_data: a pandas DataFrame of onet_data.txt
    :param job_title_data: a pandas DataFrame of onet_job_titles.txt
    :return: a pandas DataFrame
    """
    job_sample_data = job_sample_data.drop('Shown in My Next Move', axis=1)
    job_sample_data = job_sample_data.rename(
        columns={'O*NET-SOC Code': 'soc_code', 'Reported Job Title': 'job_title'})
    job_title_data = job_title_data.drop('Description', axis=1)
    job_title_data = job_title_data.rename(
        columns={'O*NET-SOC Code': 'soc_code', 'Title': 'soc_title'})
    data = pd.merge(job_sample_data, job_title_data)  # combine data sources
    data = modify_data(data)  # modify merged data
    return data

if __name__ == '__main__':
    job_sample_data = pd.read_table(os.path.join(CURR_PATH, '../data/onet_data.txt'))
    job_title_data = pd.read_table(os.path.join(CURR_PATH, '../data/onet_job_titles.txt'))
    data = apply_modifications(job_sample_data, job_title_data)
    data.to_csv(os.path.join(CURR_PATH, '../data/modified_data.csv'))
