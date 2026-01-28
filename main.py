import argparse
import os
import time
import json
import random
from datetime import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import importlib
import yaml
import torch
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from time import strftime
from copy import deepcopy
from scipy.special import softmax
from utils import color_codes, time_to_string


"""
> Arguments
"""


def parse_inputs():
    parser = argparse.ArgumentParser(
        description='Check data related to the MAGE - MRI relationship'
    )

    # Mode selector
    parser.add_argument(
        '-i', '--input-path',
        dest='path', default='/home/Data/IronMET_CGM',
        help='Path to the files (imaging and tabular data).'
    )
    options = vars(parser.parse_args())

    return options


"""
> Dummy main function
"""


def main():
    # Init
    c = color_codes()
    options = parse_inputs()

    path = options['path']
    bl_path = os.path.join(path, 'Basal_IronMET_CGM')
    fu_path = os.path.join(path, 'Follow_UP_IronMET_CGM')
    csv_file = os.path.join(path, 'data.IRMCGM.vicorob.csv')
    ironmet_data = pd.read_csv(csv_file)

    baseline_codes = os.listdir(bl_path)
    followup_codes = os.listdir(fu_path)

    patient_codes = np.unique(baseline_codes + followup_codes).tolist()

    deltas = []
    diffs = []
    surg_deltas = []
    surg_diffs = []
    surg_dict = {}
    for c in patient_codes:
        pd_idx = ironmet_data['ID'].str.contains(c)
        c_rows = ironmet_data[pd_idx]
        obesity_bl = c_rows.iloc[0]['Obesity'].tolist()
        obesity_fu = c_rows.iloc[1]['Obesity'].tolist()

        bmi_bl = c_rows.iloc[0]['BMI'].tolist()
        bmi_fu = c_rows.iloc[1]['BMI'].tolist()

        mage_bl = c_rows.iloc[0]['MAGE']
        mage_fu = c_rows.iloc[1]['MAGE']

        surgery_bl = c_rows.iloc[0]['Surgery'].tolist()
        surgery_fu = c_rows.iloc[1]['Surgery'].tolist()

        had_surgery = ironmet_data[pd_idx].iloc[0]['Surgery'].tolist() > 0

        age_bl = c_rows.iloc[0]['Age']
        age_fu = c_rows.iloc[1]['Age']

        bl_date = None
        fu_date = None
        surg_date_diff = None

        try:
            bl_date = datetime.strptime(
                c_rows.iloc[0]['Date'], '%d/%m/%Y'
            ).date()
            fu_date = datetime.strptime(
                c_rows.iloc[1]['Date'], '%d/%m/%Y'
            ).date()
            date_diff = relativedelta(fu_date, bl_date)
            diffs.append(fu_date - bl_date)
            if had_surgery:
                surg_date = datetime.strptime(
                    c_rows.iloc[1]['Surgerydate'], '%d/%m/%Y'
                ).date()
                surg_date_diff = relativedelta(
                    fu_date, surg_date
                )
        except TypeError:
            date_diff = None

        if date_diff is not None:
            deltas.append(date_diff)
            if had_surgery:
                surg_deltas.append(surg_date_diff)
                surg_diffs.append(fu_date - surg_date)

        surg_dict[c] = {
            'Obese': obesity_bl > 0,
            'HadSurgery': had_surgery,
            'DateDiff': date_diff,
            'Age': age_bl,
            'Baseline': {
                'Age': age_bl,
                'HasImage': c in baseline_codes,
                'Obesity': obesity_bl,
                'BMI': bmi_bl,
                'Surgery': surgery_bl,
                'HadSurgery': surgery_bl > 0,
                'MAGE': mage_bl,
                'Date': bl_date
            },
            'Follow-up': {
                'Age': age_fu,
                'HasImage': c in followup_codes,
                'Obesity': obesity_fu,
                'BMI': bmi_fu,
                'Surgery': surgery_fu,
                'HadSurgery': surgery_fu > 0,
                'MAGE': mage_fu,
                'Date': fu_date
            }
        }

    notobese_nosurg_mage = [
        c_data['Follow-up']['MAGE']
        for c, c_data in surg_dict.items()
        if c_data['Follow-up']['HasImage'] and not c_data['Obese'] and not c_data['HadSurgery']
    ]

    obese_nosurg_mage = [
        c_data['Follow-up']['MAGE']
        for c, c_data in surg_dict.items()
        if c_data['Follow-up']['HasImage'] and c_data['Obese'] and not c_data['HadSurgery']
    ]

    surg_mage = [
        c_data['Follow-up']['MAGE']
        for c, c_data in surg_dict.items()
        if c_data['Follow-up']['HasImage'] and c_data['HadSurgery']
    ]

    print(np.mean(deltas), np.mean(diffs), type(np.mean(diffs)))
    print(np.mean(surg_deltas), np.mean(surg_diffs))
    n_fu = len(notobese_nosurg_mage) + len(obese_nosurg_mage) + len(surg_mage)
    print(
        'Mean difference: {:d} years, {:d} months and {:d} days (from baseline)'.format(
            np.mean(deltas).years,
            np.mean(deltas).months,
            np.mean(deltas).days
        ),
    )
    print(
        'Mean difference: {:d} years, {:d} months and {:d} days (from surgery)'.format(
            np.mean(surg_deltas).years,
            np.mean(surg_deltas).months,
            np.mean(surg_deltas).days
        ),
    )

    print(
        'Not obese (no surgery)', len(notobese_nosurg_mage),
        '{: 5.2f}%'.format(100 * len(notobese_nosurg_mage) / n_fu)
    )
    print(
        'Obese (no surgery)    ', len(obese_nosurg_mage),
        '{: 5.2f}%'.format(100 * len(obese_nosurg_mage) / n_fu)
    )
    print(
        'Surgery               ', len(surg_mage),
        '{: 5.2f}%'.format(100 * len(surg_mage) / n_fu)
    )



if __name__ == '__main__':
    main()
