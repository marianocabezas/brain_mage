import argparse
import os
import nibabel as nib
import time
import json
import random
from datetime import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import importlib
import yaml
from time import strftime
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
import torch
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from utils import color_codes, time_to_string
from registration import resample



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


def get_data_dict():
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
        surg_date = None
        surg_date_diff = None

        try:
            bl_date = datetime.strptime(
                c_rows.iloc[0]['Date'], '%d/%m/%Y'
            ).date()
            fu_date = datetime.strptime(
                c_rows.iloc[1]['Date'], '%d/%m/%Y'
            ).date()
            date_diff = relativedelta(fu_date, bl_date)
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
            diffs.append(fu_date - bl_date)
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
    return surg_dict, deltas, diffs, surg_deltas, surg_diffs


def image_info(path, data_dict):
    for c, c_data in data_dict.items():
        if c_data['Follow-up']['HasImage'] and c_data['Baseline']['HasImage']:
            bl_nii = nib.load(
                os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE.nii')
            )
            fu_nii = nib.load(
                os.path.join(path, 'Follow_UP_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE.nii')
            )
            bl_x, bl_y, bl_z = bl_nii.get_fdata().shape
            bl_sx, bl_sy, bl_sz = bl_nii.header.get_zooms()
            fu_x, fu_y, fu_z = fu_nii.get_fdata().shape
            fu_sx, fu_sy, fu_sz = fu_nii.header.get_zooms()
            print(
                'Subject {:} - Baseline {:3d} x {:3d} x {:3d} ({:4.2f} x {:4.2f} x {:4.2f})'.format(
                    c, bl_x, bl_y, bl_z,  bl_sx, bl_sy, bl_sz
                ), end=' '
            )
            print(
                '- Follow-up {:3d} x {:3d} x {:3d} ({:4.2f} x {:4.2f} x {:4.2f})'.format(
                    fu_x, fu_y, fu_z, fu_sx, fu_sy, fu_sz
                )
            )
            print(fu_sx, fu_sy, fu_sz)

            bl_im = bl_nii.get_fdata()
            affine = torch.eye(4, dtype=torch.float64)

            bl_new = resample(
                bl_im, bl_nii.header.get_zooms(),
                fu_nii.shape, fu_nii.header.get_zooms(),
                affine
            )

            print(bl_new.shape)


def mage_info(path, data_dict):
    notobese_nosurg_mage = [
        c_data['Follow-up']['MAGE']
        for c, c_data in data_dict.items()
        if c_data['Follow-up']['HasImage'] and not c_data['Obese'] and not c_data['HadSurgery']
    ]

    obese_nosurg_mage = [
        c_data['Follow-up']['MAGE']
        for c, c_data in data_dict.items()
        if c_data['Follow-up']['HasImage'] and c_data['Obese'] and not c_data['HadSurgery']
    ]

    surg_mage = [
        c_data['Follow-up']['MAGE']
        for c, c_data in data_dict.items()
        if c_data['Follow-up']['HasImage'] and c_data['HadSurgery']
    ]

    n_fu = len(notobese_nosurg_mage) + len(obese_nosurg_mage) + len(surg_mage)


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

    # Absolute MAGE at follow-up

    x = ['Not obese (no surgery)'] * len(notobese_nosurg_mage) + ['Obese (no surgery)'] * len(obese_nosurg_mage) + [
        'Surgery'] * len(surg_mage)
    y = notobese_nosurg_mage + obese_nosurg_mage + surg_mage

    mage_df = pd.DataFrame(list(zip(y, x)), columns=['MAGE', 'Group'])

    # MAGE difference data

    notobese_nosurg_diffmage = [
        data['Follow-up']['MAGE'] - data['Baseline']['MAGE']
        for c, data in data_dict.items()
        if
        data['Follow-up']['HasImage'] and data['Baseline']['HasImage'] and not data['Obese'] and not data['HadSurgery']
    ]

    obese_nosurg_diffmage = [
        data['Follow-up']['MAGE'] - data['Baseline']['MAGE']
        for c, data in data_dict.items()
        if data['Follow-up']['HasImage'] and data['Baseline']['HasImage'] and data['Obese'] and not data['HadSurgery']
    ]

    surg_diffmage = [
        data['Follow-up']['MAGE'] - data['Baseline']['MAGE']
        for c, data in data_dict.items()
        if data['Follow-up']['HasImage'] and data['Baseline']['HasImage'] and data['HadSurgery']
    ]

    x = ['Not obese (no surgery)'] * len(notobese_nosurg_diffmage) + ['Obese (no surgery)'] * len(
        obese_nosurg_diffmage) + ['Surgery'] * len(surg_diffmage)
    y = notobese_nosurg_diffmage + obese_nosurg_diffmage + surg_diffmage

    diffmage_df = pd.DataFrame(list(zip(y, x)), columns=['DiffMAGE', 'Group'])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Group', y='MAGE', data=mage_df)
    plt.xticks(rotation=45)
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Group', y='DiffMAGE', data=diffmage_df)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(path, 'mage_boxplots.png'))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.violinplot(x='Group', y='MAGE', data=mage_df)
    plt.xticks(rotation=45)
    plt.subplot(1, 2, 2)
    sns.violinplot(x='Group', y='DiffMAGE', data=diffmage_df)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(path, 'mage_violinplots.png'))


def main():
    # Init
    options = parse_inputs()
    path = options['path']
    surg_dict, deltas, diffs, surg_deltas, surg_diffs = get_data_dict()

    print(np.mean(deltas), np.mean(diffs), type(np.mean(diffs)))
    print(np.mean(surg_deltas), np.mean(surg_diffs))
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

    print('-'.join([''] * 30))

    mage_info(path, surg_dict)

    print('-'.join([''] * 30))

    image_info(path, surg_dict)


if __name__ == '__main__':
    main()
