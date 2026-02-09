import os
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import nibabel as nib
from skimage import filters
from datetime import datetime
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes, label
from dateutil.relativedelta import relativedelta
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from utils import color_codes, time_to_string
from registration import resample, halfway_registration, mse_loss, xcor_loss
from registration import sitk_registration



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
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int, default=500,
        help='Number of epochs'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=100,
        help='Maximum number of epochs without improvement'
    )
    parser.add_argument(
        '-l', '--learning-rate',
        dest='learning_rate',
        type=float, default=1e-3,
        help='Number of epochs'
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

    surg_dict = {}
    for c in patient_codes:
        if c in baseline_codes and c in followup_codes:
            pd_idx = ironmet_data['ID'].str.contains(c)
            c_rows = ironmet_data[pd_idx]

            bmi_bl = c_rows.iloc[0]['BMI'].tolist()
            bmi_fu = c_rows.iloc[1]['BMI'].tolist()

            mage_bl = c_rows.iloc[0]['MAGE']
            mage_fu = c_rows.iloc[1]['MAGE']

            had_surgery = ironmet_data[pd_idx].iloc[0]['Surgery'].tolist() > 0

            age_bl = c_rows.iloc[0]['Age']
            age_fu = c_rows.iloc[1]['Age']

            surg_dict[c] = {
                'Obese': bmi_bl >= 30,
                'HadSurgery': had_surgery,
                'Age': age_bl,
                'Baseline': {
                    'Age': age_bl,
                    'HasImage': c in baseline_codes,
                    'BMI': bmi_bl,
                    'MAGE': mage_bl,
                },
                'Follow-up': {
                    'Age': age_fu,
                    'HasImage': c in followup_codes,
                    'BMI': bmi_fu,
                    'MAGE': mage_fu,
                }
            }
    return surg_dict


def show_slices(image_list, path, file_prefix):
    for i, im in enumerate(image_list):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        _, x, y, z = im.shape
        bl_2d = im[0, :, :, z // 2]
        plt.imshow((bl_2d - np.min(bl_2d)) / (np.max(bl_2d) - np.min(bl_2d)), cmap='gray')
        plt.xticks(rotation=45)
        plt.subplot(1, 2, 2)
        fu_2d = im[1, :, :, z // 2]
        plt.imshow((fu_2d - np.min(fu_2d)) / (np.max(fu_2d) - np.min(fu_2d)), cmap='gray')
        plt.savefig(os.path.join(path, '{:}_{:02d}.png'.format(file_prefix, i)))
        plt.close()



def main():
    # Init
    options = parse_inputs()
    path = options['path']
    epochs = options['epochs']
    patience = options['patience']
    lr = options['learning_rate']

    bl_path = os.path.join(path, 'Basal_IronMET_CGM')
    fu_path = os.path.join(path, 'Follow_UP_IronMET_CGM')
    csv_file = os.path.join(path, 'data.IRMCGM.vicorob.csv')
    ironmet_data = pd.read_csv(csv_file)

    baseline_codes = os.listdir(bl_path)
    followup_codes = os.listdir(fu_path)

    patient_codes = np.unique(baseline_codes + followup_codes).tolist()

    labels = []
    mages = []
    diffmages = []
    baselines = []
    followups = []
    masks = []
    healthy = []
    obese = []
    surgery = []
    for c in patient_codes:
        if c in baseline_codes and c in followup_codes:

            pd_idx = ironmet_data['ID'].str.contains(c)
            c_rows = ironmet_data[pd_idx]

            bmi_bl = c_rows.iloc[0]['BMI'].tolist()

            had_surgery = ironmet_data[pd_idx].iloc[0]['Surgery'].tolist() > 0

            is_obese = c_rows.iloc[0]['Obesity'].tolist() > 0

            mage_bl = c_rows.iloc[0]['MAGE']
            mage_fu = c_rows.iloc[1]['MAGE']

            print(
                'Subject {:} - Baseline | BMI = {:>5.2f}{:} | MAGE = {:>5.2f} / {:>6.2f}{:}'.format(
                    c, bmi_bl, ' (obese) ' if is_obese else '         ',
                    mage_bl, mage_fu, ' | Surgery' if had_surgery else ''
                )
            )

            bl_filename = os.path.join(bl_path, c, 'sT1W_3D_TFE_SENSE_coreg.nii.gz')
            bl_mask_filename = os.path.join(bl_path, c, 'sT1W_3D_TFE_SENSE_coreg_mask.nii.gz')
            baseline = nib.load(bl_filename).get_fdata()
            bl_mask = nib.load(bl_mask_filename).get_fdata().astype(bool)

            fu_filename = os.path.join(fu_path, c, 'sT1W_3D_TFE_SENSE_coreg.nii.gz')
            fu_mask_filename = os.path.join(fu_path, c, 'sT1W_3D_TFE_SENSE_coreg_mask.nii.gz')
            followup = nib.load(fu_filename).get_fdata()
            fu_mask = nib.load(fu_mask_filename).get_fdata().astype(bool)
            mask = np.logical_and(bl_mask, fu_mask)

            masks.append(mask)

            baseline_masked = baseline.copy()
            baseline_masked[np.logical_not(mask)] = 0
            followup_masked = followup.copy()
            followup_masked[np.logical_not(mask)] = 0

            if bmi_bl < 30:
                healthy.append(np.stack([baseline_masked, followup_masked], axis=0))
            elif not had_surgery:
                obese.append(np.stack([baseline_masked, followup_masked], axis=0))
            else:
                surgery.append(np.stack([baseline_masked, followup_masked], axis=0))

    mask = np.sum(masks, axis=0)
    idx = np.where(mask > 0)
    bb = (slice(None),) + tuple(
        slice(min_i, max_i)
        for min_i, max_i in zip(
            np.min(idx, axis=-1), np.max(idx, axis=-1)
        )
    )

    healthy_images = [im[bb] for im in healthy]
    obese_images = [im[bb] for im in obese]
    surgery_images = [im[bb] for im in surgery]

    show_slices(healthy_images, path, 'healthy')
    show_slices(obese_images, path, 'obese')
    show_slices(surgery_images, path, 'surgery')


if __name__ == '__main__':
    main()
