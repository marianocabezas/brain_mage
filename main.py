import os
import argparse
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
        help='Number of epochs'
    )
    parser.add_argument(
        '-s', '--scales',
        dest='scales',
        nargs='+', type=int, default=[4, 2, 1],
        help='Number of epochs'
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


def get_brain_mask(image):
    th = filters.threshold_otsu(image)

    mask_erode = binary_erosion(
        image > th, structure=np.ones((3, 3, 3)),
        iterations=3
    )

    labels, n_lab = label(mask_erode, np.ones((3, 3, 3)))

    areas = [np.sum(labels == lab) for lab in range(n_lab)]
    largest_lab = np.argmax(areas[1:]) + 1

    core_brain = labels == largest_lab
    brain = binary_dilation(core_brain, structure=np.ones((3, 3, 3)), iterations=5)

    return binary_fill_holes(brain)


def image_info(path, data_dict, epochs, patience, lr):
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

            target_spacing = (0.9583333, 0.9583333, 1.0)
            target_dims = (240, 240, 145)

            bl_im = bl_nii.get_fdata()
            fu_im = fu_nii.get_fdata()

            bl_mask = get_brain_mask(bl_im)
            fu_mask = get_brain_mask(fu_im)

            bl_hdr = bl_nii.header
            fu_hdr = fu_nii.header

            bl_mask_nii = nib.Nifti1Image(bl_mask.astype(np.uint8), None, header=bl_hdr)
            bl_mask_nii.to_filename(
                os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_mask.nii.gz')
            )
            fu_mask_nii = nib.Nifti1Image(fu_mask.astype(np.uint8), None, header=fu_hdr)
            fu_mask_nii.to_filename(
                os.path.join(path, 'Follow_UP_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_mask.nii.gz')
            )

            '''affine_fu, affine_bl, _, _ = halfway_registration(
                fu_im, bl_im, fu_nii.header.get_zooms(), bl_nii.header.get_zooms(),
                mask_a=fu_mask, mask_b=bl_mask, loss_f=mse_loss,
                shape_target=target_dims, spacing_target=target_spacing,
                scales=[4, 2, 1], epochs=epochs, patience=patience
            )'''

            affine_fu, _, _ = halfway_registration(
                fu_im, bl_im, fu_nii.header.get_zooms(), bl_nii.header.get_zooms(),
                mask_a=fu_mask, mask_b=bl_mask, loss_f=mse_loss, init_lr=lr,
                scales=[2, 1], epochs=epochs, patience=patience
            )

            bl_new = resample(
                bl_im, bl_nii.header.get_zooms(),
                target_dims, target_spacing,
                torch.inverse(affine_fu)
            ).detach().cpu().numpy()
            fu_new = resample(
                fu_im, fu_nii.header.get_zooms(),
                target_dims, target_spacing,
                affine_fu
            ).detach().cpu().numpy()
            bl_hdr.set_zooms(target_spacing)
            bl_new_nii = nib.Nifti1Image(bl_new, None, header=bl_hdr)
            bl_new_nii.to_filename(
                os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_coreg.nii.gz')
            )
            fu_hdr.set_zooms(target_spacing)
            fu_new_nii = nib.Nifti1Image(fu_new, None, header=fu_hdr)
            fu_new_nii.to_filename(
                os.path.join(path, 'Follow_UP_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_coreg.nii.gz')
            )


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
    epochs = options['epochs']
    patience = options['patience']
    lr = options['learning_rate']
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

    image_info(path, surg_dict, epochs, patience, lr)


if __name__ == '__main__':
    main()
