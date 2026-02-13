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
from registration import resample, halfway_registration, nonlinear_registration
from registration import sitk_registration, mse_loss, xcor_loss, jacobian_determinant



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
        '-s', '--scales',
        dest='scales',
        nargs='+', type=int, default=[4, 2, 1],
        help='Scale pyramid sequence'
    )
    parser.add_argument(
        '-l', '--learning-rate',
        dest='learning_rate',
        type=float, default=1e-3,
        help='Number of epochs'
    )
    parser.add_argument(
        '-d', '--deformable',
        dest='deformable',
        default=False, action='store_true',
        help='Apply deformable registration'
    )
    parser.add_argument(
        '-a', '--affine',
        dest='affine',
        default=False, action='store_true',
        help='Apply affine registration'
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
        iterations=2
    )

    labels, n_lab = label(mask_erode, np.ones((3, 3, 3)))

    areas = [np.sum(labels == lab) for lab in range(n_lab)]
    largest_lab = np.argmax(areas[1:]) + 1

    core_brain = labels == largest_lab
    brain = binary_dilation(core_brain, structure=np.ones((3, 3, 3)), iterations=2)

    return binary_fill_holes(brain)


def affine_registration(path, data_dict, scales, epochs, patience, lr):
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

            out_hdr = deepcopy(fu_nii.header)

            bl_mask_nii = nib.Nifti1Image(bl_mask.astype(np.uint8), None, header=out_hdr)
            bl_mask_nii.to_filename(
                os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_mask.nii.gz')
            )
            fu_mask_nii = nib.Nifti1Image(fu_mask.astype(np.uint8), None, header=out_hdr)
            fu_mask_nii.to_filename(
                os.path.join(path, 'Follow_UP_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_mask.nii.gz')
            )

            '''affine_fu, affine_bl, _, _ = halfway_registration(
                fu_im, bl_im, fu_nii.header.get_zooms(), bl_nii.header.get_zooms(),
                mask_a=fu_mask, mask_b=bl_mask, loss_f=mse_loss,
                shape_target=target_dims, spacing_target=target_spacing,
                scales=[4, 2, 1], epochs=epochs, patience=patience
            )'''

            out_hdr.set_zooms(target_spacing)

            # Init resample
            bl_init = resample(
                bl_im, bl_nii.header.get_zooms(),
                target_dims, target_spacing,
                torch.eye(4, dtype=torch.float64)
            ).detach().cpu().numpy()
            fu_init = resample(
                fu_im, fu_nii.header.get_zooms(),
                target_dims, target_spacing,
                torch.eye(4, dtype=torch.float64)
            ).detach().cpu().numpy()
            bl_new_nii = nib.Nifti1Image(bl_init, None, header=out_hdr)
            bl_new_nii.to_filename(
                os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_init.nii.gz')
            )
            fu_new_nii = nib.Nifti1Image(fu_init, None, header=out_hdr)
            fu_new_nii.to_filename(
                os.path.join(path, 'Follow_UP_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_init.nii.gz')
            )

            affine_fu, _, _ = halfway_registration(
                fu_im, bl_im, fu_nii.header.get_zooms(), bl_nii.header.get_zooms(),
                mask_a=fu_mask, mask_b=bl_mask, loss_f=mse_loss, init_lr=lr,
                scales=scales, epochs=epochs, patience=patience,
                shape_target=bl_nii.shape, spacing_target=bl_nii.header.get_zooms(),
            )

            # Final resample
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
            bl_new_nii = nib.Nifti1Image(bl_new, None, header=out_hdr)
            bl_new_nii.to_filename(
                os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_coreg.nii.gz')
            )
            fu_new_nii = nib.Nifti1Image(fu_new, None, header=out_hdr)
            fu_new_nii.to_filename(
                os.path.join(path, 'Follow_UP_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_coreg.nii.gz')
            )


def deformable_registration(path, data_dict, scales, epochs, patience, lr):
    label_dict = {
        1: {
            'name': 'Left-Cerebral-Exterior',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2: {
            'name': 'Left-Cerebral-White-Matter ',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        3: {
            'name': 'Left-Cerebral-Cortex',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        4: {
            'name': 'Left-Lateral-Ventricle',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        5: {
            'name': 'Left-Inf-Lat-Vent',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        6: {
            'name': 'Left-Cerebellum-Exterior',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        7: {
            'name': 'Left-Cerebellum-White-Matter',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        8: {
            'name': 'Left-Cerebellum-Cortex',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        9: {
            'name': 'Left-Thalamus',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        10: {
            'name': 'Left-Thalamus-Proper*',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        11: {
            'name': 'Left-Caudate',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        12: {
            'name': 'Left-Putamen',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        13: {
            'name': 'Left-Pallidum',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        14: {
            'name': '3rd-Ventricle',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        15: {
            'name': '4th-Ventricle',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        16: {
            'name': 'Brain-Stem',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        17: {
            'name': 'Left-Hippocampus',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        18: {
            'name': 'Left-Amygdala',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        19: {
            'name': 'Left-Insula',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        20: {
            'name': 'Left-Operculum',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        21: {
            'name': 'Line-1',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        22: {
            'name': 'Line-2',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        23: {
            'name': 'Line-3',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        24: {
            'name': 'CSF',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        25: {
            'name': 'Left-Lesion',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        26: {
            'name': 'Left-Accumbens-area',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        27: {
            'name': 'Left-Substancia-Nigra',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        28: {
            'name': 'Left-VentralDC',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        29: {
            'name': 'Left-undetermined',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        30: {
            'name': 'Left-vessel',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        31: {
            'name': 'Left-choroid-plexus',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        32: {
            'name': 'Left-F3orb',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        33: {
            'name': 'Left-lOg',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        34: {
            'name': 'Left-aOg',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        35: {
            'name': 'Left-mOg',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        36: {
            'name': 'Left-pOg',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        37: {
            'name': 'Left-Stellate',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        38: {
            'name': 'Left-Porg',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        39: {
            'name': 'Left-Aorg',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        40: {
            'name': 'Right-Cerebral-Exterior',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        41: {
            'name': 'Right-Cerebral-White-Matter',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        42: {
            'name': 'Right-Cerebral-Cortex',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        43: {
            'name': 'Right-Lateral-Ventricle',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        44: {
            'name': 'Right-Inf-Lat-Vent',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        45: {
            'name': 'Right-Cerebellum-Exterior',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        46: {
            'name': 'Right-Cerebellum-White-Matter',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        47: {
            'name': 'Right-Cerebellum-Cortex',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        48: {
            'name': 'Right-Thalamus',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        49: {
            'name': 'Right-Thalamus-Proper*',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        50: {
            'name': 'Right-Caudate',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        51: {
            'name': 'Right-Putamen',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        52: {
            'name': 'Right-Pallidum',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        53: {
            'name': 'Right-Hippocampus',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        54: {
            'name': 'Right-Amygdala',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        55: {
            'name': 'Right-Insula',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        56: {
            'name': 'Right-Operculum',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        57: {
            'name': 'Right-Lesion',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        58: {
            'name': 'Right-Accumbens-area',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        59: {
            'name': 'Right-Substancia-Nigra',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        60: {
            'name': 'Right-VentralDC',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1001: {
            'name': 'ctx-lh-bankssts',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1002: {
            'name': 'ctx-lh-caudalanteriorcingulate',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1003: {
            'name': 'ctx-lh-caudalmiddlefrontal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1004: {
            'name': 'ctx-lh-corpuscallosum',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1005: {
            'name': 'ctx-lh-cuneus',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1006: {
            'name': 'ctx-lh-entorhinal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1007: {
            'name': 'ctx-lh-fusiform',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1008: {
            'name': 'ctx-lh-inferiorparietal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1009: {
            'name': 'ctx-lh-inferiortemporal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1010: {
            'name': 'ctx-lh-isthmuscingulate',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1011: {
            'name': 'ctx-lh-lateraloccipital',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1012: {
            'name': 'ctx-lh-lateralorbitofrontal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1013: {
            'name': 'ctx-lh-lingual',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1014: {
            'name': 'ctx-lh-medialorbitofrontal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1015: {
            'name': 'ctx-lh-middletemporal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1016: {
            'name': 'ctx-lh-parahippocampal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1017: {
            'name': 'ctx-lh-paracentral',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1018: {
            'name': 'ctx-lh-parsopercularis',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1019: {
            'name': 'ctx-lh-parsorbitalis',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1020: {
            'name': 'ctx-lh-parstriangularis',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1021: {
            'name': 'ctx-lh-pericalcarine',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1022: {
            'name': 'ctx-lh-postcentral',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1023: {
            'name': 'ctx-lh-posteriorcingulate',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1024: {
            'name': 'ctx-lh-precentral',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1025: {
            'name': 'ctx-lh-precuneus',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1026: {
            'name': 'ctx-lh-rostralanteriorcingulate',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1027: {
            'name': 'ctx-lh-rostralmiddlefrontal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1028: {
            'name': 'ctx-lh-superiorfrontal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1029: {
            'name': 'ctx-lh-superiorparietal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1030: {
            'name': 'ctx-lh-superiortemporal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1031: {
            'name': 'ctx-lh-supramarginal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1032: {
            'name': 'ctx-lh-frontalpole',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1033: {
            'name': 'ctx-lh-temporalpole',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1034: {
            'name': 'ctx-lh-transversetemporal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        1035: {
            'name': 'ctx-lh-insula',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2001: {
            'name': 'ctx-rh-bankssts',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2002: {
            'name': 'ctx-rh-caudalanteriorcingulate',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2003: {
            'name': 'ctx-rh-caudalmiddlefrontal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2004: {
            'name': 'ctx-rh-corpuscallosum',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2005: {
            'name': 'ctx-rh-cuneus',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2006: {
            'name': 'ctx-rh-entorhinal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2007: {
            'name': 'ctx-rh-fusiform',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2008: {
            'name': 'ctx-rh-inferiorparietal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2009: {
            'name': 'ctx-rh-inferiortemporal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2010: {
            'name': 'ctx-rh-isthmuscingulate',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2011: {
            'name': 'ctx-rh-lateraloccipital',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2012: {
            'name': 'ctx-rh-lateralorbitofrontal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2013: {
            'name': 'ctx-rh-lingual',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2014: {
            'name': 'ctx-rh-medialorbitofrontal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2015: {
            'name': 'ctx-rh-middletemporal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2016: {
            'name': 'ctx-rh-parahippocampal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2017: {
            'name': 'ctx-rh-paracentral',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2018: {
            'name': 'ctx-rh-parsopercularis',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2019: {
            'name': 'ctx-rh-parsorbitalis',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2020: {
            'name': 'ctx-rh-parstriangularis',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2021: {
            'name': 'ctx-rh-pericalcarine',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2022: {
            'name': 'ctx-rh-postcentral',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2023: {
            'name': 'ctx-rh-posteriorcingulate',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2024: {
            'name': 'ctx-rh-precentral',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2025: {
            'name': 'ctx-rh-precuneus',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2026: {
            'name': 'ctx-rh-rostralanteriorcingulate',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2027: {
            'name': 'ctx-rh-rostralmiddlefrontal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2028: {
            'name': 'ctx-rh-superiorfrontal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2029: {
            'name': 'ctx-rh-superiorparietal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2030: {
            'name': 'ctx-rh-superiortemporal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2031: {
            'name': 'ctx-rh-supramarginal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2032: {
            'name': 'ctx-rh-frontalpole',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2033: {
            'name': 'ctx-rh-temporalpole',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2034: {
            'name': 'ctx-rh-transversetemporal',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
        2035: {
            'name': 'ctx-rh-insula',
            'healthy_atrophy': [],
            'obese_atrophy': [],
            'surgery_atrophy': [],
            'healthy_batrophy': [],
            'obese_batrophy': [],
            'surgery_batrophy': [],
        },
    }

    n_subjects = 0
    for c, c_data in data_dict.items():
        if c_data['Follow-up']['HasImage'] and c_data['Baseline']['HasImage']:
            n_subjects += 1
            bl_nii = nib.load(
                os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_coreg.nii.gz')
            )
            fu_nii = nib.load(
                os.path.join(path, 'Follow_UP_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_coreg.nii.gz')
            )

            bl_im = bl_nii.get_fdata()
            fu_im = fu_nii.get_fdata()
            bl_x, bl_y, bl_z = bl_nii.get_fdata().shape
            bl_sx, bl_sy, bl_sz = bl_nii.header.get_zooms()
            fu_x, fu_y, fu_z = fu_nii.get_fdata().shape
            fu_sx, fu_sy, fu_sz = fu_nii.header.get_zooms()

            print(
                'Subject {:} - Baseline {:3d} x {:3d} x {:3d} ({:4.2f} x {:4.2f} x {:4.2f})'.format(
                    c, bl_x, bl_y, bl_z, bl_sx, bl_sy, bl_sz
                ), end=' '
            )
            print(
                '- Follow-up {:3d} x {:3d} x {:3d} ({:4.2f} x {:4.2f} x {:4.2f})'.format(
                    fu_x, fu_y, fu_z, fu_sx, fu_sy, fu_sz
                )
            )

            try:
                bl_mask_nii = nib.load(
                    os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_coreg_mask.nii.gz')
                )
                bl_mask = bl_mask_nii.get_fdata() > 0
            except IOError:
                bl_mask = get_brain_mask(bl_im)
            try:
                fu_mask_nii = nib.load(
                    os.path.join(path, 'Follow_UP_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_coreg_mask.nii.gz')
                )
                fu_mask = fu_mask_nii.get_fdata() > 0
            except IOError:
                fu_mask = get_brain_mask(fu_im)

            out_hdr = deepcopy(fu_nii.header)

            try:
                df_nii = nib.load(
                    os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_df.nii.gz')
                )
                df_numpy = df_nii.get_fdata()
            except IOError:
                df, _, _, log = nonlinear_registration(
                    bl_im, fu_im, bl_mask, fu_mask, loss_f=mse_loss, init_lr=lr,
                    scales=scales, epochs=epochs, patience=patience
                )

                moved = resample(
                    bl_im, bl_nii.header.get_zooms(),
                    bl_im.shape, bl_nii.header.get_zooms(),
                    torch.eye(4, dtype=torch.float64), df
                ).detach().cpu().numpy()

                moved_nii = nib.Nifti1Image(moved, None, header=out_hdr)
                moved_nii.to_filename(
                    os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_warped.nii.gz')
                )

                df_numpy = np.moveaxis(df.detach().cpu().numpy(), 0,-1)
                df_nii = nib.Nifti1Image(df_numpy, None, header=out_hdr)
                df_nii.to_filename(
                    os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_df.nii.gz')
                )

            jacobian_det = jacobian_determinant(df_numpy, (fu_sx, fu_sy, fu_sz))
            jacobian_nii = nib.Nifti1Image(jacobian_det, None, header=out_hdr)
            jacobian_nii.to_filename(
                os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_jacobian.nii.gz')
            )

            print(
                'Subject {:} - Jacobian mean value (< 1 ~ atrophy) = {:6.4f}'.format(
                    c, np.mean(jacobian_det[fu_mask])
                )
            )

            seg_nii = nib.load(
                os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_coreg_seg.nii.gz')
            )
            seg = seg_nii.get_fdata()

            for i in np.unique(seg).astype(np.int32)[1:]:
                if i != 4 and i != 5 and i != 14 and i != 15 and i != 43 and i != 44 and i != 72:
                    lab_mask = seg == i
                    atrophy = np.mean(jacobian_det[lab_mask])
                    inner_mask = binary_erosion(lab_mask, structure=np.ones((3, 3, 3)))
                    boundary_mask = np.logical_and(
                        lab_mask, np.logical_not(inner_mask)
                    )
                    bound_atrophy = np.mean(jacobian_det[boundary_mask])
                    if not c_data['Obese'] and not c_data['HadSurgery']:
                        label_dict[i]['healthy_atrophy'].append(atrophy)
                        label_dict[i]['healthy_batrophy'].append(bound_atrophy)
                    elif not c_data['HadSurgery']:
                        label_dict[i]['obese_atrophy'].append(atrophy)
                        label_dict[i]['obese_batrophy'].append(bound_atrophy)
                    else:
                        label_dict[i]['surgery_atrophy'].append(atrophy)
                        label_dict[i]['surgery_batrophy'].append(bound_atrophy)

            print('-'.join([''] * 100))

    final_dict = {
        s: s_data
        for s, s_data in label_dict.items()
        if (len(s_data['healthy_atrophy'] + s_data['obese_atrophy'] + s_data['surgery_atrophy'])) == n_subjects
    }
    return final_dict


def mage_info(data_dict):
    notobese_nosurg_mage_fu = []
    obese_nosurg_mage_fu = []
    surg_mage_fu = []
    notobese_nosurg_mage_bl = []
    obese_nosurg_mage_bl = []
    surg_mage_bl = []
    notobese_nosurg_diffmage = []
    obese_nosurg_diffmage = []
    surg_diffmage = []
    for c, c_data in data_dict.items():
        if c_data['Follow-up']['HasImage'] and not c_data['Obese'] and not c_data['HadSurgery']:
            notobese_nosurg_mage_fu.append(
                c_data['Follow-up']['MAGE']
            )
            notobese_nosurg_mage_bl.append(
                c_data['Baseline']['MAGE']
            )
            notobese_nosurg_diffmage.append(
                c_data['Follow-up']['MAGE'] - c_data['Baseline']['MAGE']
            )
        elif c_data['Follow-up']['HasImage'] and c_data['Obese'] and not c_data['HadSurgery']:
            obese_nosurg_mage_fu.append(
                c_data['Follow-up']['MAGE']
            )
            obese_nosurg_mage_bl.append(
                c_data['Baseline']['MAGE']
            )
            obese_nosurg_diffmage.append(
                c_data['Follow-up']['MAGE'] - c_data['Baseline']['MAGE']
            )
        elif c_data['Follow-up']['HasImage'] and c_data['HadSurgery']:
            surg_mage_fu.append(
                c_data['Follow-up']['MAGE']
            )
            surg_mage_bl.append(
                c_data['Baseline']['MAGE']
            )
            surg_diffmage.append(
                c_data['Follow-up']['MAGE'] - c_data['Baseline']['MAGE']
            )

    n_fu = len(notobese_nosurg_mage_fu) + len(obese_nosurg_mage_fu) + len(surg_mage_fu)


    print(
        'Not obese (no surgery)', len(notobese_nosurg_mage_fu),
        '{:>5.2f}%'.format(100 * len(notobese_nosurg_mage_fu) / n_fu),
        '{:>5.2f}'.format(np.mean(notobese_nosurg_mage_fu)),
        '{:>6.2f}±{:>5.2f}'.format(
            np.mean(notobese_nosurg_diffmage),
            np.std(notobese_nosurg_diffmage)
        ),
        '|'
    )
    print(
        'Obese (no surgery)    ', len(obese_nosurg_mage_fu),
        '{:>5.2f}%'.format(100 * len(obese_nosurg_mage_fu) / n_fu),
        '{:>5.2f}'.format(np.mean(obese_nosurg_mage_fu)),
        '{:>6.2f}±{:>5.2f}'.format(
            np.mean(obese_nosurg_diffmage),
            np.std(notobese_nosurg_diffmage)
        ),
        '|'
    )
    print(
        'Surgery               ', len(surg_mage_fu),
        '{:>5.2f}%'.format(100 * len(surg_mage_fu) / n_fu),
        '{:>5.2f}'.format(np.mean(surg_mage_fu)),
        '{:>6.2f}±{:>5.2f}'.format(
            np.mean(surg_diffmage),
            np.std(surg_diffmage)
        ),
        '|'
    )



def main():
    # Init
    c = color_codes()
    options = parse_inputs()
    path = options['path']
    epochs = options['epochs']
    patience = options['patience']
    lr = options['learning_rate']
    scales = options['scales']
    surg_dict, deltas, diffs, surg_deltas, surg_diffs = get_data_dict()

    print(np.mean(deltas), np.mean(diffs))
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

    print('-'.join([''] * 100))

    print('{:}MAGE{:} statistics'.format(c['b'], c['nc']), ''.join([' '] * 35), '|')
    mage_info(surg_dict)

    if parse_inputs()['affine']:
        print('-'.join([''] * 100))
        print('{:}Affine{:} registration'.format(c['b'], c['nc']))
        affine_registration(path, surg_dict, scales, epochs, patience, lr)

    if parse_inputs()['deformable']:
        print('-'.join([''] * 100))
        print('{:}Deformable{:} registration'.format(c['b'], c['nc']))
        atrophy_dict = deformable_registration(path, surg_dict, scales, epochs, patience, lr)
        for region, atrophy_data in atrophy_dict.items():
            print(
                'Region {:<20d}:'.format(region),
                'Healthy = {:4.2f} ± {:4.2f} | Obese = {:4.2f} ± {:4.2f} | Surgery = {:4.2f} ± {:4.2f}'.format(
                    np.mean(atrophy_data['healthy_atrophy']), np.std(atrophy_data['healthy_atrophy']),
                    np.mean(atrophy_data['obese_atrophy']), np.std(atrophy_data['obese_atrophy']),
                    np.mean(atrophy_data['surgery_atrophy']), np.std(atrophy_data['surgery_atrophy']),
                )
            )


if __name__ == '__main__':
    main()
