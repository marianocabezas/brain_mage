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

    diffs = []
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
        except TypeError:
            print(c_rows.iloc[0]['Date'], c_rows.iloc[1]['Date'], c_rows.iloc[1]['Surgerydate'])
            date_diff = None

        if date_diff is not None:
            diffs.append(fu_date - bl_date)
            if had_surgery:
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
    return surg_dict, diffs, surg_diffs


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

            try:
                nib.load(
                    os.path.join(path, 'Basal_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_coreg.nii.gz')
                )
                nib.load(
                    os.path.join(path, 'Follow_UP_IronMET_CGM', c, 'sT1W_3D_TFE_SENSE_coreg.nii.gz')
                )
            except IOError:
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


def struct_dict(structure):
    s_dict =  {
        'name': structure,
        'healthy_natrophy': [],
        'healthy_atrophy': [],
        'obese_atrophy': [],
        'obese_natrophy': [],
        'surgery_atrophy': [],
        'surgery_natrophy': [],
        'healthy_batrophy': [],
        'healthy_bnatrophy': [],
        'obese_batrophy': [],
        'obese_bnatrophy': [],
        'surgery_batrophy': [],
        'surgery_bnatrophy': [],
    }

    return s_dict


def deformable_registration(path, data_dict, scales, epochs, patience, lr):
    label_dict = {
        1: struct_dict('Left-Cerebral-Exterior'),
        2: struct_dict('Left-Cerebral-White-Matter '),
        3: struct_dict('Left-Cerebral-Cortex'),
        4: struct_dict('Left-Lateral-Ventricle'),
        5: struct_dict('Left-Inf-Lat-Vent'),
        6: struct_dict('Left-Cerebellum-Exterior'),
        7: struct_dict('Left-Cerebellum-White-Matter'),
        8: struct_dict('Left-Cerebellum-Cortex'),
        9: struct_dict('Left-Thalamus'),
        10: struct_dict('Left-Thalamus-Proper*'),
        11: struct_dict('Left-Caudate'),
        12: struct_dict('Left-Putamen'),
        13: struct_dict('Left-Pallidum'),
        14: struct_dict('3rd-Ventricle'),
        15: struct_dict('4th-Ventricle'),
        16: struct_dict('Brain-Stem'),
        17: struct_dict('Left-Hippocampus'),
        18: struct_dict('Left-Amygdala'),
        19: struct_dict('Left-Insula'),
        20: struct_dict('Left-Operculum'),
        21: struct_dict('Line-1'),
        22: struct_dict('Line-2'),
        23: struct_dict('Line-3'),
        24: struct_dict('CSF'),
        25: struct_dict('Left-Lesion'),
        26: struct_dict('Left-Accumbens-area'),
        27: struct_dict('Left-Substancia-Nigra'),
        28: struct_dict('Left-VentralDC'),
        29: struct_dict('Left-undetermined'),
        30: struct_dict('Left-vessel'),
        31: struct_dict('Left-choroid-plexus'),
        32: struct_dict('Left-F3orb'),
        33: struct_dict('Left-lOg'),
        34: struct_dict('Left-aOg'),
        35: struct_dict('Left-mOg'),
        36: struct_dict('Left-pOg'),
        37: struct_dict('Left-Stellate'),
        38: struct_dict('Left-Porg'),
        39: struct_dict('Left-Aorg'),
        40: struct_dict('Right-Cerebral-Exterior'),
        41: struct_dict('Right-Cerebral-White-Matter'),
        42: struct_dict('Right-Cerebral-Cortex'),
        43: struct_dict('Right-Lateral-Ventricle'),
        44: struct_dict('Right-Inf-Lat-Vent'),
        45: struct_dict('Right-Cerebellum-Exterior'),
        46: struct_dict('Right-Cerebellum-White-Matter'),
        47: struct_dict('Right-Cerebellum-Cortex'),
        48: struct_dict('Right-Thalamus'),
        49: struct_dict('Right-Thalamus-Proper*'),
        50: struct_dict('Right-Caudate'),
        51: struct_dict('Right-Putamen'),
        52: struct_dict('Right-Pallidum'),
        53: struct_dict('Right-Hippocampus'),
        54: struct_dict('Right-Amygdala'),
        55: struct_dict('Right-Insula'),
        56: struct_dict('Right-Operculum'),
        57: struct_dict('Right-Lesion'),
        58: struct_dict('Right-Accumbens-area'),
        59: struct_dict('Right-Substancia-Nigra'),
        60: struct_dict('Right-VentralDC'),
        1001: struct_dict('ctx-lh-bankssts'),
        1002: struct_dict('ctx-lh-caudalanteriorcingulate'),
        1003: struct_dict('ctx-lh-caudalmiddlefrontal'),
        1004: struct_dict('ctx-lh-corpuscallosum'),
        1005: struct_dict('ctx-lh-cuneus'),
        1006: struct_dict('ctx-lh-entorhinal'),
        1007: struct_dict('ctx-lh-fusiform'),
        1008: struct_dict('ctx-lh-inferiorparietal'),
        1009: struct_dict('ctx-lh-inferiortemporal'),
        1010: struct_dict('ctx-lh-isthmuscingulate'),
        1011: struct_dict('ctx-lh-lateraloccipital'),
        1012: struct_dict('ctx-lh-lateralorbitofrontal'),
        1013: struct_dict('ctx-lh-lingual'),
        1014: struct_dict('ctx-lh-medialorbitofrontal'),
        1015: struct_dict('ctx-lh-middletemporal'),
        1016: struct_dict('ctx-lh-parahippocampal'),
        1017: struct_dict('ctx-lh-paracentral'),
        1018: struct_dict('ctx-lh-parsopercularis'),
        1019: struct_dict('ctx-lh-parsorbitalis'),
        1020: struct_dict('ctx-lh-parstriangularis'),
        1021: struct_dict('ctx-lh-pericalcarine'),
        1022: struct_dict('ctx-lh-postcentral'),
        1023: struct_dict('ctx-lh-posteriorcingulate'),
        1024: struct_dict('ctx-lh-precentral'),
        1025: struct_dict('ctx-lh-precuneus'),
        1026: struct_dict('ctx-lh-rostralanteriorcingulate'),
        1027: struct_dict('ctx-lh-rostralmiddlefrontal'),
        1028: struct_dict('ctx-lh-superiorfrontal'),
        1029: struct_dict('ctx-lh-superiorparietal'),
        1030: struct_dict('ctx-lh-superiortemporal'),
        1031: struct_dict('ctx-lh-supramarginal'),
        1032: struct_dict('ctx-lh-frontalpole'),
        1033: struct_dict('ctx-lh-temporalpole'),
        1034: struct_dict('ctx-lh-transversetemporal'),
        1035: struct_dict('ctx-lh-insula'),
        2001: struct_dict('ctx-rh-bankssts'),
        2002: struct_dict('ctx-rh-caudalanteriorcingulate'),
        2003: struct_dict('ctx-rh-caudalmiddlefrontal'),
        2004: struct_dict('ctx-rh-corpuscallosum'),
        2005: struct_dict('ctx-rh-cuneus'),
        2006: struct_dict('ctx-rh-entorhinal'),
        2007: struct_dict('ctx-rh-fusiform'),
        2008: struct_dict('ctx-rh-inferiorparietal'),
        2009: struct_dict('ctx-rh-inferiortemporal'),
        2010: struct_dict('ctx-rh-isthmuscingulate'),
        2011: struct_dict('ctx-rh-lateraloccipital'),
        2012: struct_dict('ctx-rh-lateralorbitofrontal'),
        2013: struct_dict('ctx-rh-lingual'),
        2014: struct_dict('ctx-rh-medialorbitofrontal'),
        2015: struct_dict('ctx-rh-middletemporal'),
        2016: struct_dict('ctx-rh-parahippocampal'),
        2017: struct_dict('ctx-rh-paracentral'),
        2018: struct_dict('ctx-rh-parsopercularis'),
        2019: struct_dict('ctx-rh-parsorbitalis'),
        2020: struct_dict('ctx-rh-parstriangularis'),
        2021: struct_dict('ctx-rh-pericalcarine'),
        2022: struct_dict('ctx-rh-postcentral'),
        2023: struct_dict('ctx-rh-posteriorcingulate'),
        2024: struct_dict('ctx-rh-precentral'),
        2025: struct_dict('ctx-rh-precuneus'),
        2026: struct_dict('ctx-rh-rostralanteriorcingulate'),
        2027: struct_dict('ctx-rh-rostralmiddlefrontal'),
        2028: struct_dict('ctx-rh-superiorfrontal'),
        2029: struct_dict('ctx-rh-superiorparietal'),
        2030: struct_dict('ctx-rh-superiortemporal'),
        2031: struct_dict('ctx-rh-supramarginal'),
        2032: struct_dict('ctx-rh-frontalpole'),
        2033: struct_dict('ctx-rh-temporalpole'),
        2034: struct_dict('ctx-rh-transversetemporal'),
        2035: struct_dict('ctx-rh-insula'),
    }

    n_subjects = 0
    for c, c_data in data_dict.items():
        if c_data['Follow-up']['HasImage'] and c_data['Baseline']['HasImage'] and c_data['DateDiff'] is not None:
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

            t_diff = c_data['DateDiff'].years + c_data['DateDiff'].months / 12

            for i in np.unique(seg).astype(np.int32)[1:]:
                if i not in [4, 5, 14, 15, 24, 43, 44, 72]:
                    lab_mask = seg == i
                    atrophy = 100 * (np.mean(jacobian_det[lab_mask]) - 1)
                    inner_mask = binary_erosion(lab_mask, structure=np.ones((3, 3, 3)))
                    boundary_mask = np.logical_and(
                        lab_mask, np.logical_not(inner_mask)
                    )
                    bound_atrophy = 100 * (np.mean(jacobian_det[boundary_mask]) - 1)
                    if not c_data['Obese'] and not c_data['HadSurgery']:
                        label_dict[i]['healthy_atrophy'].append(atrophy)
                        label_dict[i]['healthy_batrophy'].append(bound_atrophy)
                        label_dict[i]['healthy_natrophy'].append(atrophy / t_diff)
                        label_dict[i]['healthy_bnatrophy'].append(bound_atrophy  / t_diff)
                    elif not c_data['HadSurgery']:
                        label_dict[i]['obese_atrophy'].append(atrophy)
                        label_dict[i]['obese_batrophy'].append(bound_atrophy)
                        label_dict[i]['obese_natrophy'].append(atrophy / t_diff)
                        label_dict[i]['obese_bnatrophy'].append(bound_atrophy / t_diff)
                    else:
                        label_dict[i]['surgery_atrophy'].append(atrophy)
                        label_dict[i]['surgery_batrophy'].append(bound_atrophy)
                        label_dict[i]['surgery_natrophy'].append(atrophy / t_diff)
                        label_dict[i]['surgery_bnatrophy'].append(bound_atrophy / t_diff)

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
    surg_dict, diffs, surg_diffs = get_data_dict()
    diffs_days = np.mean(diffs).days
    surg_days = np.mean(surg_diffs).days
    print('{:}Time to follow-up{:} distribution'.format(c['b'], c['nc']))
    print(
        'Mean difference: {:d} years, {:d} months and {:d} days (from baseline)'.format(
            diffs_days // 365,
            (diffs_days % 365) // 30,
            (diffs_days % 365) % 30
        ),
    )
    print(
        'Mean difference: {:d} years, {:d} months and {:d} days (from surgery)'.format(
            surg_days // 365,
            (surg_days % 365) // 30,
            (surg_days % 365) % 30
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
                'Healthy = {:4.2f}% ± {:4.2f} | Obese = {:4.2f}% ± {:4.2f} | Surgery = {:4.2f}% ± {:4.2f}'.format(
                    np.mean(atrophy_data['healthy_natrophy']), np.std(atrophy_data['healthy_natrophy']),
                    np.mean(atrophy_data['obese_natrophy']), np.std(atrophy_data['obese_natrophy']),
                    np.mean(atrophy_data['surgery_natrophy']), np.std(atrophy_data['surgery_natrophy']),
                )
            )


if __name__ == '__main__':
    main()
