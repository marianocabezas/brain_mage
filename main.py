import os
import time
import argparse
from copy import deepcopy
import numpy as np
import pandas as pd
import nibabel as nib
from skimage import filters
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes, label
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from utils import color_codes, time_to_string
from datasets import LongitudinalDataset
from models import FeatureNet, ClassifierNet


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
        dest='path', default='/home/mariano/IronMET_CGM',
        help='Path to the files (imaging and tabular data).'
    )
    parser.add_argument(
        '-m', '--model-path',
        dest='model_path', default='/home/mariano/IronMET_CGM',
        help='Path to the model files.'
    )
    parser.add_argument(
        '-n', '--number-images',
        dest='n_images',
        type=int, default=2,
        help='Number of images per batch'
    )
    parser.add_argument(
        '-c', '--conv-filters',
        dest='conv_filters',
        nargs='+', type=int, default=[32, 64, 128, 256, 512],
        help='Number of filters per convolutional layer'
    )
    parser.add_argument(
        '-b', '--batch-size',
        dest='batch_size',
        type=int, default=2,
        help='Number of images per batch'
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
    parser.add_argument(
        '-f', '--folds',
        dest='folds',
        type=int, default=5,
        help='Number of folds for cross-validation'
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


def split_data(image_list, idx_list, i, test_length, trainval_split):
    test_ini = np.round(i * test_length).astype(int)
    test_out = np.round((i + 1) * test_length).astype(int)
    test = idx_list[test_ini:test_out]
    trainval = idx_list[:test_ini] + idx_list[test_out:]
    val = idx_list[:int(trainval_split * len(trainval))]
    train = idx_list[int(trainval_split * len(trainval)):]

    im_train = [im for im in np.stack(image_list)[train]]
    im_val = [im for im in np.stack(image_list)[val]]
    im_test = [im for im in np.stack(image_list)[test]]

    return im_train, im_val, im_test


def train_net(
    net, model_name, train_dataset, validation_dataset, verbose=1
):
    """
        Function that trains a CNN with train_dataset and validates it
        with validation_dataset.
        :param net:
        :param model_name:
        :param train_dataset:
        :param validation_dataset:
        :param verbose: Verbosity level.
        :return: None.
        """

    # Init
    c = color_codes()

    epochs = parse_inputs()['epochs']
    patience = parse_inputs()['patience']
    n_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )

    model_path = os.path.join(parse_inputs()['model_path'], 'weights')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    try:
        net.load_model(os.path.join(model_path, model_name))
        print(
            '{:}Network loaded{:} ({:d} parameters)'.format(
                c['c'], c['nc'], n_params
            )
        )
    except IOError:
        if verbose > 0:
            print(
                '{:}Starting training{:} ({:d} parameters)'.format(
                    c['c'], c['nc'], n_params
                )
            )

        # Datasets / Dataloaders should be added here
        if verbose > 1:
            print('Preparing the training datasets / dataloaders')
        batch_size = parse_inputs()['batch_size']
        num_workers = batch_size * 2

        print(
            '{:}Loading the {:}training{:} data ({:03d} subjects)'.format(
                c['clr'], c['b'], c['nc'], len(train_dataset)
            )
        )
        print(
            '{:}Loading the {:}validation{:} data ({:03d} subjects)'.format(
                c['clr'], c['b'], c['nc'], len(validation_dataset)
            )
        )

        if verbose > 1:
            print('{:}Training dataset (with validation)'.format(c['clr']))
        train_dataloader = DataLoader(
            train_dataset, batch_size, True, num_workers=num_workers
        )

        if verbose > 1:
            print('{:}Validation dataset (with validation)'.format(c['clr']))
        val_dataloader = DataLoader(
            validation_dataset, 4 * batch_size, num_workers=num_workers
        )

        training_start = time.time()

        net.fit(
            train_dataloader,
            val_dataloader,
            epochs=epochs,
            patience=patience
        )

        if verbose > 0:
            time_str = time.strftime(
                '%H hours %M minutes %S seconds',
                time.gmtime(time.time() - training_start)
            )
            print(
                '{:}Training finished{:} (total time {:})'.format(
                    c['r'], c['nc'], time_str
                )
            )

    net.save_model(os.path.join(model_path, model_name))

def test_net(net, test_dataset, verbose=1):
    """
    Function that tests a CNN with test_dataset and
    returns classification metrics.
    :param net:
    :param test_dataset:
    :param verbose:
    :return:
    """
    # Init
    c = color_codes()

    batch_size = parse_inputs()['batch_size']
    num_workers = batch_size * 2

    print(
        '{:}Loading the {:}testing{:} data ({:03d} subjects)'.format(
            c['clr'], c['b'], c['nc'], len(test_dataset)
        )
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size, True, num_workers=num_workers
    )

    test_start = time.time()
    tests = len(test_dataloader)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    pos_pr_list = []
    neg_pr_list = []
    for i, (x, y) in enumerate(test_dataloader):
        test_elapsed = time.time() - test_start
        test_eta = tests * test_elapsed / (i + 1)
        print(
            '{:}Testing batch {:d}/{:d} {:} (ETA {:})'.format(
                c['clr'], i + 1, tests,
                time_to_string(test_elapsed),
                time_to_string(test_eta),
            ),
            end='\r'
        )

        pred = net(x.to(net.device)).cpu().detach()

        pred_y = torch.sigmoid(pred)
        tp += torch.logical_and(y == 1, pred_y >= 0.5).sum()
        fp += torch.logical_and(y == 0, pred_y >= 0.5).sum()
        tn += torch.logical_and(y == 0, pred_y < 0.5).sum()
        fn += torch.logical_and(y == 1, pred_y < 0.5).sum()
        pos_pr_list += pred_y[pred_y >= 0.5].numpy().tolist()
        neg_pr_list += pred_y[pred_y < 0.5].numpy().tolist()

    if verbose > 0:
        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - test_start)
        )
        print(
            '{:}Testing finished{:} (total time {:})'.format(
                c['clr'] + c['r'], c['nc'], time_str
            )
        )

    return tp, fp, tn, fn, pos_pr_list, neg_pr_list


def main():
    # Init
    options = parse_inputs()
    path = options['path']
    n_images = options['n_images']
    folds = options['folds']

    trainval_split = 0.2
    conv_filters = options['conv_filters']

    c = color_codes()

    bl_path = os.path.join(path, 'Basal_IronMET_CGM')
    fu_path = os.path.join(path, 'Follow_UP_IronMET_CGM')
    csv_file = os.path.join(path, 'data.IRMCGM.vicorob.csv')
    ironmet_data = pd.read_csv(csv_file)

    baseline_codes = os.listdir(bl_path)
    followup_codes = os.listdir(fu_path)

    patient_codes = np.unique(baseline_codes + followup_codes).tolist()

    masks = []
    healthy = []
    obese = []
    surgery = []
    mage_healthy_bl = []
    mage_healthy_fu = []
    mage_obese_bl = []
    mage_obese_fu = []
    mage_surgery_bl = []
    mage_surgery_fu = []
    mage_diff_high = 0
    mage_diff_low = 0
    bmi_healthy_bl = []
    bmi_healthy_fu = []
    bmi_obese_bl = []
    bmi_obese_fu = []
    bmi_surgery_bl = []
    bmi_surgery_fu = []
    for p in patient_codes:
        if p in baseline_codes and p in followup_codes:

            pd_idx = ironmet_data['ID'].str.contains(p)
            p_rows = ironmet_data[pd_idx]

            bmi_bl = p_rows.iloc[0]['BMI'].tolist()
            bmi_fu = p_rows.iloc[1]['BMI'].tolist()

            had_surgery = ironmet_data[pd_idx].iloc[0]['Surgery'].tolist() > 0

            is_obese = p_rows.iloc[0]['Obesity'].tolist() > 0

            mage_bl = p_rows.iloc[0]['MAGE']
            mage_fu = p_rows.iloc[1]['MAGE']
            diff_mage = mage_fu - mage_bl
            if diff_mage > 20:
                diff_mage_s = '\033[31m{:>6.2f}\033[0m'.format(diff_mage)
                mage_diff_high += 1
            elif diff_mage < 0:
                mage_diff_low += 1
                diff_mage_s = '\033[32m{:>6.2f}\033[0m'.format(diff_mage)
            else:
                diff_mage_s = '{:>6.2f}'.format(diff_mage)

            print(
                'Subject {:} - Baseline | BMI = {:}{:} / {:} | MAGE = {:>5.2f} / {:>6.2f} / {:}{:}'.format(
                    p,
                    '\033[31m{:>5.2f}\033[0m'.format(bmi_bl) if bmi_bl > 30 else '{:>5.2f}'.format(bmi_bl),
                    ' (obese) ' if is_obese else '         ',
                    '\033[31m{:>5.2f}\033[0m'.format(bmi_fu) if bmi_fu > 30 else '{:>5.2f}'.format(bmi_fu),
                    mage_bl, mage_fu, diff_mage_s,
                    ' | Surgery' if had_surgery else ''
                )
            )

            bl_filename = os.path.join(bl_path, p, 'sT1W_3D_TFE_SENSE_coreg.nii.gz')
            bl_mask_filename = os.path.join(bl_path, p, 'sT1W_3D_TFE_SENSE_coreg_mask.nii.gz')
            baseline = nib.load(bl_filename).get_fdata()
            bl_mask = nib.load(bl_mask_filename).get_fdata().astype(bool)

            fu_filename = os.path.join(fu_path, p, 'sT1W_3D_TFE_SENSE_coreg.nii.gz')
            fu_mask_filename = os.path.join(fu_path, p, 'sT1W_3D_TFE_SENSE_coreg_mask.nii.gz')
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
                mage_healthy_bl.append(mage_bl)
                mage_healthy_fu.append(mage_fu)
                bmi_healthy_bl.append(bmi_bl)
                bmi_healthy_fu.append(bmi_fu)
            elif not had_surgery:
                obese.append(np.stack([baseline_masked, followup_masked], axis=0))
                mage_obese_bl.append(mage_bl)
                mage_obese_fu.append(mage_fu)
                bmi_obese_bl.append(bmi_bl)
                bmi_obese_fu.append(bmi_fu)
            else:
                surgery.append(np.stack([baseline_masked, followup_masked], axis=0))
                mage_surgery_bl.append(mage_bl)
                mage_surgery_fu.append(mage_fu)
                bmi_surgery_bl.append(bmi_bl)
                bmi_surgery_fu.append(bmi_fu)
    print(
        '{:d} subjects | {:d} healthy | {:d} obese | {:d} surgery | {:d} MAGE diff > 20 | {:d} MAGE diff -'.format(
            len(masks), len(healthy), len(obese), len(surgery), mage_diff_high, mage_diff_low
        )
    )
    print(
        'MAGE healthy: {:5.2f} ± {:5.2f} / {:5.2f} ± {:5.2f}'.format(
            np.mean(mage_healthy_bl), np.std(mage_healthy_bl),
            np.mean(mage_healthy_fu), np.std(mage_healthy_fu)
        ), '|',
        'BMI healthy: {:5.2f} ± {:5.2f} / {:5.2f} ± {:5.2f}'.format(
            np.mean(bmi_healthy_bl), np.std(bmi_healthy_bl),
            np.mean(bmi_healthy_fu), np.std(bmi_healthy_fu)
        )
    )
    print(
        'MAGE obese:   {:5.2f} ± {:5.2f} / {:5.2f} ± {:5.2f}'.format(
            np.mean(mage_obese_bl), np.std(mage_obese_bl),
            np.mean(mage_obese_fu), np.std(mage_obese_fu)
        ), '|',
        'BMI obese:   {:5.2f} ± {:5.2f} / {:5.2f} ± {:5.2f}'.format(
            np.mean(bmi_obese_bl), np.std(bmi_obese_bl),
            np.mean(bmi_obese_fu), np.std(bmi_obese_fu)
        )
    )
    print(
        'MAGE surgery: {:5.2f} ± {:5.2f} / {:5.2f} ± {:5.2f}'.format(
            np.mean(mage_surgery_bl), np.std(mage_surgery_bl),
            np.mean(mage_surgery_fu), np.std(mage_surgery_fu)
        ), '|',
        'BMI surgery: {:5.2f} ± {:5.2f} / {:5.2f} ± {:5.2f}'.format(
            np.mean(bmi_surgery_bl), np.std(bmi_surgery_bl),
            np.mean(bmi_surgery_fu), np.std(bmi_surgery_fu)
        )
    )

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

    healthy_idxs = np.random.permutation(len(healthy_images)).tolist()
    obese_idxs = np.random.permutation(len(obese_images)).tolist()
    surgery_idxs = np.random.permutation(len(surgery_images)).tolist()

    healthy_slots = len(healthy_idxs) / folds
    obese_slots = len(obese_idxs) / folds
    surgery_slots = len(surgery_idxs) / folds

    f_string = ''.join(['c{:}'.format(f) for f in conv_filters])

    t_tp_fr = 0
    t_fp_fr = 0
    t_tn_fr = 0
    t_fn_fr = 0
    t_ppr_fr = []
    t_npr_fr = []

    t_tp_un = 0
    t_fp_un = 0
    t_tn_un = 0
    t_fn_un = 0
    t_ppr_un = []
    t_npr_un = []

    t_tp_sc = 0
    t_fp_sc = 0
    t_tn_sc = 0
    t_fn_sc = 0
    t_ppr_sc = []
    t_npr_sc = []

    t_bacc_fr = 0
    t_bacc_un = 0
    t_bacc_sc = 0
    t_acc_fr = 0
    t_acc_un = 0
    t_acc_sc = 0
    for i in range(folds):
        print(
            '{:}Fold {:}{:2d}/{:2d}{:} (n-folds cross-val)'.format(
                c['clr'] + c['c'], c['g'], i + 1, folds, c['nc']
            )
        )

        # Data split
        healthy_train, healthy_val, healthy_test = split_data(
            healthy_images, healthy_idxs, i, healthy_slots, trainval_split
        )
        obese_train, obese_val, obese_test = split_data(
            obese_images, obese_idxs, i, obese_slots, trainval_split
        )
        surgery_train, surgery_val, surgery_test = split_data(
            surgery_images, surgery_idxs, i, surgery_slots, trainval_split
        )

        test_data = healthy_test + obese_test + surgery_test
        val_data = healthy_val + obese_val + surgery_val
        train_data = healthy_train + obese_train + surgery_train
        train_labels = [0] * len(healthy_train) + [0] * len(obese_train) + [1] * len(surgery_train)
        val_labels = [0] * len(healthy_val) + [0] * len(obese_val) + [1] * len(surgery_val)
        test_labels = [0] * len(healthy_test) + [0] * len(obese_test) + [1] * len(surgery_test)

        # Dataset objects
        train_ds = LongitudinalDataset(train_data, train_labels)
        val_ds = LongitudinalDataset(val_data, val_labels)
        test_ds = LongitudinalDataset(test_data, test_labels)

        # Contrastive pre-training (can include self-supervision)
        print(
            'Training with {:}contrastive{:} learning'.format(
                c['b'], c['nc']
            )
        )
        net = FeatureNet(conv_filters=conv_filters, n_images=n_images)
        train_net(net, 'feature-net_{:}_n{:d}.pt'.format(f_string, i), train_ds, val_ds)

        # Using pre-trained weights but frozen features
        print(
            'Fine tuning with {:}frozen{:} feature layers'.format(
                c['b'], c['nc']
            )
        )
        classifier = ClassifierNet(conv_filters=conv_filters, n_images=n_images)
        classifier.encoder = deepcopy(net.encoder)
        classifier.encoder.freeze()
        train_net(classifier, 'class-frozen-net_{:}_n{:d}.pt'.format(f_string, i), train_ds, val_ds)
        tp_fr, fp_fr, tn_fr, fn_fr, ppr_fr, npr_fr = test_net(classifier, test_ds)
        t_tp_fr += tp_fr
        t_fp_fr += fp_fr
        t_tn_fr += tn_fr
        t_fn_fr += fn_fr
        t_ppr_fr += ppr_fr
        t_npr_fr += npr_fr
        bacc_fr = 0.5 * (tp_fr / (tp_fr + fn_fr) + tn_fr / (fp_fr + tn_fr))
        acc_fr = (tp_fr + tn_fr) / (tp_fr + tn_fr + fp_fr + fn_fr)
        pr_fr = tp_fr / (tp_fr + fn_fr)
        re_fr = tp_fr / (tp_fr + fp_fr)
        t_bacc_fr += bacc_fr
        t_acc_fr += acc_fr

        # Using pre-trained weights unfrozen
        print(
            'Fine tuning with {:}unfrozen{:} feature layers'.format(
                c['b'], c['nc']
            )
        )
        classifier = ClassifierNet(conv_filters=conv_filters, n_images=n_images)
        classifier.encoder = deepcopy(net.encoder)
        train_net(classifier, 'class-unfrozen-net_{:}_n{:d}.pt'.format(f_string, i), train_ds, val_ds)
        tp_un, fp_un, tn_un, fn_un, ppr_un, npr_un = test_net(classifier, test_ds)
        t_tp_un += tp_un
        t_fp_un += fp_un
        t_tn_un += tn_un
        t_fn_un += fn_un
        t_ppr_un += ppr_un
        t_npr_un += npr_un
        bacc_un = 0.5 * (tp_un / (tp_un + fn_un) + tn_un / (fp_un + tn_un))
        acc_un = (tp_un + tn_un) / (tp_un + tn_un + fp_un + fn_un)
        pr_un = tp_un / (tp_un + fn_un)
        re_un = tp_un / (tp_un + fp_un)
        t_bacc_un += bacc_un
        t_acc_un += acc_un

        # Training from scratch
        print(
            'Training {:}from scratch{:}'.format(
                c['b'], c['nc']
            )
        )
        classifier = ClassifierNet(conv_filters=conv_filters, n_images=n_images)
        train_net(classifier, 'class-net_{:}_n{:d}.pt'.format(f_string, i), train_ds, val_ds)
        tp_sc, fp_sc, tn_sc, fn_sc, ppr_sc, npr_sc = test_net(classifier, test_ds)
        t_tp_sc += tp_sc
        t_fp_sc += fp_sc
        t_tn_sc += tn_sc
        t_fn_sc += fn_sc
        t_ppr_sc += ppr_sc
        t_npr_sc += npr_sc
        bacc_sc = 0.5 * (tp_sc / (tp_sc + fn_sc) + tn_sc / (fp_sc + tn_sc))
        acc_sc = (tp_sc + tn_sc) / (tp_sc + tn_sc + fp_sc + fn_sc)
        pr_sc = tp_sc / (tp_sc + fn_sc)
        re_sc = tp_sc / (tp_sc + fp_sc)
        t_bacc_sc += bacc_sc
        t_acc_sc += acc_sc

        # Results per fold
        print(
            'Frozen (pre-trained)   | TP = {:03d} | FP = {:03d} | TN = {:03d} | FN = {:03d} |'.format(
                tp_fr, fp_fr, tn_fr, fn_fr
            ), end=' '
        )
        print(
            'Precision = {:5.3f} | Recall = {:5.3f} |'.format(
                pr_fr, re_fr
            ), end=' '
        )
        print(
            'ACC = {:5.3f} | BACC = {:5.3f} | + Scores {:5.3f} ± {:5.3f} | - Scores {:5.3f} ± {:5.3f}'.format(
                acc_fr, bacc_fr, np.mean(ppr_fr), np.std(ppr_fr), np.mean(npr_fr), np.std(npr_fr)
            )
        )
        print(
            'Unfrozen (pre-trained) | TP = {:03d} | FP = {:03d} | TN = {:03d} | FN = {:03d} |'.format(
                tp_un, fp_un, tn_un, fn_un
            ), end=' '
        )
        print(
            'Precision = {:5.3f} | Recall = {:5.3f} |'.format(
                pr_un, re_un
            ), end=' '
        )
        print(
            'ACC = {:5.3f} | BACC = {:5.3f} | + Scores {:5.3f} ± {:5.3f} | - Scores {:5.3f} ± {:5.3f}'.format(
                acc_un, bacc_un, np.mean(ppr_un), np.std(ppr_un), np.mean(npr_un), np.std(npr_un)
            )
        )

        print(
            'From scratch           | TP = {:03d} | FP = {:03d} | TN = {:03d} | FN = {:03d} |'.format(
                tp_sc, fp_sc, tn_sc, fn_sc
            ), end=' '
        )
        print(
            'Precision = {:5.3f} | Recall = {:5.3f} |'.format(
                pr_sc, re_sc
            ), end=' '
        )
        print(
            'ACC = {:5.3f} | BACC = {:5.3f} | + Scores {:5.3f} ± {:5.3f} | - Scores {:5.3f} ± {:5.3f}'.format(
                acc_sc, bacc_sc, np.mean(ppr_sc), np.std(ppr_sc), np.mean(npr_sc), np.std(npr_sc)
            )
        )

    print(
        '{:}Final{:} results'.format(
            c['b'], c['nc']
        )
    )

    print('-'.join([''] * 60))

    # Total results
    print(
        'Frozen (pre-trained)   | TP = {:03d} | FP = {:03d} | TN = {:03d} | FN = {:03d} |'.format(
            t_tp_fr, t_fp_fr, t_tn_fr, t_fn_fr
        ), end=' '
    )

    print(
        'ACC = {:5.3f} | BACC = {:5.3f} | + Scores {:5.3f} ± {:5.3f} | - Scores {:5.3f} ± {:5.3f}'.format(
            t_acc_fr / 5, t_bacc_fr / 5, np.mean(t_ppr_fr), np.std(t_ppr_fr), np.mean(t_npr_fr), np.std(t_npr_fr)
        )
    )
    print(
        'Unfrozen (pre-trained) | TP = {:03d} | FP = {:03d} | TN = {:03d} | FN = {:03d} |'.format(
            t_tp_un, t_fp_un, t_tn_un, t_fn_un
        ), end=' '
    )

    print(
        'ACC = {:5.3f} | BACC = {:5.3f} | + Scores {:5.3f} ± {:5.3f} | - Scores {:5.3f} ± {:5.3f}'.format(
            t_acc_un / 5, t_bacc_un / 5, np.mean(t_ppr_un), np.std(t_ppr_un), np.mean(t_npr_un), np.std(t_npr_un)
        )
    )
    print(
        'From scratch           | TP = {:03d} | FP = {:03d} | TN = {:03d} | FN = {:03d} |'.format(
            t_tp_sc, t_fp_sc, t_tn_sc, t_fn_sc
        ), end=' '
    )

    print(
        'ACC = {:5.3f} | BACC = {:5.3f} | + Scores {:5.3f} ± {:5.3f} | - Scores {:5.3f} ± {:5.3f}'.format(
            t_acc_sc / 5, t_bacc_sc / 5, np.mean(t_ppr_sc), np.std(t_ppr_sc), np.mean(t_npr_sc), np.std(t_npr_sc)
        )
    )

if __name__ == '__main__':
    main()
