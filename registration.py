import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as func
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim


"""
> Similarity-based losses
"""


def xcor_loss(fixed, moved, mask=None):
    if mask is None:
        fixed_norm = fixed - torch.mean(fixed)
        moved_norm = moved - torch.mean(moved)
    else:
        valid_fixed = fixed[mask]
        valid_moved = moved[mask]
        fixed_norm = valid_fixed - torch.mean(valid_fixed)
        moved_norm = valid_moved - torch.mean(valid_moved)
    fixed_sq = torch.sum(fixed_norm ** 2)
    moved_sq = torch.sum(moved_norm ** 2)

    den = torch.sqrt(fixed_sq * moved_sq)
    num = torch.sum(fixed_norm * moved_norm)

    xcor = num / den if den > 0 else 0

    return 1. - xcor


def xcor_patch_loss(fixed, moved, mask=None, k=8):
    if len(fixed.shape) < 4:
        unsqueeze = (1,) * (4 - len(fixed.shape))
        fixed = torch.reshape(fixed, unsqueeze + fixed.shape)
    if len(moved.shape) < 4:
        unsqueeze = (1,) * (4 - len(moved.shape))
        moved = torch.reshape(moved, unsqueeze + moved.shape)
    fixed_mean = func.interpolate(
        func.avg_pool2d(fixed, k),
        fixed.shape[2:]
    )
    moved_mean = func.interpolate(
        func.avg_pool2d(moved, k),
        moved.shape[2:]
    )
    fixed_norm = fixed - fixed_mean
    moved_norm = moved - moved_mean
    fixed_sq = func.avg_pool2d(fixed_norm ** 2, k)
    moved_sq = func.avg_pool2d(moved_norm ** 2, k)

    den = torch.sqrt(fixed_sq * moved_sq)
    num = func.avg_pool2d(fixed_norm * moved_norm, k)

    xcor = torch.mean(num / den)

    return 1. - xcor


def mse_loss(fixed, moved, mask=None):
    if mask is None:
        mse_val = func.mse_loss(moved, fixed)
    else:
        valid_fixed = fixed[mask]
        valid_moved = moved[mask]
        mse_val = func.mse_loss(valid_moved, valid_fixed)

    return mse_val


"""
> Registration code
"""


def resample(
    moving, moving_spacing, output_dims, output_spacing,
    affine, mode='bilinear'
):
    m_width, m_height, m_depth = moving.shape
    m_width_s, m_height_s, m_depth_s = moving_spacing
    f_width, f_height, f_depth = output_dims
    f_width_s, f_height_s, f_depth_s = output_spacing

    image_tensor = torch.from_numpy(
        moving.astype(np.float32)
    ).view(
        (1, 1, m_width, m_height, m_depth)
    ).to(affine.device)

    if f_width_s == m_width_s:
        x_step = 1
    else:
        x_step = f_width_s / m_width_s
    if f_height_s == m_height_s:
        y_step = 1
    else:
        y_step = f_height_s / m_height_s
    if f_depth_s == m_depth_s:
        z_step = 1
    else:
        z_step = f_depth_s / m_depth_s

    # Initial grid
    x = torch.arange(
        start=0, end=x_step * f_width, step=x_step
    ).to(dtype=torch.float64, device=affine.device)
    y = torch.arange(
        start=0, end=y_step * f_height, step=y_step
    ).to(dtype=torch.float64, device=affine.device)
    z = torch.arange(
        start=0, end=z_step * f_depth, step=z_step
    ).to(dtype=torch.float64, device=affine.device)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack([
        grid_z.flatten(),
        grid_y.flatten(),
        grid_x.flatten(),
        torch.ones_like(grid_x.flatten())
    ], dim=0)

    scales = torch.tensor(
        [[m_depth], [m_height], [m_width]],
        dtype=torch.float64, device=affine.device
    )

    affine_grid = 2 * (affine @ grid)[:3, :] / scales - 1

    tensor_grid = torch.swapaxes(affine_grid, 0, 1).view(
        1, f_width, f_height, f_depth, 3
    )

    moved = func.grid_sample(
        image_tensor,
        tensor_grid.to(torch.float32),
        align_corners=True, mode=mode
    ).view(output_dims)

    return moved


def halfway_registration(
    image_a, image_b, spacing_a, spacing_b, mask_a=None, mask_b=None,
    shape_target=None, spacing_target=None,
    scales=None, epochs=500, patience=100, init_lr=1e-3, loss_f=xcor_loss,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
):
    if scales is None:
        scales = [8, 4, 2, 1]

    best_fit = np.inf
    final_e = 0
    final_fit = np.inf

    id_affine = np.eye(4)

    image_a = (image_a - image_a.mean()) / image_a.std()
    image_b = (image_b - image_b.mean()) / image_b.std()

    if shape_target is None:
        shape_target = image_a
    if spacing_target is None:
        spacing_target = spacing_a
    R = torch.tensor(
        id_affine[:3, :3], device=device,
        requires_grad=True, dtype=torch.float64
    )
    T = torch.tensor(
        id_affine[:3, 3:], device=device,
        requires_grad=True, dtype=torch.float64
    )
    fixed_affine = torch.tensor(
        id_affine[3:, :], device=device,
        requires_grad=False, dtype=torch.float64
    )

    lr = init_lr
    best_R = R.detach().cpu().numpy()
    best_T = T.detach().cpu().numpy()

    for s in scales:
        optimizer = torch.optim.Adam([R, T], lr=lr)
        no_improv = 0
        for e in range(epochs):
            learnable_affine_a = torch.cat([R, T], dim=1)
            affine_a = torch.cat([learnable_affine_a, fixed_affine])
            Rt = R.transpose(0, 1)
            RtT = Rt @ T
            learnable_affine_b = torch.cat([Rt, RtT], dim=1)
            affine_b = torch.cat([learnable_affine_b, fixed_affine])

            moved_a = resample(
                image_a, spacing_a,
                shape_target, spacing_target,
                affine_a
            )
            moved_b = resample(
                image_b, spacing_b,
                shape_target, spacing_target,
                affine_b
            )
            tensor_a = moved_a.view((1, 1) + shape_target)
            tensor_b = moved_b.view((1, 1) + shape_target)
            tensor_a_s = func.avg_pool3d(tensor_a, s)
            tensor_b_s = func.avg_pool3d(tensor_b, s)

            if mask_a is not None:
                mask_tensor_a = resample(
                    mask_a.astype(np.float32), spacing_a,
                    shape_target, spacing_target,
                    affine_a,
                    mode='nearest'
                ).view((1, 1) + shape_target)
                mask_tensor_a_s = func.max_pool3d(
                    mask_tensor_a, s
                ) > 0
            else:
                mask_tensor_a_s = None

            if mask_b is not None:
                mask_tensor_b = resample(
                    mask_b.astype(np.float32), spacing_b,
                    shape_target, spacing_target,
                    affine_b,
                    mode='nearest'
                ).view((1, 1) + shape_target)
                mask_tensor_b_s = func.max_pool3d(
                    mask_tensor_b, s
                ) > 0
            else:
                mask_tensor_b_s = None

            if mask_tensor_a_s is not None and mask_tensor_b_s is not None:
                mask_tensor = mask_tensor_a_s * mask_tensor_b_s
            elif mask_tensor_a_s is not None:
                mask_tensor = mask_tensor_a_s
            elif mask_tensor_b_s is not None:
                mask_tensor = mask_tensor_b_s
            else:
                mask_tensor = None

            if mask_tensor is None:
                loss = loss_f(tensor_a_s, tensor_b_s)
            else:
                loss = loss_f(tensor_a_s, tensor_b_s, mask_tensor)

            loss_value = loss.detach().cpu().numpy().tolist()
            if loss_value < best_fit:
                final_e = e
                final_fit = loss_value
                best_fit = loss_value
                best_R = R.detach().cpu().numpy()
                best_T = T.detach().cpu().numpy()
            else:
                no_improv += 1
                if no_improv == patience:
                    break
            optimizer.zero_grad()
            loss.backward()
            if e == 0:
                print('Pytorch - Epoch {:04d} [scale {:02d}]: {:8.4f}'.format(
                    e + 1, s, loss_value
                ))
            optimizer.step()
        R = torch.tensor(
            best_R, device=device, requires_grad=True,
            dtype=torch.float64
        )
        T = torch.tensor(
            best_T, device=device, requires_grad=True,
            dtype=torch.float64
        )
        print('Pytorch - Epoch {:04d} [scale {:02d}]: {:8.4f}'.format(
            final_e + 1, s, final_fit
        ))
        best_fit = np.inf
        # lr = lr / 5
    learnable_affine_a = torch.cat([R, T], dim=1)
    Rt = R.transpose(0, 1)
    RtT = Rt @ T
    affine_a = torch.cat([learnable_affine_a, fixed_affine.detach()])
    learnable_affine_b = torch.cat([Rt, RtT], dim=1)
    affine_b = torch.cat([learnable_affine_b, fixed_affine.detach()])
    return affine_a, affine_b, final_e, final_fit


def sitk_registration(
        imagename_a, imagename_b, outputname_a, outputname_b,
        shape_target, spacing_target
):
    fixed_image = sitk.ReadImage(imagename_a, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(imagename_b, sitk.sitkFloat32)

    initial_transform = sitk.Euler3DTransform()

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    # registration_method.AddCommand(sitk.sitkStartEvent, )
    # registration_method.AddCommand(sitk.sitkEndEvent, )

    final_transform = registration_method.Execute(
        fixed_image, moving_image
    ).GetNthTransform(0)

    # Always check the reason optimization terminated.
    print("SimpleITK - Final metric value: {0}".format(registration_method.GetMetricValue()))
    print(
        "SimpleITK - Optimizer's stopping condition, {0}".format(
            registration_method.GetOptimizerStopConditionDescription()
        )
    )

    angle_x = final_transform.GetAngleX()
    angle_y = final_transform.GetAngleY()
    angle_z = final_transform.GetAngleZ()

    t = final_transform.GetTranslation()

    half_x = angle_x / 2
    half_y = angle_y / 2
    half_z = angle_z / 2
    half_t = [t_i / 2 for t_i in t]
    neg_half_t = [-t_i / 2 for t_i in t]

    affine_a = sitk.Euler3DTransform(
        final_transform.GetCenter(), half_x, half_y, half_z, half_t
    )

    affine_b = sitk.Euler3DTransform(
        final_transform.GetCenter(), -half_x, -half_y, -half_z, neg_half_t
    )



    a_resampled = sitk.Resample(
        moving_image,
        size=shape_target,
        outputSpacing=spacing_target,
        transform=affine_a,
        interpolator=sitk.sitkLinear,
        defaultPixelValue=0.0,
        PixelIDValueEnum=moving_image.GetPixelID(),
    )
    sitk.WriteImage(
        a_resampled, outputname_a
    )

    b_resampled = sitk.Resample(
        fixed_image,
        size=shape_target,
        outputSpacing=spacing_target,
        transform=affine_b,
        interpolator=sitk.sitkLinear,
        defaultPixelValue=0.0,
        PixelIDValueEnum=fixed_image.GetPixelID(),
    )
    sitk.WriteImage(
        b_resampled, outputname_b
    )
