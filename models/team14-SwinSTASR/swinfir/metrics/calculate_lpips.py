


import lpips
import torch
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY

loss_fn_alex = None


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False, strict_shape=False, **kwargs):
    """Calculate LPIPS

    Ref: https://github.com/richzhang/PerceptualSimilarity

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """
    global loss_fn_alex
    if loss_fn_alex is None:
        loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores

        if torch.cuda.is_available():
            loss_fn_alex.cuda()
    if strict_shape:
        assert img.shape == img2.shape, (f'Image shapes are differnet: {img.shape}, {img2.shape}.')
    else:
        h, w, c = img.shape
        img2 = img2[0:h, 0:w, 0:c]
        if img.shape != img2.shape:
            h, w, c = img2.shape
            img = img[0:h, 0:w, 0:c]
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    def np2tensor(x):
        """

        Args:
            x: RGB [0 ~ 255] HWC ndarray

        Returns: RGB [-1, 1]

        """
        return torch.tensor((x * 2 / 255.0) - 0.5).permute(2, 0, 1).unsqueeze(0).float()

    # np2tensor
    img = np2tensor(img)
    img2 = np2tensor(img2)

    if torch.cuda.is_available():
        img = img.cuda()
        img2 = img2.cuda()

    with torch.no_grad():
        d = loss_fn_alex(img, img2)
    return d.view(1).cpu().numpy()[0]






"""
import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
from basicsr.utils.registry import METRIC_REGISTRY

from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')

@METRIC_REGISTRY.register
def calculate_lpips():
    # Configurations
    # -------------------------------------------------------------------------
    folder_gt = 'datasets/'
    folder_restored = 'datasets/'
    # crop_border = 4
    suffix = ''
    # -------------------------------------------------------------------------
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(osp.join(folder_restored, basename + suffix + ext), cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())

        print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val:.6f}.')
        lpips_all.append(lpips_val)

    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')



if __name__ == '__main__':
    calculate_lpips()

"""
