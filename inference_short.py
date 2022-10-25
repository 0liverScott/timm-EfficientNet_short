#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import logging
import numpy as np
import torch
import datetime

from timm.models import create_model
from timm.data import ImageDataset, create_loader
from timm.data.transforms import str_to_interp_mode
from timm.utils import AverageMeter, setup_default_logging
from torchvision import transforms

from PIL import Image

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')

class ToNumpy:
    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img

def prepare():
    setup_default_logging()

    # create model
    model = create_model(
        'tf_efficientnet_b6_ns',
        num_classes=num_class,
        in_chans=3,
        checkpoint_path='pth/tf_efficientnet_b6_ns_0530_9874_9898.tar')
    if num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu))).cuda()
    else:
        model = model.cuda()
    model.eval()
    return model

def classify2():
    filename = os.listdir(input_path)
    labels = []
    t0 = time.time()
    img_list = []

    for im in os.listdir(input_path):
        img = Image.open(f'{input_path}/{im}').convert('RGB')
        timg = trans(img)
        timg = timg.float().cuda()
        img_list.append(timg)
    images = torch.stack(img_list, 0)
    out = model(images)
    topk = out.topk(1)[1]
    labels.append(topk.cpu().numpy())

    print(labels[0])

    for file, label in zip(filename, labels[0]):
        if label == 0:
            print(f"{file} 0_With_Mask")
        if label == 1:
            print(f"{file} 1_Wrong_Mask")
        else:
            print(f"{file} 2_Without_Mask")
    print(f"{time.time()-t0}s estimated.")

def classify1():
    loader = create_loader(
        ImageDataset(input_path),
        input_size=(3, input_size, input_size),
        batch_size=batch_size,
        use_prefetcher=True,
        num_workers=num_workers)
    k = min(1, 3)  # topk, num class
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            input = input.cuda()
            labels = model(input)
            topk = labels.topk(k)[1]
            topk_ids.append(topk.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

    topk_ids = np.concatenate(topk_ids, axis=0)
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S.%f')[:-4]

    with open(os.path.join(output_path, f'./{date}.csv'), 'w') as out_file:
        filenames = loader.dataset.filenames(basename=True)
        for filename, label in zip(filenames, topk_ids):
            out_file.write('{0},{1}\n'.format(
                filename, ','.join(
                    [("0_With_Mask" if v == 0 else "1_Wrong_Mask" if v == 1 else "2_Without_Mask") for v in label])))



if __name__ == '__main__':

    num_gpu = 1
    log_freq = 1
    batch_size = 1
    num_workers = 2
    input_path = 'input'
    output_path = 'output'
    num_class = 3
    classes = ["0_With_Mask", "1_Wrong_Mask", "2_Without_Mask"]
    input_size = 224
    trans = transforms.Compose([transforms.Resize(237, interpolation=str_to_interp_mode('bicubic')),
                                transforms.CenterCrop((input_size, input_size)), transforms.ToTensor(),
                                transforms.Normalize(mean=torch.tensor((0.485, 0.456, 0.406)),
                                                     std=torch.tensor((0.229, 0.224, 0.225)))])
    print('prepare')
    model = prepare()
    print('start')
    classify2(model)

