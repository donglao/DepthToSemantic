import os
from PIL import Image
import numpy as np
import torch

def load_image(path, normalize=True, data_format='HWC',reshape=None):
    '''
    Loads an RGB image

    Arg(s):
        path : str
            path to RGB image
        normalize : bool
            if set, then normalize image between [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : H x W x C or C x H x W image
    '''

    # Load image
    if reshape is None:
        image = Image.open(path).convert('RGB')
    else:
        image = Image.open(path).convert('RGB').resize(reshape)

    # Convert to numpy
    image = np.asarray(image, np.float32)

    if data_format == 'CHW':
        image = np.transpose(image, (2, 0, 1))

    # Normalize
    image = image / 255.0 if normalize else image

    return image

def load_semantic_label(path, data_format='HW',reshape=None):
    if reshape is None:
        label = np.asarray(Image.open(path).convert('L'), dtype=np.uint8)
    else:
        label = np.asarray(Image.open(path).convert('L').resize(reshape), dtype=np.uint8)
    ignore_list = [-1,0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
    for i in ignore_list:
        label[label==i] = 0 #remove unwanted classes

    label[label>33] = 0
    label[label<0] = 0
        
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        label = np.expand_dims(label, axis=0)
    elif data_format == 'HWC':
        label = np.expand_dims(label, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return label


class KITTISemanticSegmentation(torch.utils.data.Dataset):
    def __init__(self, data_path=None,  train=True, training_samples=50, data_format = 'CHW',reshape=None):
    # KITTI has 200 samples for semantic segmentation, take the first X as training set        
        self.data_path = data_path
        self.train = train
        self.data_format = data_format
        self.training_samples = training_samples
        self.reshape = reshape

    def __len__(self):
        if self.train is True:
            return self.training_samples
        else:
            return 200

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path,'training','image_2', str(index).zfill(6)+'_10.png')
        label_path = os.path.join(self.data_path,'training','semantic', str(index).zfill(6)+'_10.png')
        # Load image
        image = load_image(
            path=image_path,
            normalize=True,
            data_format=self.data_format,
            reshape = self.reshape)
        # Load labels
        label = load_semantic_label(
            path=label_path,
            reshape=self.reshape) #automatically loas as HW

        return image, label.astype(np.uint8)
