'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Safa Cicek <safacicek@ucla.edu>
If this code is useful to you, please consider citing the following paper:
A. Wong, S. Cicek, and S. Soatto. Targeted Adversarial Perturbations for Monocular Depth Prediction.
https://arxiv.org/pdf/2006.08602.pdf
@inproceedings{wong2020targeted,
    title={Targeted Adversarial Perturbations for Monocular Depth Prediction},
    author={Wong, Alex and Safa Cicek and Soatto, Stefano},
    booktitle={Advances in neural information processing systems},
    year={2020}
}
'''
import os, gdown
from zipfile import ZipFile


# URL to checkpoints
MONODEPTH2_RESNET18_MONO_STEREO_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip'

MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip'

MONODEPTH2_RESNET50_MONO_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_resnet50_640x192.zip'

MONODEPTH2_RESNET50_MONO_NOPT_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_resnet50_no_pt_640x192.zip'


MONODEPTH2_MODELS_DIRPATH = os.path.join('pretrained_models', 'monodepth2')

MONODEPTH2_RESNET18_MODELS_DIRPATH = os.path.join(MONODEPTH2_MODELS_DIRPATH, 'resnet18')

MONODEPTH2_RESNET18_MONO_STEREO_MODEL = 'mono_stereo_640x192'
MONODEPTH2_RESNET18_MONO_STEREO_MODEL_FILENAME = MONODEPTH2_RESNET18_MONO_STEREO_MODEL + '.zip'
MONODEPTH2_RESNET18_MONO_STEREO_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET18_MODELS_DIRPATH,
    MONODEPTH2_RESNET18_MONO_STEREO_MODEL_FILENAME)

MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL = 'mono_stereo_nopt_640x192'
MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL_FILENAME = MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL + '.zip'
MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET18_MODELS_DIRPATH,
    MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL_FILENAME)

MONODEPTH2_RESNET50_MODELS_DIRPATH = os.path.join(MONODEPTH2_MODELS_DIRPATH, 'resnet50')

MONODEPTH2_RESNET50_MONO_MODEL = 'mono_640x192'
MONODEPTH2_RESNET50_MONO_MODEL_FILENAME = MONODEPTH2_RESNET50_MONO_MODEL + '.zip'
MONODEPTH2_RESNET50_MONO_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET50_MODELS_DIRPATH,
    MONODEPTH2_RESNET50_MONO_MODEL_FILENAME)

MONODEPTH2_RESNET50_MONO_NOPT_MODEL = 'mono_nopt_640x192'
MONODEPTH2_RESNET50_MONO_NOPT_MODEL_FILENAME = MONODEPTH2_RESNET50_MONO_NOPT_MODEL + '.zip'
MONODEPTH2_RESNET50_MONO_NOPT_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET50_MODELS_DIRPATH,
    MONODEPTH2_RESNET50_MONO_NOPT_MODEL_FILENAME)

for dirpath in [MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET50_MODELS_DIRPATH]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Download ResNet18 monocular, stereo with pretrained ImageNet
if not os.path.exists(MONODEPTH2_RESNET18_MONO_STEREO_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet18 stereo video 640x192 model with pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET18_MONO_STEREO_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET18_MONO_STEREO_MODEL_URL,
        MONODEPTH2_RESNET18_MONO_STEREO_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet18 stereo video 640x192 model with pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET18_MONO_STEREO_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET18_MONO_STEREO_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET18_MONO_STEREO_MODEL))

# Download ResNet18 monocular, stereo without pretrained ImageNet
if not os.path.exists(MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet18 stereo video 640x192 model without pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL_URL,
        MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet18 stereo video 640x192 model without pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET18_MONO_STEREO_NOPT_MODEL))

# Download ResNet50 monocular with pretrained ImageNet
if not os.path.exists(MONODEPTH2_RESNET50_MONO_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet50 monocular 640x192 model with pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET50_MONO_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET50_MONO_MODEL_URL,
        MONODEPTH2_RESNET50_MONO_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet50 monocular 640x192 model with pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET50_MONO_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET50_MONO_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET50_MODELS_DIRPATH, MONODEPTH2_RESNET50_MONO_MODEL))

# Download ResNet50 monocular without pretrained ImageNet
if not os.path.exists(MONODEPTH2_RESNET50_MONO_NOPT_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet50 monocular 640x192 model without pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET50_MONO_NOPT_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET50_MONO_NOPT_MODEL_URL,
        MONODEPTH2_RESNET50_MONO_NOPT_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet50 monocular 640x192 model without pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET50_MONO_NOPT_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET50_MONO_NOPT_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET50_MODELS_DIRPATH, MONODEPTH2_RESNET50_MONO_NOPT_MODEL))
