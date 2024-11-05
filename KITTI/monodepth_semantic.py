'''
Modified from code template
A. Wong, S. Cicek, and S. Soatto. Targeted Adversarial Perturbations for Monocular Depth Prediction.
https://arxiv.org/pdf/2006.08602.pdf
@inproceedings{wong2020targeted,
    title={Targeted Adversarial Perturbations for Monocular Depth Prediction},
    author={Wong, Alex and Safa Cicek and Soatto, Stefano},
    booktitle={Advances in neural information processing systems},
    year={2020}
}
'''
import os, sys
import torch
import net_utils
sys.path.insert(0, os.path.join(os.getcwd(), 'external_src', 'monodepth2'))
sys.path.insert(0, os.path.join(os.getcwd(), 'external_src', 'monodepth2', 'networks'))
from resnet_encoder import ResnetEncoder
from semantic_decoder import SemanticDecoder

class Monodepth2SemanticModel(object):
    '''

    Wrapper class for Monodepth2

    Args:
        device : torch.device
            device to run on
    '''
    def __init__(self, device=torch.device('cuda'),encoder='resnet50'):

        # Depth range settings
        self.device = device

        # Restore depth prediction network
        if encoder == 'resnet50':
            self.encoder = ResnetEncoder(50, False)
        elif encoder == 'resnet18':
            self.encoder = ResnetEncoder(18, False)
            
        self.decoder = SemanticDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            scales=range(4))

        # Move to device
        self.to(self.device)
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, image):
        # Forward through network
        latent = self.encoder(image)
        outputs = self.decoder(latent)

        return outputs

    def to(self, device):
        '''
        Moves model to device

        Args:
            device : torch.device
        '''

        self.encoder.to(device)
        self.decoder.to(device)
    
    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder.train()
        self.decoder.train()

    def freeze_encoder(self):
        '''
        Sets model to training mode
        '''

        self.encoder.eval() # no grad
        self.decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder.eval()
        self.decoder.eval()

    def parameters(self):
        '''
        Fetches model parameters

        Returns:
            list : list of model parameters
        '''

        parameters = \
            list(self.encoder.parameters()) + \
            list(self.decoder.parameters())

        return parameters

    def restore_model(self,
                      encoder_restore_path=None,
                      decoder_restore_path=None):
        '''
        Restores model from checkpoint path

        Args:
            encoder_restore_path : str
                checkpoint for encoder
            decoder_depth_restore_path : str
                checkpoint for decoder for depth
            decoder_pose_restore_path : str
                checkpoint for decoder for pose
        '''

        if encoder_restore_path is not None:
            # Restore weights for encoder
            loaded_dict_encoder = torch.load(encoder_restore_path, map_location=self.device)

            if 'model_state_dict' in loaded_dict_encoder.keys():
                loaded_dict_encoder = loaded_dict_encoder['model_state_dict']

            filtered_dict_encoder = {
                k : v for k, v in loaded_dict_encoder.items() if k in self.encoder.state_dict()
            }

            self.encoder.load_state_dict(filtered_dict_encoder)

        if decoder_restore_path is not None:
            # Restore weights for decoder for depth
            loaded_dict_decoder = torch.load(decoder_restore_path, map_location=self.device)
            
            if 'model_state_dict' in loaded_dict_decoder.keys():
                loaded_dict_decoder = loaded_dict_decoder['model_state_dict']

            current_model_dict = self.decoder.state_dict()
            # to ignore layers with size mismatch
            new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_dict_decoder.values())}
            self.decoder.load_state_dict(new_state_dict, strict=False) # change last layer!

            # self.decoder.load_state_dict(loaded_dict_decoder, strict=False) 

    def load_ImageNet_Pretrain(self, encoder_type=None):
        import torchvision.models as models

        if encoder_type == 'resnet18':
            network = models.resnet18(pretrained=True)
        if encoder_type == 'resnet50':
            network = models.resnet50(pretrained=True)

        loaded_dict_encoder = network.state_dict()
        for key in list(loaded_dict_encoder.keys()):
            loaded_dict_encoder['encoder.' + key] = loaded_dict_encoder.pop(key)

            
        filtered_dict_encoder = {
            k : v for k, v in loaded_dict_encoder.items() if k in self.encoder.state_dict()
            }

        self.encoder.load_state_dict(filtered_dict_encoder)
