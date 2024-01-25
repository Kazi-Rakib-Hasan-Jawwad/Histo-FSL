import torch
from torch import nn
from torchvision.models import resnet18, resnet50

def get_backbone(model_path, model_name):
    # Load the pretrained model
    if model_name == 'resnet18':
        convolutional_network = resnet18(pretrained=False)
    elif model_name == 'resnet50':
        convolutional_network = resnet50(pretrained=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    # Load the state_dict from the provided path
    state_dict = torch.load(model_path)
    '''
    # Remove the last fully connected layer (fc) from the model
    backbone = nn.Sequential(*list(convolutional_network.children())[:-1])
    # Load the pretrained weights into the backbone
    backbone.load_state_dict(state_dict)
    '''
    # Replace the last fully connected layer (fc) and flatten the output
    convolutional_network.fc = nn.Flatten()
    convolutional_network.load_state_dict(state_dict)
    return convolutional_network
