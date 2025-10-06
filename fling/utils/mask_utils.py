import torch

def init_mask_zeros(model: nn.Module) -> OrderedDict:
    """Initialize binary mask of zeros for model parameters.

    Args:
        model: Neural network model

    Returns:
        OrderedDict containing binary masks initialized to zeros
    """
    mask = OrderedDict()
    for name, param in model.named_parameters():
        mask[name] = torch.zeros_like(param)
    return mask     
