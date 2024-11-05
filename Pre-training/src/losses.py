import torch
import loss_utils


def color_consistency_loss(src, tgt, w=None, reduce_loss=True):
    '''
    Computes the color consistency loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 1 x H x W weights
        reduce_loss : bool
            if set, then reduce loss over height and weight dimensions
    Returns:
        Either
        (1) [reduce_loss=True] torch.Tensor[float32] : mean absolute difference between source and target images
        (2) [reduce_loss=False] torch.Tensor[float32] : absolute difference between source and target images, N x 1 x H x W
    '''

    if w is None:
        w = torch.ones_like(src)

    loss = torch.sum(w * torch.abs(tgt - src), dim=1, keepdim=True)

    if reduce_loss:
        return torch.mean(loss)
    else:
        return loss

def structural_consistency_loss_func(src, tgt, w=None, reduce_loss=True):
    '''
    Computes the structural consistency loss using SSIM

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 3 x H x W weights
        reduce_loss : bool
            if set, then reduce loss over height and weight dimensions
    Returns:
        Either
        (1) [reduce_loss=True] torch.Tensor[float32] : mean absolute difference between source and target images
        (2) [reduce_loss=False] torch.Tensor[float32] : absolute difference between source and target images, N x 1 x H x W
    '''

    if w is None:
        w = torch.ones_like(src)

    refl = torch.nn.ReflectionPad2d(1)

    src = refl(src)
    tgt = refl(tgt)

    scores = loss_utils.ssim(src, tgt)

    loss = torch.sum(w * scores, dim=1, keepdim=True)

    if reduce_loss:
        return torch.mean(loss)
    else:
        return loss

def smoothness_loss_func(predict, image):
    '''
    Computes the local smoothness loss

    Arg(s):
        predict : torch.Tensor[float32]
            N x 1 x H x W predictions
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
    Returns:
        torch.Tensor[float32] : mean edge weighted smoothness loss
    '''

    predict_dy, predict_dx = loss_utils.gradient_yx(predict)
    image_dy, image_dx = loss_utils.gradient_yx(image)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))

    return smoothness_x + smoothness_y
