# Adapted from https://github.com/wannabeOG/MAS-PyTorch

import torch
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from networks.loss import class_cross_entropy_loss


# from networks.mainnetwork import *


class LocalSgd(optim.SGD):
    def __init__(self, params, lambda_reg: float, lr: float = 1e-5, momentum: float = 0, dampening: float = 0,
                 weight_decay: float = 0, nesterov: bool = False):
        super(LocalSgd, self).__init__(params, lr,
                                       momentum, dampening, weight_decay, nesterov)
        self.lambda_reg = lambda_reg

    def __setstate__(self, state):
        super(LocalSgd, self).__setstate__(state)

    @torch.no_grad()
    def step(self, reg_params, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if p in reg_params:
                    param_dict = reg_params[p]

                    omega = param_dict['omega']
                    init_val = param_dict['init_val']


                    local_grad = torch.mul(
                        2 * self.lambda_reg * omega, p.data - init_val)


                    d_p = d_p.add(local_grad)

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss


class OmegaSgd(optim.SGD):
    def __init__(self, params, lr: float = 5e-6, momentum: float = 0.9, dampening: float = 0, weight_decay: float = 0.0005,
                 nesterov: bool = False):
        super(OmegaSgd, self).__init__(params, lr,
                                       momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(OmegaSgd, self).__setstate__(state)

    @torch.no_grad()
    def step(self, reg_params, batch_index: int, batch_size: int, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for param in group['params']:
                if param.grad is None:
                    continue

                if param in reg_params:
                    d_param = param.grad
                    d_param_copy = d_param.clone().abs()

                    param_dict = reg_params[param]

                    omega = param_dict['omega']

                    current_size = (batch_index + 1) * batch_size
                    step_size = 1 / float(current_size)

                    omega = omega + step_size * \
                            (d_param_copy - batch_size * omega)

                    param_dict['omega'] = omega
                    reg_params[param] = param_dict

        return loss


from torch.nn.functional import upsample


def compute_omega_grads_norm(net, dataloader, optimizer):
    device = net.device()
    net.eval()
    num_batches = len(dataloader)

    print('Computing omega grads (norm output)')
    refinement_num_max = 1
    threshold = 0.95
    for index, sample_batched in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        gts = sample_batched['crop_gt']
        inputs = sample_batched['concat']
        void_pixels = sample_batched['crop_void_pixels']
        IOG_points = sample_batched['IOG_points']



        for i in range(0, refinement_num_max + 1):
            inputs.requires_grad_()
            inputs, gts, void_pixels, IOG_points = inputs.to(device), gts.to(device), void_pixels.to(
                device), IOG_points.to(device)
            if i == 0:
                distance_map_512 = None
                init_mask = torch.zeros_like(gts)
                correction_map = -torch.ones(IOG_points.size(0), 1, IOG_points.size(2),
                                             IOG_points.size(3)).cuda()
            output_glo1, output_glo2, output_glo3, output_glo4, output_refine, iou_i, distance_map_512, \
            correction_map = net.forward(inputs, IOG_points, gts, i, distance_map_512, correction_map)

            # Compute the losses, side outputs and fuse
            loss_output_glo1 = class_cross_entropy_loss(output_glo1, gts, void_pixels=void_pixels,
                                                        size_average=False, batch_average=True)
            loss_output_glo2 = class_cross_entropy_loss(output_glo2, gts, void_pixels=void_pixels,
                                                        size_average=False, batch_average=True)
            loss_output_glo3 = class_cross_entropy_loss(output_glo3, gts, void_pixels=void_pixels,
                                                        size_average=False, batch_average=True)

            loss_output_glo4 = class_cross_entropy_loss(output_glo4, gts, void_pixels=void_pixels,
                                                        size_average=False, batch_average=True)
            loss_output_refine = class_cross_entropy_loss(output_refine, gts, void_pixels=void_pixels,
                                                          size_average=False, batch_average=True)

            if i == 0:
                loss = loss_output_glo1 + loss_output_glo2 + loss_output_glo3 + loss_output_glo4 + loss_output_glo4 + loss_output_refine
                iou1 = iou_i
            if i == 1:
                loss = loss_output_glo1 + loss_output_glo2 + loss_output_glo3 + loss_output_glo4 + loss_output_glo4 + loss_output_refine
                iou2 = iou_i

            loss.backward()

            optimizer.step(net.reg_params, batch_index=index,
                       batch_size=inputs.size(0))

    return net


def sanity_model(model):
    for name, param in model.named_parameters():

        print(name)

        if param in model.reg_params:
            param_dict = model.reg_params[param]
            omega = param_dict['omega']

            print("Max omega is", omega.max())
            print("Min omega is", omega.min())
            print("Mean value of omega is", omega.mean())