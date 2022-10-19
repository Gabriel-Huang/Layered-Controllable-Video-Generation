import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.losses.lpips import LPIPS
from modules.discriminator.model import NLayerDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def min_mask_loss(mask, min_mask_coverage, max_mask_coverage):
    min = F.relu(min_mask_coverage - mask.mean(dim=(1, 2, 3))).mean()
    max = F.relu(mask.mean(dim=(1, 2, 3)) - max_mask_coverage).mean()
    return min + max

def min_permask_loss(mask, min_mask_coverage):
    '''
    One object mask per channel in this case
    '''
    return F.relu(min_mask_coverage - mask.mean(dim=(2, 3))).mean()

def binarization_loss(mask):
    return torch.min(1-mask, mask).mean()

def rgb2gray(input):
    input = input.permute(0,2,3,1)
    gray = torch.matmul(input, torch.tensor([0.299, 0.587, 0.144]).cuda())
    return torch.unsqueeze(gray, 1)

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", 
                 BG_weight = 10, mask_bound_weight = 5, mask_loss_weight = 1, mask_binarization_weight = 1, BG_sparsity_ratio = 80, FG_weight = 20):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()

        self.perceptual_weight = perceptual_weight
        self.mask_binarization_weight = mask_binarization_weight
        self.BG_weight = BG_weight
        self.mask_bound_weight = mask_bound_weight
        self.mask_loss_weight = mask_loss_weight
        self.BG_sparsity_ratio = BG_sparsity_ratio
        self.FG_weight = FG_weight
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def calculate_sparsity_weight(self, bg_loss, sparsity_loss, global_step):
        
        if global_step % 2000 == 0 and global_step != 0:
            self.BG_sparsity_ratio /= 2

        if self.BG_sparsity_ratio < 5:
            self.BG_sparsity_ratio = 5

        if global_step == 10000:
            self.BG_weight = 5

        # if global_step == 10000:
        #     self.BG_weight = 1

        mask_sparsity_weight = self.BG_weight/self.BG_sparsity_ratio
        
        return mask_sparsity_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", mask_gen = None, current = None, finetune = True):
        # reconstruction loss:
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        # perceptual loss eq.(10):
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        mask_loss = torch.tensor([0.0]).cuda()
        L1_loss = nn.L1Loss()

        # phase II training, mask regularizations:
        binary_loss = torch.tensor([0.0]).cuda()
        sparsity_loss = torch.tensor([0.0]).cuda()
        inv_mask = torch.ones(mask_gen.shape).cuda() - mask_gen

        if not finetune:
            
            # BG loss, eq.(12):
            BG_GT = inputs * inv_mask
            BG_rec = current * inv_mask
            BG_L1 = L1_loss(BG_GT.contiguous(), BG_rec.contiguous())
            bg_loss = self.BG_weight * BG_L1
            mask_loss += bg_loss

            # binary loss eq.(15):
            binary_loss = self.mask_binarization_weight * torch.nan_to_num(binarization_loss(mask_gen))
            mask_loss += binary_loss

            # sparsity constrain eq.(14):
            if optimizer_idx == 0:
                diff = torch.abs(rgb2gray(current) - rgb2gray(inputs))
                diff = torch.where(diff > 0.04, 1, 0) * 1.0
                sparsity_loss = F.relu(mask_gen.mean(dim=(1, 2, 3)) - diff.mean(dim=(1, 2, 3))).mean()
                sparsity_loss = sparsity_loss.mean()

                sparsity_weight = self.calculate_sparsity_weight(bg_loss, sparsity_loss, global_step)
                sparsity_loss = sparsity_weight * sparsity_loss

            mask_loss +=  sparsity_loss
            mask_loss = mask_loss * self.mask_loss_weight

        else:
            # modified BG loss, eq.(19)
            BG_GT = inputs * inv_mask
            BG_rec = reconstructions * inv_mask
            BG_L1 = L1_loss(BG_GT.contiguous(), BG_rec.contiguous())
            bg_loss = self.BG_weight * BG_L1
            mask_loss += bg_loss

            FG_GT = inputs * mask_gen
            FG_rec = reconstructions * mask_gen
            FG_L1 = L1_loss(FG_GT.contiguous(), FG_rec.contiguous())
            fg_loss = self.FG_weight * FG_L1
            mask_loss += fg_loss

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean() + mask_loss
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/mask_loss".format(split): mask_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   'BG_loss': bg_loss.detach().mean(),
                   'mask_sparsity': sparsity_loss.detach().mean(),
                   'mask_binary_loss': binary_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
