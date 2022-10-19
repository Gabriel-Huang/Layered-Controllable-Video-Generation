import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from modules.autoencoder.model import Encoder, Decoder
from modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from modules.vqvae.quantize import GumbelQuantize
import modules.vqvae.networks as networks
import torch.nn.functional as F

class VQARAutoMaskModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 finetune=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.resnet_mask = networks.ResnetGenerator(3, 1, 64, n_blocks=9)

        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key

        self.finetune = finetune

        if finetune:
            print('Training Phase II')
        else:
            print('Training Phase I')

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def rgb2gray(self, input):
        input = input.permute(0,2,3,1)
        gray = torch.matmul(input, torch.tensor([0.299, 0.587, 0.144]).cuda())
        return torch.unsqueeze(gray, 1)

    def forward(self, input, maskGT = None, input_b = None, finetune = False):

        # eq.(1) in paper:

        # training phase II: finetune for controllability
        if input_b != None:
            mask_b = self.resnet_mask(input_b)
            # binarize the softmasks:
            mask_b = torch.where(mask_b > 0.5, 1, 0) * 1.0
            mask = mask_b
            self.mask_gen = mask
        
        # training phase I: learning to generate masks:
        else:
            # during test time, mask will be provided as user input:
            if maskGT != None:
                mask = maskGT
                self.mask_gen = mask
            else:
                mask = self.resnet_mask(input)
                self.mask_gen = mask

        # eq.(3) in paper:
        inv_mask = torch.ones(mask.shape).cuda() - mask

        self.foreground = input * mask
        self.background = input * inv_mask

        quant_f, loss_f, _ = self.encode(self.foreground)
        quant_b, loss_b, _ = self.encode(self.background)

        gen_quant = quant_f + quant_b
        dec = self.decode(gen_quant)

        loss = loss_f + loss_b

        return dec, loss, self.mask_gen

    def get_input(self, batch, k):
        x = batch[k]
        if k == 'image' or k == 'image_B':
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            return x.float()
        else:
            return x

    def training_step(self, batch, batch_idx, optimizer_idx):

        # switch to stage II:
        if batch_idx >= 15000:
            self.finetune = True

        x = self.get_input(batch, self.image_key)
        x_next = self.get_input(batch, 'image_B')

        if self.finetune:
            # maskGT = self.get_input(batch, 'mask')
            maskGT = None
            # freeze mask net in Phase II
            for param in self.resnet_mask.parameters():
                param.requires_grad = False
            xrec, qloss, mask = self(x, input_b = x_next, finetune = self.finetune)

        else:
            xrec, qloss, mask = self(x, input_b = None)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x_next, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", mask_gen = mask, current = x,
                                            finetune = self.finetune)

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            self.log("bg_loss", log_dict_ae['BG_loss'], prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("sparsity_loss", log_dict_ae['mask_sparsity'], prog_bar=True, logger=True, on_step=True, on_epoch=True)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x_next, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", mask_gen = mask, current = x,
                                            finetune = self.finetune)

            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        x_next = self.get_input(batch, 'image_B')

        xrec, qloss, mask = self(x, maskGT = None , input_b = x_next)
        aeloss, log_dict_ae = self.loss(qloss, x_next, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val", mask_gen = mask, current = x)

        discloss, log_dict_disc = self.loss(qloss, x_next, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val", mask_gen = mask, current = x)
        rec_loss = log_dict_ae["val/rec_loss"]

        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.resnet_mask.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x_next = self.get_input(batch, 'image_B')

        x = x.to(self.device)
        x_next = x_next.to(self.device)

        xrec, qloss, mask = self(x, input_b = x_next)       
        if x.shape[1] > 3:
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        if self.finetune:
            log["inputs"] = x
            log["reconstructions"] = xrec
            log['masks'] = self.mask_gen
        else:
            log["reconstructions"] = xrec
            log['masks'] = self.mask_gen

        return log

    def to_rgb(self, x):
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


