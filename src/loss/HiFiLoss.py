import torch
from src.preprocessing.melspec import MelSpectrogram
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(self, fm_coef=2, mel_coef=45):
        super().__init__()
        self.fm_coef = fm_coef
        self.mel_coef = mel_coef
        self.l1_loss = nn.L1Loss()
        self.mel_transform = MelSpectrogram()

    def forward(self, 
                spectrogram, 
                gened,
                pg_outs,
                pr_feat,
                pg_feat,
                sg_outs,
                sr_feat,
                sg_feat,
                **kwargs):
        gened = gened.squeeze(1)
        generated_spectrogram = self.mel_transform(gened) 
        if spectrogram.shape[-1] < generated_spectrogram.shape[-1]:
            diff = generated_spectrogram.shape[-1] - spectrogram.shape[-1]
            pad_value = self.mel_transform.config.pad_value
            pad = torch.zeros((spectrogram.shape[0], spectrogram.shape[1], diff))
            pad = pad.fill_(pad_value).to(spectrogram.device)
            spectrogram = torch.cat([spectrogram, pad], dim=-1)
        adv_loss = 0
        for p in pg_outs:
            adv_loss = adv_loss + torch.mean((p - 1) ** 2)
        for s in sg_outs:
            adv_loss = adv_loss + torch.mean((s - 1) ** 2)
        fm_loss = 0
        for real, gen in zip(pr_feat, pg_feat):
            fm_loss = fm_loss + self.l1_loss(gen, real)
        for real, gen in zip(sr_feat, sg_feat):
            fm_loss = fm_loss + self.l1_loss(gen, real)
        mel_loss = self.l1_loss(generated_spectrogram, spectrogram)
        G_loss = adv_loss + self.fm_coef * fm_loss + self.mel_coef * mel_loss
        return G_loss, adv_loss, fm_loss, mel_loss

class DescriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pr_outs,
        pg_outs,
        sr_outs,
        sg_outs,
        **kwargs):
        D_loss = 0
        for pr, pg in zip(pr_outs, pg_outs):
            D_loss = D_loss + torch.mean((pr - 1) ** 2) + torch.mean((pg - 0) ** 2)
        for sr, sg in zip(sr_outs, sg_outs):
            D_loss = D_loss + torch.mean((sr - 1) ** 2) + torch.mean((sg - 0) ** 2)
        return D_loss
