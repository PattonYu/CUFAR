"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import numpy as np
import torch
import torch.nn.functional as F

def apply_augment(fine, coarse):    
    aug_methods = ["mixup", "cutmix", "cutmixup", "cutout", "blend",]
    idx = np.random.choice(len(aug_methods), p= None)
    aug = aug_methods[idx]
    prob_dict = {'blend': 1.0, 'mixup':1.0, 'cutout':1.0, 'cutmix': 1.0, 'cutmixup': 1.0}
    alpha_dict = {'blend': 0.6, 'mixup':1.2, 'cutout':0.001, 'cutmix': 0.7, 'cutmixup': 0.7}

    prob = float(prob_dict[aug])
    alpha = float(alpha_dict[aug])
    aux_prob, aux_alpha = 1.0, 1.2
    mask = None

    if aug == "none":
        fine_aug, coarse_aug = fine.clone(), coarse.clone()
    elif aug == "blend":
        fine_aug, coarse_aug = blend(
            fine.clone(), coarse.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "mixup":
        fine_aug, coarse_aug, = mixup(
            fine.clone(), coarse.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutout":
        fine_aug, coarse_aug, mask, _ = cutout(
            fine.clone(), coarse.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "cutmix":
        fine_aug, coarse_aug = cutmix(
            fine.clone(), coarse.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutmixup":
        fine_aug, coarse_aug = cutmixup(
            fine.clone(), coarse.clone(),
            mixup_prob=aux_prob, mixup_alpha=aux_alpha,
            cutmix_prob=prob, cutmix_alpha=alpha,
        )
    else:
        raise ValueError("{} is not invalid.".format(aug))

    return fine_aug, coarse_aug, mask, aug


def blend(fine, coarse, prob=1.0, alpha=0.6):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return fine, coarse

    c = torch.empty((coarse.size(0), 1, 1, 1), device=coarse.device)
    rcoarse = c.repeat((1, 1, coarse.size(2), coarse.size(3)))
    rfine = c.repeat((1, 1, fine.size(2), fine.size(3)))
    v = np.random.uniform(alpha, 1)

    fine = v * fine + (1-v) * rfine
    coarse = v * coarse + (1-v) * rcoarse

    return fine, coarse


def mixup(fine, coarse, prob=1.0, alpha=1.2):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return fine, coarse

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(fine.size(0)).to(coarse.device)

    fine = v * fine + (1-v) * fine[r_index, :]
    coarse = v * coarse + (1-v) * coarse[r_index, :]
    return fine, coarse


def _cutmix(coarse, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = coarse.size(2), coarse.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)

    fcy = np.random.randint(0, h-ch+1)
    fcx = np.random.randint(0, w-cw+1)
    tcy, tcx = fcy, fcx
    rindex = torch.randperm(coarse.size(0)).to(coarse.device)

    return {
        "rindex": rindex, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }


def cutmix(fine, coarse, prob=1.0, alpha=1.0):
    c = _cutmix(coarse, prob, alpha)
    if c is None:
        return fine, coarse

    scale = fine.size(2) // coarse.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    coarse[..., tcy:tcy+ch, tcx:tcx+cw] = coarse[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
    fine[..., htcy:htcy+hch, htcx:htcx+hcw] = fine[rindex, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]

    return fine, coarse


def cutmixup(fine, coarse,
    mixup_prob=1.0, mixup_alpha=1.0,
    cutmix_prob=1.0, cutmix_alpha=1.0):
    c = _cutmix(coarse, cutmix_prob, cutmix_alpha)
    if c is None:
        return fine, coarse

    scale = fine.size(2) // coarse.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        coarse_aug = coarse[rindex, :]
        fine_aug = fine[rindex, :]

    else:
        coarse_aug = v * coarse + (1-v) * coarse[rindex, :]
        fine_aug = v * fine + (1-v) * fine[rindex, :]
    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        coarse[..., tcy:tcy+ch, tcx:tcx+cw] = coarse_aug[..., fcy:fcy+ch, fcx:fcx+cw]
        fine[..., htcy:htcy+hch, htcx:htcx+hcw] = fine_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
    else:
        coarse_aug[..., tcy:tcy+ch, tcx:tcx+cw] = coarse[..., fcy:fcy+ch, fcx:fcx+cw]
        fine_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = fine[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        coarse, fine = coarse_aug, fine_aug

    return fine, coarse


def cutout(fine, coarse, prob=1.0, alpha=0.1):
    scale = fine.size(2) // coarse.size(2)
    fsize = (coarse.size(0), 1)+coarse.size()[2:]

    if alpha <= 0 or np.random.rand(1) >= prob:
        fcoarse = np.ones(fsize)
        fcoarse = torch.tensor(fcoarse, dtype=torch.float, device=coarse.device)
        ffine = F.interpolate(fcoarse, scale_factor=scale, mode="nearest")
        return fine, coarse, ffine, fcoarse

    fcoarse = np.random.choice([0.0, 1.0], size=fsize, p=[alpha, 1-alpha])
    fcoarse = torch.tensor(fcoarse, dtype=torch.float, device=coarse.device)
    ffine = F.interpolate(fcoarse, scale_factor=scale, mode="nearest")

    coarse *= fcoarse

    return fine, coarse, ffine, fcoarse