from config import *
from data import EPIDataset, SimDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import EDSR, NoiseNetwork
import nibabel as nib
import numpy as np
from time import time
from utils import MovingAverage, rician_correct, mkdir, psnr, ssim
import torch.multiprocessing as mp

if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)

    assert 0 <= use_k < num_split and num_split % 2 == 0
    mkdir("../results/" + model_name + "/outputs")

    # model
    if is_denoise:
        model = NoiseNetwork(in_channels=1, out_channels=1, blindspot=True).cuda()
    else:
        model = EDSR(cin=2 * use_k + 1, n_resblocks=16, n_feats=64, res_scale=1).cuda()

        #model.load_state_dict(
        #    torch.load("../results/" + model_name + "/" + model_name + ".pt")
        #)

    if denoiser is not None:
        denoiser_name = denoiser
        denoiser = NoiseNetwork(in_channels=1, out_channels=1, blindspot=True)
        denoiser.load_state_dict(torch.load(denoiser_name))

    # dataset
    Dataset = SimDataset if use_sim else EPIDataset
    train_dataset = Dataset(num_split, "train", is_denoise, denoiser)
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=False, pin_memory=True
    )
    dataiter = iter(train_dataloader)
    test_dataset = Dataset(num_split, "test", is_denoise, denoiser)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, pin_memory=True)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    average = MovingAverage(0.999)

    t_start = time()
    for i in range(1, num_iter + 1):

        lr, hr = next(dataiter)
        lr = lr.cuda()
        hr = hr.cuda()

        if not is_denoise:
            out = model(lr[:, num_split - 1 - use_k : num_split + use_k])
            loss = torch.mean(torch.abs(hr[:, num_split - 1 : num_split] - out))
            loss_b = torch.mean(
                torch.abs(
                    hr[:, num_split - 1 : num_split] - lr[:, num_split - 1 : num_split]
                )
            )
            average("loss", loss.item())
            average("loss_b", loss_b.item())
        else:
            out = model(hr)
            loss = torch.mean((hr - out) ** 2)
            average("loss", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("i = %d, %s, time = %d" % (i, average, time() - t_start))

        if i % 1000 == 0 or i == num_iter:
            torch.save(
                model.state_dict(),
                "../results/" + model_name + "/" + model_name + ".pt",
            )

            average_test = MovingAverage(0)

            for j, data in enumerate(test_dataloader):

                if is_denoise:
                    img = data[0].cuda().permute(3, 0, 1, 2)
                    if use_sim:
                        gt = data[1].cuda().permute(3, 0, 1, 2)
                else:
                    img = data[0][0].cuda()
                    img_mid = img[num_split - 1]
                    img = img.permute(2, 0, 3, 1)
                    combined = data[-1][0].cuda()
                    gt = data[1][0].cuda()

                with torch.no_grad():
                    if is_denoise:
                        out = model(img)
                    else:
                        out = (
                            model(img[:, num_split - 1 - use_k : num_split + use_k])
                            .squeeze()
                            .permute(2, 0, 1)
                        )

                    if is_denoise:
                        if use_sim:
                            average_test(
                                "mse", ((out - gt) ** 2)[gt > 0.01].mean().item()
                            )
                            np.save(
                                "../results/" + model_name + "/outputs/gt_%d" % j,
                                gt.cpu().numpy(),
                            )
                        else:
                            np.save(
                                "../results/" + model_name + "/outputs/in_%d" % j,
                                img.cpu().numpy(),
                            )
                        np.save(
                            "../results/" + model_name + "/outputs/out_%d" % j,
                            out.cpu().numpy(),
                        )
                    else:
                        if use_sim:
                            out = rician_correct(
                                out,
                                None if sigma and (denoiser is not None) else 0,
                                gt < 0.01,
                            )

                            average_test(
                                "mse_cubic",
                                torch.sqrt(
                                    ((gt - img_mid) ** 2).mean() / (gt**2).mean()
                                ).item(),
                            )
                            average_test(
                                "mse_out",
                                torch.sqrt(
                                    ((gt - out) ** 2).mean() / (gt**2).mean()
                                ).item(),
                            )
                            average_test(
                                "mse_combined",
                                torch.sqrt(
                                    ((gt - combined) ** 2).mean() / (gt**2).mean()
                                ).item(),
                            )
                            average_test(
                                "mm_cubic",
                                ((gt - img_mid) ** 2)[gt > 0.01].mean().item(),
                            )
                            average_test(
                                "mm_out", ((gt - out) ** 2)[gt > 0.01].mean().item()
                            )
                            average_test(
                                "mm_combined",
                                ((gt - combined) ** 2)[gt > 0.01].mean().item(),
                            )

                            nib.save(
                                nib.Nifti1Image(out.cpu().numpy() * 1000, np.eye(4)),
                                "../results/"
                                + model_name
                                + "/outputs/out_%d.nii.gz" % j,
                            )
                            nib.save(
                                nib.Nifti1Image(gt.cpu().numpy() * 1000, np.eye(4)),
                                "../results/"
                                + model_name
                                + "/outputs/gt_%d.nii.gz" % j,
                            )
                            nib.save(
                                nib.Nifti1Image(
                                    img_mid.cpu().numpy() * 1000, np.eye(4)
                                ),
                                "../results/"
                                + model_name
                                + "/outputs/in_%d.nii.gz" % j,
                            )
                            nib.save(
                                nib.Nifti1Image(
                                    combined.cpu().numpy() * 1000, np.eye(4)
                                ),
                                "../results/"
                                + model_name
                                + "/outputs/combined_%d.nii.gz" % j,
                            )

                        else:
                            out = out * 100 + 70
                            out[out < 0] = 0
                            img_mid = img_mid * 100 + 70
                            combined = combined * 100 + 70
                            gt = gt * 100 + 70

                            out = out.cpu().numpy()
                            img_mid = img_mid.cpu().numpy()
                            combined = combined.cpu().numpy()
                            gt = gt.cpu().numpy()
                            sti = (img_mid + combined) / 2

                            if num_split == 4:
                                mask = gt > 0
                                average_test("psnr_si", psnr(img_mid, gt, mask))
                                average_test("psnr_ti", psnr(combined, gt, mask))
                                average_test("psnr_sti", psnr(sti, gt, mask))
                                average_test("psnr_out", psnr(out, gt, mask))

                                nib.save(
                                    nib.Nifti1Image(gt, np.eye(4)),
                                    "../results/"
                                    + model_name
                                    + "/outputs/gt_%d.nii.gz" % j,
                                )

                            nib.save(
                                nib.Nifti1Image(out, np.eye(4)),
                                "../results/"
                                + model_name
                                + "/outputs/out_%d.nii.gz" % j,
                            )
                            nib.save(
                                nib.Nifti1Image(combined, np.eye(4)),
                                "../results/"
                                + model_name
                                + "/outputs/combined_%d.nii.gz" % j,
                            )
                            nib.save(
                                nib.Nifti1Image(img_mid, np.eye(4)),
                                "../results/"
                                + model_name
                                + "/outputs/in_%d.nii.gz" % j,
                            )

            print("%d, %s" % (i // 1000, average_test))
