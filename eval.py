import torch
from utils import *
from datasets import SRDataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models.models_Conv2D import SRResNet, Generator
from models.models_FDConv import FDSRResNet, FD_Generator
import torch.serialization
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Allowlist SRResNet & SRGAN Generator for PyTorch >= 2.6
# torch.serialization.add_safe_globals([SRResNet, Generator])


def evaluate_model(model, data_folder="./", test_data_names=None):
    """
    Evaluate a given SR model on multiple datasets.

    Args:
        model: torch.nn.Module, SR model (SRResNet or SRGAN Generator).
        data_folder: str, root folder containing datasets.
        test_data_names: list[str], dataset names to test on.

    Returns:
        dict: {dataset_name: {"PSNR": value, "SSIM": value}}
    """
    if test_data_names is None:
        test_data_names = ["BSDS100"]

    results = {}

    for test_data_name in test_data_names:
        print(f"\nEvaluating on {test_data_name}...\n")

        test_dataset = SRDataset(data_folder,
                                 split="test",
                                 crop_size=0,
                                 scaling_factor=4,
                                 lr_img_type="imagenet-norm",
                                 hr_img_type="[-1, 1]",
                                 test_data_name=test_data_name)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
        )

        PSNRs = AverageMeter()
        SSIMs = AverageMeter()
        LPIPSs = AverageMeter()

        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(
                tqdm(test_loader, desc=f"{test_data_name}", unit="image")
            ):
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                # Forward pass
                sr_imgs = model(lr_imgs)

                # Convert to Y-channel
                sr_imgs_y = convert_image(sr_imgs, source="[-1, 1]", target="y-channel").squeeze(0)
                hr_imgs_y = convert_image(hr_imgs, source="[-1, 1]", target="y-channel").squeeze(0)

                # Metrics
                psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
                ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
                lpips = compute_lpips(sr_imgs, hr_imgs)

                PSNRs.update(psnr, lr_imgs.size(0))
                SSIMs.update(ssim, lr_imgs.size(0))
                LPIPSs.update(lpips.item(), lr_imgs.size(0))

                tqdm.write(f"Image {i+1}/{len(test_loader)} - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

        # Store results
        results[test_data_name] = {"PSNR": PSNRs.avg, "SSIM": SSIMs.avg, "LPIPS": LPIPSs.avg}

        print(f"\n{test_data_name} Results: PSNR {PSNRs.avg:.3f}, SSIM {SSIMs.avg:.3f}, LPIPS {LPIPSs.avg:.3f}\n")

    return results


if __name__ == "__main__":
    # Checkpoints
    srresnet_checkpoint = "./checkpoint_srresnet_2.pth.tar"
    fd_srresnet_checkpoint = "./checkpoint_fd_srresnet.pth.tar"
    srgan_checkpoint = "./checkpoint_srgan_2.pth.tar"
    fd_srgan_checkpoint = "./checkpoint_fd_srgan.pth.tar"

    # Load models
    srresnet = load_model(srresnet_checkpoint, key="model")
    fd_srresnet = load_model(fd_srresnet_checkpoint, key="model")
    srgan_generator = load_model(srgan_checkpoint, key="generator")
    fd_srgan_generator = load_model(fd_srgan_checkpoint, key="generator")

    # Evaluate both
    print("==== Evaluating SRResNet ====")
    resnet_results = evaluate_model(srresnet)

    print("==== Evaluating SRGAN ====")
    srgan_results = evaluate_model(srgan_generator)

    print("==== Evaluating FD-SRResNet ====")
    fd_resnet_results = evaluate_model(fd_srresnet)

    print("==== Evaluating FD-SRGAN ====")
    fd_srgan_results = evaluate_model(fd_srgan_generator)

    # Final summary
    print("\nSummary Results:")
    for dataset in resnet_results.keys():
        print(f"{dataset}:")
        print(f"  SRResNet    -> PSNR: {resnet_results[dataset]['PSNR']:.3f}, SSIM: {resnet_results[dataset]['SSIM']:.3f}, LPIPS: {resnet_results[dataset]['LPIPS']:.3f}")
        print(f"  SRGAN       -> PSNR: {srgan_results[dataset]['PSNR']:.3f}, SSIM: {srgan_results[dataset]['SSIM']:.3f}, LPIPS: {srgan_results[dataset]['LPIPS']:.3f}")
        print(f"  FD-SRResNet -> PSNR: {fd_resnet_results[dataset]['PSNR']:.3f}, SSIM: {fd_resnet_results[dataset]['SSIM']:.3f}, LPIPS: {fd_resnet_results[dataset]['LPIPS']:.3f}")
        print(f"  FD-SRGAN    -> PSNR: {fd_srgan_results[dataset]['PSNR']:.3f}, SSIM: {fd_srgan_results[dataset]['SSIM']:.3f}, LPIPS: {fd_srgan_results[dataset]['LPIPS']:.3f}")