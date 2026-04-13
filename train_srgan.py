import time
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

from models import Generator, Discriminator, TruncatedVGG19, SRResNet, ConvolutionalBlock
from datasets import SRDataset
from utils import AverageMeter, clip_gradient, compute_metrics

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

data_folder = "./"
crop_size = 96
scaling_factor = 4
batch_size = 16
workers = 4

# Generator
large_kernel_size_g = 9
small_kernel_size_g = 3
n_channels_g = 64
n_blocks_g = 16
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

# Discriminator
kernel_size_d = 3
n_channels_d = 64
n_blocks_d = 8
fc_size_d = 1024

# Training
iterations = 2e5
lr = 1e-4
grad_clip = None
vgg19_i, vgg19_j = 5, 4
beta = 1e-3

checkpoint_path = None

# Safe load globals for PyTorch 2.6+
torch.serialization.add_safe_globals([SRResNet, Generator, ConvolutionalBlock, nn.Sequential, nn.Conv2d])

# Load SRResNet safely
srresnet_model = torch.load(srresnet_checkpoint, map_location=device, weights_only=False)['model']
srresnet_model.eval()
srresnet_model.to(device)

# Helper functions
def normalize_for_tb(x):
    """
    Convert tensor from [-1,1] to [0,1] for TensorBoard

    Args:
        x: input tensor, typically an image batch, of any shape and with pixel values in the range [-1, 1]
    """
    if x.min() < 0:
        x = (x + 1) / 2.0
    return torch.clamp(x, 0, 1)


# Train function
def train_srgan(generator, discriminator, optimizer_g, optimizer_d,
                content_loss_criterion, adversarial_loss_criterion,
                truncated_vgg19, train_loader, epochs, writer):
    """
     Train the SRGAN model.
     
        Args:
            generator: the SRGAN Generator model to be trained
            discriminator: the SRGAN Discriminator model to be trained
            optimizer_g: optimizer for the generator
            optimizer_d: optimizer for the discriminator
            content_loss_criterion: loss function for the content loss (e.g. MSELoss)
            adversarial_loss_criterion: loss function for the adversarial loss (e.g. BCEWithLogitsLoss)
            truncated_vgg19: a truncated VGG19 model to extract features for the content loss
            train_loader: DataLoader for the training dataset
            epochs: number of epochs to train for
            writer: TensorBoard SummaryWriter for logging losses and metrics
     """

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_c = AverageMeter()
        losses_a = AverageMeter()
        losses_d = AverageMeter()
        start = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", ncols=120)

        for i, (lr_imgs, hr_imgs) in pbar:
            data_time.update(time.time() - start)

            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Generator update
            sr_imgs = generator(lr_imgs)
            sr_feats = truncated_vgg19(sr_imgs)
            hr_feats = truncated_vgg19(hr_imgs).detach()

            sr_disc = discriminator(sr_imgs)

            content_loss = content_loss_criterion(sr_feats, hr_feats)
            adv_loss = adversarial_loss_criterion(sr_disc, torch.ones_like(sr_disc))
            perceptual_loss = content_loss + beta * adv_loss

            optimizer_g.zero_grad()
            perceptual_loss.backward()
            if grad_clip is not None:
                clip_gradient(optimizer_g, grad_clip)
            optimizer_g.step()

            losses_c.update(content_loss.item(), lr_imgs.size(0))
            losses_a.update(adv_loss.item(), lr_imgs.size(0))

            # Discriminator update
            hr_disc = discriminator(hr_imgs)
            sr_disc_detach = discriminator(sr_imgs.detach())
            adv_loss_d = adversarial_loss_criterion(sr_disc_detach, torch.zeros_like(sr_disc_detach)) + \
                         adversarial_loss_criterion(hr_disc, torch.ones_like(hr_disc))

            optimizer_d.zero_grad()
            adv_loss_d.backward()
            if grad_clip is not None:
                clip_gradient(optimizer_d, grad_clip)
            optimizer_d.step()

            losses_d.update(adv_loss_d.item(), hr_imgs.size(0))

            # Logging & metrics
            global_step = epoch * len(train_loader) + i
            writer.add_scalar("Loss/Content_step", content_loss.item(), global_step)
            writer.add_scalar("Loss/Adv_G_step", adv_loss.item(), global_step)
            writer.add_scalar("Loss/Adv_D_step", adv_loss_d.item(), global_step)

            if i % 1000 == 0:
                lr_vis = normalize_for_tb(lr_imgs)
                hr_vis = normalize_for_tb(hr_imgs)
                sr_vis = normalize_for_tb(sr_imgs)

                writer.add_images("Images/LR", lr_vis, global_step)
                writer.add_images("Images/HR", hr_vis, global_step)
                writer.add_images("Images/SR", sr_vis, global_step)

                psnr_val, ssim_val = compute_metrics(sr_vis, hr_vis)
                writer.add_scalar("Metrics/PSNR", psnr_val, global_step)
                writer.add_scalar("Metrics/SSIM", ssim_val, global_step)

            batch_time.update(time.time() - start)
            start = time.time()

            pbar.set_postfix({
                "L_c": f"{losses_c.val:.4f}({losses_c.avg:.4f})",
                "L_aG": f"{losses_a.val:.4f}({losses_a.avg:.4f})",
                "L_aD": f"{losses_d.val:.4f}({losses_d.avg:.4f})"
            })

        # Epoch-level logging
        writer.add_scalar("Loss/Content_epoch", losses_c.avg, epoch)
        writer.add_scalar("Loss/Adv_G_epoch", losses_a.avg, epoch)
        writer.add_scalar("Loss/Adv_D_epoch", losses_d.avg, epoch)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'generator': generator,
            'discriminator': discriminator,
            'optimizer_g': optimizer_g,
            'optimizer_d': optimizer_d
        }, 'checkpoint_srgan.pth.tar')

# Main
if __name__ == "__main__":
    generator = Generator(
        large_kernel_size=large_kernel_size_g,
        small_kernel_size=small_kernel_size_g,
        n_channels=n_channels_g,
        n_blocks=n_blocks_g,
        scaling_factor=scaling_factor
    ).to(device)

    # Initialize generator weights from SRResNet
    generator.initialize_with_srresnet(srresnet_model=srresnet_model)

    optimizer_g = Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=lr)
    discriminator = Discriminator(
        kernel_size=kernel_size_d,
        n_channels=n_channels_d,
        n_blocks=n_blocks_d,
        fc_size=fc_size_d
    ).to(device)
    optimizer_d = Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr)

    content_loss_criterion = nn.MSELoss().to(device)
    adversarial_loss_criterion = nn.BCEWithLogitsLoss().to(device)
    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j).to(device).eval()

    train_dataset = SRDataset(
        data_folder,
        split='train',
        crop_size=crop_size,
        scaling_factor=scaling_factor,
        lr_img_type='imagenet-norm',
        hr_img_type='imagenet-norm'
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    epochs = int(iterations // len(train_loader) + 1)
    writer = SummaryWriter(log_dir="runs/srgan")

    train_srgan(generator, discriminator, optimizer_g, optimizer_d,
                content_loss_criterion, adversarial_loss_criterion,
                truncated_vgg19, train_loader, epochs, writer)

    writer.close()
    print("Training completed!")
