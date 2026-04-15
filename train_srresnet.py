import time
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

from models import SRResNet, ConvolutionalBlock
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

# Model parameters
large_kernel_size = 9
small_kernel_size = 3
n_channels = 64
n_blocks = 16

# Training parameters
iterations = 2e5
lr = 1e-4
grad_clip = None
checkpoint = None  # Path to checkpoint if resuming

# Safe load globals for PyTorch 2.6+
# torch.serialization.add_safe_globals([SRResNet, ConvolutionalBlock, nn.Sequential, nn.Conv2d])

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
def train_srresnet(model, optimizer, criterion, train_loader, epochs, writer, start_epoch=0):
    """
    Train the SRResNet model.

    Args:
        model: the SRResNet model to train
        optimizer: the optimizer to use for training
        criterion: the loss function to optimize
        train_loader: DataLoader for the training dataset
        epochs: total number of epochs to train for
        writer: TensorBoard SummaryWriter for logging
        start_epoch: epoch to start training from (useful for resuming from checkpoint)
    """
    model.train()

    for epoch in range(start_epoch, epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        start = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", ncols=120)

        for i, (lr_imgs, hr_imgs) in pbar:
            data_time.update(time.time() - start)

            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Forward pass
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
            
            optimizer.step()

            # Track loss
            losses.update(loss.item(), lr_imgs.size(0))

            # Logging & metrics
            global_step = epoch * len(train_loader) + i
            writer.add_scalar("Loss/train_step", loss.item(), global_step)

            if i % 1000 == 0:
                lr_vis = normalize_for_tb(lr_imgs)
                hr_vis = normalize_for_tb(hr_imgs)
                sr_vis = normalize_for_tb(sr_imgs)

                writer.add_images("Images/LR", lr_vis, global_step)
                writer.add_images("Images/HR", hr_vis, global_step)
                writer.add_images("Images/SR", sr_vis, global_step)

                # Assuming compute_metrics returns (psnr, ssim)
                psnr_val, ssim_val = compute_metrics(sr_vis, hr_vis)
                writer.add_scalar("Metrics/PSNR", psnr_val, global_step)
                writer.add_scalar("Metrics/SSIM", ssim_val, global_step)

            batch_time.update(time.time() - start)
            start = time.time()

            pbar.set_postfix({
                "Loss": f"{losses.val:.4f}({losses.avg:.4f})",
                "Batch_Time": f"{batch_time.val:.3f}s"
            })

        # Epoch-level logging
        writer.add_scalar("Loss/train_epoch", losses.avg, epoch)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
        }, 'checkpoint_srresnet.pth.tar')



# Main
if __name__ == "__main__":
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir="runs/srresnet")

    # Initialize model or load from checkpoint
    if checkpoint is None:
        model = SRResNet(
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            n_channels=n_channels,
            n_blocks=n_blocks,
            scaling_factor=scaling_factor
        ).to(device)
        
        optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        start_epoch = 0
    else:
        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
        start_epoch = checkpoint_data['epoch'] + 1
        model = checkpoint_data['model'].to(device)
        optimizer = checkpoint_data['optimizer']

    criterion = nn.MSELoss().to(device)

    # Dataset & Loader
    train_dataset = SRDataset(
        data_folder,
        split='train',
        crop_size=crop_size,
        scaling_factor=scaling_factor,
        lr_img_type='imagenet-norm',
        hr_img_type='[-1, 1]' # Consistent with SRGAN HR expectations
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    epochs = int(iterations // len(train_loader) + 1)

    print(f"Starting SRResNet training from epoch {start_epoch}...")
    train_srresnet(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        epochs=epochs,
        writer=writer,
        start_epoch=start_epoch
    )

    writer.close()
    print("SRResNet Training completed!")