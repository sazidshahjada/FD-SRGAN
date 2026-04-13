import torch
from utils import *
from PIL import Image, ImageDraw, ImageFont
from models import SRResNet, Generator, ConvolutionalBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

# Allowlist SRResNet, SRGAN Generator, and ConvolutionalBlock
torch.serialization.add_safe_globals([SRResNet, Generator, ConvolutionalBlock])

# Load models (force weights_only=False to load pickled objects safely)
srresnet = torch.load(srresnet_checkpoint, map_location=device, weights_only=False)['model'].to(device)
srresnet.eval()
srgan_generator = torch.load(srgan_checkpoint, map_location=device, weights_only=False)['generator'].to(device)
srgan_generator.eval()


def infer_and_visualize(model, img_path, title="Model Output", halve=False):
    """
    Run inference with a given SR model and visualize comparison.

    Args:
        model (torch.nn.Module): Super-resolution model (e.g., SRResNet or SRGAN Generator).
        img_path (str): Path to the high-resolution image.
        title (str): Label for the model output in the visualization.
        halve (bool): If True, halves HR image size before processing (to fit screen).
    """
    # Load and prepare HR image
    hr_img = Image.open(img_path).convert("RGB")
    if halve:
        hr_img = hr_img.resize((hr_img.width // 2, hr_img.height // 2), Image.LANCZOS)

    # Create LR image (downsample by 4)
    lr_img = hr_img.resize((hr_img.width // 4, hr_img.height // 4), Image.BICUBIC)

    # Bicubic upsampling back to HR size
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    # Super-resolve with model
    with torch.no_grad():
        inp = convert_image(lr_img, source="pil", target="imagenet-norm").unsqueeze(0).to(device)
        sr_img = model(inp).squeeze(0).cpu().detach()
        sr_img = convert_image(sr_img, source="[-1, 1]", target="pil")

    # Create visualization grid (3 images: LR-Bicubic | Model SR | HR)
    margin = 30
    grid_width = 3 * hr_img.width + 4 * margin
    grid_height = hr_img.height + 2 * margin
    grid_img = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    # Add font
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("calibril.ttf", size=20)
    except OSError:
        font = ImageFont.load_default()

    # Paste images
    grid_img.paste(bicubic_img, (margin, margin))
    draw.text((margin + bicubic_img.width // 2 - 30, 5), "Bicubic", font=font, fill="black")

    grid_img.paste(sr_img, (2 * margin + hr_img.width, margin))
    draw.text((2 * margin + hr_img.width + sr_img.width // 2 - 40, 5),
              title, font=font, fill="black")

    grid_img.paste(hr_img, (3 * margin + 2 * hr_img.width, margin))
    draw.text((3 * margin + 2 * hr_img.width + hr_img.width // 2 - 40, 5),
              "Original HR", font=font, fill="black")

    # Show visualization
    grid_img.show()

    return grid_img



def visualize_sr(img, halve=False):
    """
    Visualizes the super-resolved images from the SRResNet and SRGAN for comparison with the bicubic-upsampled image
    and the original high-resolution (HR) image, as done in the paper.

    Args:
        img: path to the original HR image to be super-resolved and visualized
        halve: whether to halve the dimensions of the original HR image before super-resolving and visualizing (this is just to make the visualization fit on the screen better, since the original HR images in the test datasets are quite large)
    """
    # Load image, downsample to obtain low-res version
    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert('RGB')
    if halve:
        hr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
                               Image.LANCZOS)
    lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
                           Image.BICUBIC)

    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    # Super-resolution (SR) with SRResNet
    sr_img_srresnet = srresnet(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')

    # Super-resolution (SR) with SRGAN
    sr_img_srgan = srgan_generator(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

    # Create grid
    margin = 40
    grid_img = Image.new('RGB', (2 * hr_img.width + 3 * margin, 2 * hr_img.height + 3 * margin), (255, 255, 255))

    # Font
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("calibril.ttf", size=23)
        # It will also look for this file in your OS's default fonts directory, where you may have the Calibri Light font installed if you have MS Office
        # Otherwise, use any TTF font of your choice
    except OSError:
        print(
            "Defaulting to a terrible font. To use a font of your choice, include the link to its TTF file in the function.")
        font = ImageFont.load_default()

    # Place bicubic-upsampled image
    grid_img.paste(bicubic_img, (margin, margin))
    text_size = font.getsize("Bicubic")
    draw.text(xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, margin - text_size[1] - 5], text="Bicubic",
              font=font,
              fill='black')

    # Place SRResNet image
    grid_img.paste(sr_img_srresnet, (2 * margin + bicubic_img.width, margin))
    text_size = font.getsize("SRResNet")
    draw.text(
        xy=[2 * margin + bicubic_img.width + sr_img_srresnet.width / 2 - text_size[0] / 2, margin - text_size[1] - 5],
        text="SRResNet", font=font, fill='black')

    # Place SRGAN image
    grid_img.paste(sr_img_srgan, (margin, 2 * margin + sr_img_srresnet.height))
    text_size = font.getsize("SRGAN")
    draw.text(
        xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, 2 * margin + sr_img_srresnet.height - text_size[1] - 5],
        text="SRGAN", font=font, fill='black')

    # Place original HR image
    grid_img.paste(hr_img, (2 * margin + bicubic_img.width, 2 * margin + sr_img_srresnet.height))
    text_size = font.getsize("Original HR")
    draw.text(xy=[2 * margin + bicubic_img.width + sr_img_srresnet.width / 2 - text_size[0] / 2,
                  2 * margin + sr_img_srresnet.height - text_size[1] - 1], text="Original HR", font=font, fill='black')

    # Display grid
    grid_img.show()

    return grid_img



if __name__ == '__main__':
    img_path = "/media/iot/HDD2TB/sr_eval/urban100/img_003.png"
    # grid_img = visualize_sr("/media/iot/HDD2TB/sr_eval/General100/im_006.png")
    grid_img = infer_and_visualize(srresnet, img_path, title="SRResNet", halve=True)