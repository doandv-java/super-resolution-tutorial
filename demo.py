import os

import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from torchvision.utils import make_grid, save_image
from models.sr_gan import Generator
from utils import common

# Constant
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
IMG_PATH = "data/test/test.jpg"
MODEL_CHECKPOINT_PATH = "data/saved_models/generator_10.pth"
prepare_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


def sr_image(generator, img_path, sr_dir_path: str):
    # Load image
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_data = prepare_transform(img)
    img_data = torch.unsqueeze(img_data, 0)
    # Predict
    sr_image = generator(img_data)
    sr_image = make_grid(sr_image, nrow=1, normalize=True)
    os.makedirs(sr_dir_path, exist_ok=True)
    sr_path = os.path.join(sr_dir_path, img_name)
    save_image(sr_image, sr_path)


def sr_image_patch(model, img_path, patch_size):
    img = cv2.imread(img_path)
    print("====================Create Patches ====================")
    patch_dir = "data/patches"
    sr_dir_path = "data/sr_image"
    assemble_dir_path = "data/assemble"
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(sr_dir_path, exist_ok=True)
    os.makedirs(assemble_dir_path, exist_ok=True)
    patches = common.create_patches(img, patch_size)
    patch_list = []
    for i, patch in enumerate(patches):
        patch_name = "%0.4d.jpg" % i
        print(patch_name)
        patch_path = os.path.join(patch_dir, patch_name)
        patch_list.append(patch_path)
        cv2.imwrite(patch_path, patch)
    print("SR Patches")
    for patch_path in patch_list:
        print(os.path.basename(patch_path))
        sr_image(model, patch_path, sr_dir_path=sr_dir_path)
    print("Load SR image")
    sr_path_list = []
    sr_images = []
    common.get_img_from_directory(sr_path_list, sr_dir_path)
    sr_path_list.sort()
    for sr_path in sr_path_list:
        sr = cv2.imread(sr_path)
        sr_images.append(sr)
    print("Assemble")
    mat_size = img.shape
    assemble = common.assemble_patches(sr_images, mat_size, patch_size, up_size=4)
    cv2.imwrite(os.path.join(assemble_dir_path, os.path.basename(img_path)), assemble)
    # Clear
    print("Clear")
    common.clear_folder(patch_dir)
    common.clear_folder(sr_dir_path)


if __name__ == '__main__':
    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # Load model
    generator = Generator()
    model_dict = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
    generator.load_state_dict(model_dict)
    generator.eval()
    ##Sr patch
    sr_image_patch(generator, IMG_PATH, 384)

    # model = model.to(device)
    # sr_image(generator, IMG_PATH, "data/sr_image")
    # sr_image = torch.squeeze(sr_image)
    # sr_image = sr_image.mul(255).add_(0.5).clamp_(0, 255).to(device, torch.uint8).numpy()
    # cv2.imwrite("result.jpg", sr_image)
    # Close
    exit(0)
