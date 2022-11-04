import sys
from torch.utils.data import DataLoader
import torch
from torch.nn import functional
from torch.autograd import Variable
from models.sr_gan import Generator, Discriminator, FeatureExtractor
import numpy as np
from torchvision.utils import save_image, make_grid
import os


class SRGANTrainer(object):
    def __init__(self, config):
        super(SRGANTrainer, self).__init__()
        self.gpu_in_use = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.gpu_in_use else 'cpu')
        # Model
        self.generator = None
        self.discriminator = None
        self.feature_extractor = None
        # Config model
        self.criterionGAN = None
        self.criterion_content = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.lr = config.lr
        self.b1 = config.b1
        self.b2 = config.b2
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.start_decay_epoch = config.start_decay_epoch
        self.hr_width = config.hr_width
        self.hr_height = config.hr_height
        self.sample_interval = config.sample_interval
        self.checkpoint_interval = config.checkpoint_interval
        # Data for train
        self.data_loader = DataLoader(config.dataset, batch_size=self.batch_size, shuffle=True)
        self.example_folder_root = config.example_folder
        self.saved_model_folder = config.saved_model_folder

    def build_model(self):
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator(input_shape=(3, self.hr_height, self.hr_width)).to(self.device)
        self.feature_extractor = FeatureExtractor().to(self.device)
        # Set feature in inference model
        self.feature_extractor.eval()
        # Loss
        self.criterionGAN = torch.nn.MSELoss()
        self.criterion_content = torch.nn.L1Loss()
        # if use gpu
        if self.gpu_in_use:
            self.feature_extractor.cuda()
            self.criterion_content.cuda()
            self.criterionGAN.cuda()
            self.discriminator.cuda()
            self.generator.cuda()

        # Optimizer
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

    def train(self):
        Tensor = torch.cuda.FloatTensor if self.gpu_in_use else torch.Tensor
        for epoch in range(0, self.epochs):
            for i, images in enumerate(self.data_loader):
                lr_images = Variable(images['lr']).to(self.device)
                hr_images = Variable(images['hr']).to(self.device)
                valid = Variable(Tensor(np.ones((lr_images.size(0), *self.discriminator.output_shape))),
                                 requires_grad=False)
                fake = Variable(Tensor(np.zeros((lr_images.size(0), *self.discriminator.output_shape))),
                                requires_grad=False)
                # Train generator
                self.optimizer_G.zero_grad()
                sr_images = self.generator(lr_images)
                # GAN loss
                loss_gan = self.criterionGAN(self.discriminator(sr_images), valid)

                sr_features = self.feature_extractor(sr_images)
                hr_features = self.feature_extractor(hr_images)

                loss_content = self.criterion_content(sr_features, hr_features.detach())

                loss_generator = loss_content + 1e-3 * loss_gan

                loss_generator.backward()
                self.optimizer_G.step()

                # Train Discriminator

                self.optimizer_D.zero_grad()

                loss_real = self.criterionGAN(self.discriminator(hr_images), valid)
                loss_fake = self.criterionGAN(self.discriminator(sr_images.detach()), fake)
                loss_discriminator = (loss_real + loss_fake) / 2

                loss_discriminator.backward()
                self.optimizer_D.step()

                # Loss of real image and fake image
                sys.stdout.write("[Epoch {}/{}] [Batch {}/{}] [G loss :{}] [D loss:{}]\n".format(epoch, self.epochs, i,
                                                                                                 len(self.data_loader),
                                                                                                 loss_generator.item(),
                                                                                                 loss_discriminator.item()))
                batches_done = epoch * len(self.data_loader) + i
                # Save examples
                self.save_samples(batches_done, lr_images, sr_images, hr_images)

            # Save checkpoint
            self.save_checkpoint(epoch)

    def save_samples(self, batches_done, lr_images, sr_images, hr_images):
        # Check save dir exist
        os.makedirs(self.example_folder_root, exist_ok=True)
        os.makedirs(os.path.join(self.example_folder_root, 'lr_images'), exist_ok=True)
        os.makedirs(os.path.join(self.example_folder_root, 'hr_images'), exist_ok=True)
        os.makedirs(os.path.join(self.example_folder_root, 'sr_images'), exist_ok=True)
        os.makedirs(os.path.join(self.example_folder_root, 'example'), exist_ok=True)
        # save sample
        if batches_done % self.sample_interval == 0:
            # Create path
            lr_image_path = "%s/lr_images/%d.jpg" % (self.example_folder_root, batches_done)
            hr_image_path = "%s/hr_images/%d.jpg" % (self.example_folder_root, batches_done)
            sr_image_path = "%s/sr_images/%d.jpg" % (self.example_folder_root, batches_done)
            example_image_path = "%s/example/%d.jpg" % (self.example_folder_root, batches_done)
            # Save image
            save_image(lr_images, lr_image_path, normalize=False)
            save_image(hr_images, hr_image_path, normalize=False)
            lr_images = functional.interpolate(lr_images, scale_factor=4)
            sr_images = make_grid(sr_images, nrow=1, normalize=True)
            lr_images = make_grid(lr_images, nrow=1, normalize=True)
            hr_images = make_grid(hr_images, nrow=1, normalize=True)
            save_image(sr_images, sr_image_path, normalize=False)
            example_images = torch.cat((lr_images, sr_images, hr_images), -1)
            save_image(example_images, example_image_path, normalize=False)

    def save_checkpoint(self, epoch):
        os.makedirs(self.saved_model_folder, exist_ok=True)
        if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:
            torch.save(self.generator.state_dict(), "{}/generator_{}.pth".format(self.saved_model_folder, epoch))
            torch.save(self.discriminator.state_dict(),
                       "{}/discriminator_{}.pth".format(self.saved_model_folder, epoch))

    def test(self):
        self.generator.eval()
        avg_peak_signal_noise = 0
        with torch.no_grad():
            for i, images in enumerate(self.data_loader):
                lr_images = Variable(images['lr'])
                hr_images = Variable(images['hr'])

    def run(self):
        self.build_model()
        self.train()
