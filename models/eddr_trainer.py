from __future__ import print_function

import os

import torch
import torch.backends.cudnn as cudnn

from models.edsr import EDSR
from models.config import EDSRConfig
from torch.utils.data import DataLoader
from torch.autograd import Variable


class EDSRTrainer(object):
    def __init__(self, config: EDSRConfig):
        super(EDSRTrainer, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        # Config for train
        self.lr = config.lr
        self.b1 = config.b1
        self.b2 = config.b2
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        # Config for model
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.saved_model_folder = config.saved_model_folder
        self.training_loader = DataLoader(config.dataset, batch_size=self.batch_size, shuffle=True)

    def build_model(self):
        self.model = EDSR(num_channels=3, upscale_factor=4, base_channel=64, num_residuals=4).to(
            self.device)
        self.criterion = torch.nn.L1Loss()

        if self.GPU_IN_USE:
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100],
                                                              gamma=0.5)  # lr decay

    def save(self):
        model_out_path = os.path.join(self.saved_model_folder, "EDSR_bug.pth")
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, images in enumerate(self.training_loader):
            lr_images = Variable(images['lr']).to(self.device)
            hr_images = Variable(images['hr']).to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(lr_images), hr_images)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            print("{}/{}:Loss:{}\n".format(batch_num, len(self.training_loader), loss.item()))

    def run(self):
        self.build_model()
        for epoch in range(1, self.epochs + 1):
            print("\n===> Epoch {} starts:\n".format(epoch))
            self.train()
            self.scheduler.step(epoch)
