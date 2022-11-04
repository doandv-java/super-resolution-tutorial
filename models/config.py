from .datasets import ImageDataset
from torch.utils.data import DataLoader
from utils import common


class BaseConfig(object):

    def __init__(self, config_json):
        super(BaseConfig, self).__init__()
        self.train_data = config_json['train_data']
        self.epochs = config_json["epochs"]
        self.batch_size = config_json['batch_size']
        self.hr_height = config_json['hr_height']
        self.hr_width = config_json['hr_width']
        self.start_decay_epoch = config_json['start_decay_epoch']
        self.example_folder = config_json['example_folder']
        self.saved_model_folder = config_json['saved_model_folder']
        self.checkpoint_interval = config_json['checkpoint_interval']
        self.sample_interval = config_json['sample_interval']

    @property
    def hr_shape(self) -> tuple:
        return self.hr_height, self.hr_width

    @property
    def dataset(self):
        return ImageDataset(self.train_data, hr_shape=self.hr_shape)

    @property
    def dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


class SrGanConfig(BaseConfig):
    def __init__(self, config_file):
        config_data = common.load_file_json(config_file)
        super(SrGanConfig, self).__init__(config_data)
        self.lr = config_data['lr']
        self.b1 = config_data['b1']
        self.b2 = config_data['b2']


class EDSRConfig(BaseConfig):
    def __init__(self, config_file):
        config_data = common.load_file_json(config_file)
        super(EDSRConfig, self).__init__(config_data)
        self.lr = config_data['lr']
        self.b1 = config_data['b1']
        self.b2 = config_data['b2']
