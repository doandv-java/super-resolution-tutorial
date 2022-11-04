import torch
from models.srgan_trainer import SRGANTrainer
from models.eddr_trainer import EDSRTrainer
from models.config import SrGanConfig, EDSRConfig

# Constant
CONFIG_FILE = "data/config/edsr.json"
MODEL_NAME = "EDSR"

if __name__ == '__main__':

    if MODEL_NAME == 'SRGAN':
        config = SrGanConfig(config_file=CONFIG_FILE)
        trainer = SRGANTrainer(config=config)
        trainer.run()
    elif MODEL_NAME == 'EDSR':
        config = EDSRConfig(config_file=CONFIG_FILE)
        trainer = EDSRTrainer(config=config)
        trainer.run()
