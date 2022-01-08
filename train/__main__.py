from pytorch_lightning.utilities.cli import LightningCLI
from .lightning import DataModule, Module

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    LightningCLI(
            model_class=Module,
            datamodule_class=DataModule,
            save_config_callback=None,
        )
