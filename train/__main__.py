from .cli import MyLightningCLI
from .lightning import DataModule, Module

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    MyLightningCLI(
            model_class=Module,
            datamodule_class=DataModule,
            save_config_callback=None,
        )
