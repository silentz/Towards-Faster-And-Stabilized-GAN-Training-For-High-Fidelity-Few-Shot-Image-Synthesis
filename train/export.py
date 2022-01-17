import torch
from .lightning import DataModule, Module
from .cli import MyLightningCLI

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    cli = MyLightningCLI(
            model_class=Module,
            datamodule_class=DataModule,
            save_config_callback=None,
            run=False,
        )

    input_sample = torch.rand(1, 256, 1, 1)

    cli.model.to_torchscript(
            file_path=cli.config['torchscript']['path'],
            method='script',
            example_inputs=input_sample,
        )
