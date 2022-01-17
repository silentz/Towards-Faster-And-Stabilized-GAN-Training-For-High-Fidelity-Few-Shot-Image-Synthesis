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

    input_sample = torch.randn(4, 256, 1, 1)

    if cli.config['from_ckpt'] is not None:
        cli.model = Module.load_from_checkpoint(cli.config['from_ckpt'])

    cli.model.to_torchscript(
            file_path=cli.config['torchscript']['path'],
            method='trace',
            example_inputs=input_sample,
        )
