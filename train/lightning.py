import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import lpips
import wandb
import random
import pytorch_lightning as pl
from typing import Any, Dict, List

from src.collate import collate_fn, Batch
from src.models import Generator, Discriminrator
from src.augment import DiffAugment
from src.utils import crop_image_part, init_weights


class DataModule(pl.LightningDataModule):

    def __init__(self, train_dataset: Dataset,
                       train_batch_size: int,
                       train_num_workers: int,
                       val_dataset: Dataset,
                       val_batch_size: int,
                       val_num_workers: int):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataloader_kwargs = {
                'batch_size': train_batch_size,
                'num_workers': train_num_workers,
                'collate_fn': collate_fn,
            }
        self.val_dataloader_kwargs = {
                'batch_size': val_batch_size,
                'num_workers': val_num_workers,
            }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)


class Module(pl.LightningModule):

    def __init__(self, optimizer_lr: float,
                       optimizer_beta_1: float,
                       optimizer_beta_2: float):
        super().__init__()
        self.automatic_optimization = False

        self._in_channels = 256
        self._optim_lr = optimizer_lr
        self._optim_beta_1 = optimizer_beta_1
        self._optim_beta_2 = optimizer_beta_2

        self.diff_aug = DiffAugment()
        self.percept_loss = lpips.LPIPS(net='vgg', eval_mode=True)

        self.generator = Generator(in_channels=self._in_channels, out_channels=3)
        self.discriminator = Discriminrator(in_channels=3)

        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)

    def configure_optimizers(self):
        gen_optim = torch.optim.SGD(
                self.generator.parameters(),
                lr=self._optim_lr,
                momentum=0.9,
            )
        dis_optim = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self._optim_lr,
                betas=(self._optim_beta_1, self._optim_beta_2),
            )
        return [dis_optim, gen_optim]

    def _normalize(self, input: torch.Tensor) -> torch.Tensor:
        return (input - 127.5) / 127.5

    def _denormalize(self, input: torch.Tensor) -> torch.Tensor:
        return (input * 127.5) + 127.5

    def training_step(self, batch: Batch, batch_idx: int) -> None:
        dis_optim, gen_optim = self.optimizers()
        real_images = self._normalize(batch.images)

        noise = torch.zeros(real_images.shape[0], self._in_channels, 1, 1,
                device=real_images.device).normal_(0.0, 1.0)
        fake_images_1024, fake_images_128 = self.generator(noise)

        # diff augment
        real_images = self.diff_aug(real_images)
        fake_images_1024 = self.diff_aug(fake_images_1024)
        fake_images_128 = self.diff_aug(fake_images_128)

        # discriminator step real
        dis_optim.zero_grad()

        image_type = random.choice([
                Discriminrator.ImageType.REAL_UP_L,
                Discriminrator.ImageType.REAL_UP_R,
                Discriminrator.ImageType.REAL_DOWN_R,
                Discriminrator.ImageType.REAL_DOWN_L,
            ])

        images_part = crop_image_part(real_images, image_type)
        disc_out, (dec_large, dec_small, dec_piece) = self.discriminator(real_images,
                                                                         F.interpolate(real_images, size=128),
                                                                         image_type)

        disc_real_loss = F.relu(torch.rand_like(disc_out) * 0.2 + 0.8 - disc_out).mean() + \
                         self.percept_loss(dec_large, F.interpolate(real_images, dec_large.shape[2])).sum() + \
                         self.percept_loss(dec_small, F.interpolate(real_images, dec_small.shape[2])).sum() + \
                         self.percept_loss(dec_piece, F.interpolate(images_part, dec_piece.shape[2])).sum()

        self.manual_backward(disc_real_loss)

        # discriminator step fake
        disc_out = self.discriminator(fake_images_1024.detach(),
                                      fake_images_128.detach(),
                                      Discriminrator.ImageType.FAKE)
        disc_fake_loss = F.relu(torch.rand_like(disc_out) * 0.2 + 0.8 + disc_out).mean()

        self.manual_backward(disc_fake_loss)
        dis_optim.step()

        # generator step
        gen_optim.zero_grad()

        disc_out = self.discriminator(fake_images_1024,
                                      fake_images_128,
                                      Discriminrator.ImageType.FAKE)
        gen_loss = -disc_out.mean()

        self.manual_backward(gen_loss)
        gen_optim.step()

        self.log('gen_loss', gen_loss.item())
        self.log('disc_fake_loss', disc_fake_loss.item())
        self.log('disc_real_loss', disc_real_loss.item())

    def validation_step(self, noise: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        fake_images, _ = self.generator(noise)
        fake_images = self._denormalize(fake_images)

        return {
                'images': fake_images.detach(),
            }

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]):
        images = []

        for output in outputs:
            for image in output['images']:
                image = wandb.Image(image)
                images.append(image)

        self.logger.experiment.log({
                'images': images,
            })

