import pytorch_lightning as pl
import torch

from arguments import init_args
from data.dataset import DataModule
from iSeg import (
    iSeg,
)


def main():
    config = init_args()
    dm = DataModule(
        train_data_dir=config.train_data_dir,
        val_data_dir=config.val_data_dir,
        test_data_dir=config.test_data_dir,
        batch_size=config.batch_size,
        train_mask_size=config.train_mask_size,
        test_mask_size=config.test_mask_size,
        dataset_name=config.dataset_name,
        root_datasets_path=config.root_datasets_path,
    )
    model = iSeg(config=config)
    if config.model_file is not None:
        model.load_state_dict(torch.load(config.model_file))
    if isinstance(config.gpu_id, int):
        gpu_id = [config.gpu_id]
    else:
        gpu_id = config.gpu_id
    trainer = pl.Trainer(
        accelerator="gpu",
        default_root_dir=config.output_dir,
        max_epochs=config.epochs,
        devices=gpu_id,
        log_every_n_steps=1,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    if config.train:
        trainer.fit(model=model, datamodule=dm)
    else:
        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
