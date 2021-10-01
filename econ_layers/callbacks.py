import pytorch_lightning as pl
import pytorch_lightning.callbacks

class ResetOptimizers(pl.Callback):
    def __init__(self, verbose: bool, epoch_reset_field: str = "pretrain_epochs"):
        super().__init__()
        self.verbose = verbose
        self.epoch_reset_field= epoch_reset_field

    def on_train_epoch_end(self, trainer, pl_module):
        reset_epoch = getattr(pl_module.hparams, self.epoch_reset_field) - 1
        if trainer.current_epoch == reset_epoch:
            if self.verbose:
                print("\nPretraining complete, resetting optimizers and schedulers")
            trainer.accelerator.setup_optimizers(trainer)
