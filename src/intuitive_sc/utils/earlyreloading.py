import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from intuitive_sc.utils.logging import get_logger

LOGGER = get_logger(__name__)


class DataLoaderReloader(EarlyStopping):
    def __init__(self, reload_every_n_epochs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reload_every_n_epochs = reload_every_n_epochs
        self.num_reloads = 0
        self.epochs_wo_reload = 0

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_validation_end(trainer, pl_module)
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return

        if self.reload_every_n_epochs > 0:
            # Check if the final subset is reached and stop the training
            # TODO check!
            num_fracs = trainer.max_epochs // self.reload_every_n_epochs
            final_subset = False
            if num_fracs == self.num_reloads + 1:
                final_subset = True

            # Check if the validation threshold is reached
            self._run_early_stopping_check_adapt(trainer, final_subset)

        else:
            return

    def _run_early_stopping_check_adapt(
        self, trainer: "pl.Trainer", final_subset: bool
    ) -> None:
        """Checks whether the early stopping condition is met and if so \
            tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(
            # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)

        if final_subset:
            trainer.should_stop = trainer.should_stop or should_stop
        if should_stop and final_subset:
            LOGGER.info("Stopping training on final subset.")
            self.stopped_epoch = trainer.current_epoch
        elif should_stop and not final_subset:
            LOGGER.info("Validation threshold reached. Early reloading the dataloader.")
            self.num_reloads += 1
            self.epochs_wo_reload = 0
            # reset best_score and count to prevent early stopping in the next epoch
            torch_inf = torch.tensor(np.Inf)
            self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
            self.wait_count = 0
            trainer.datamodule.train_dataloader()
        elif (
            not should_stop and self.epochs_wo_reload == self.reload_every_n_epochs - 1
        ):
            if final_subset:
                LOGGER.info("Stopping training on final subset.")
                self.stopped_epoch = trainer.current_epoch
            else:
                LOGGER.info(
                    f"Reloading dataloader after {self.reload_every_n_epochs} epochs."
                )
                self.epochs_wo_reload = 0
                # reset best_score and count to prevent early stopping in the next epoch
                torch_inf = torch.tensor(np.Inf)
                self.best_score = (
                    torch_inf if self.monitor_op == torch.lt else -torch_inf
                )
                self.wait_count = 0
                self.num_reloads += 1
                trainer.datamodule.train_dataloader()
        else:
            self.epochs_wo_reload += 1
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)
