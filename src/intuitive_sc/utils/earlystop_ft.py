import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from intuitive_sc.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


class EarlyStoppingFT(EarlyStopping):
    def __init__(self, monitor_ft: str, monitor_pt: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor_ft = monitor_ft
        self.monitor_pt = monitor_pt

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return

    def _run_early_stopping_check_custom(
        self,
        trainer: pl.Trainer,
    ) -> None:
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(
            # disable early_stopping with fast_dev_run
            logs
        ):
            return

        #  TODO have different criteria (e.g. patience)?
        current_ft = logs[self.monitor_ft].squeeze()
        current_pt = logs[self.monitor_pt].squeeze()
        current_combo = current_ft + current_pt
        should_stop_combo, reason_combo = self._evaluate_stopping_criteria(
            current_combo
        )
        should_stop_ft, reason_ft = self._evaluate_stopping_criteria(current_ft)
        should_stop_pt, reason_pt = self._evaluate_stopping_criteria(current_pt)

        should_stop = should_stop_ft and any([should_stop_combo, should_stop_pt])
        reason = reason_ft + " " + reason_combo + " " + reason_pt

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)

        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)
