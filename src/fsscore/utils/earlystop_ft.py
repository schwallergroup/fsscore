from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from torch import Tensor

from fsscore.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


class EarlyStoppingFT(EarlyStopping):
    def __init__(
        self,
        monitor_ft: str,
        monitor_pt: str,
        mode_ft: str = "min",
        mode_pt: str = "max",
        *args,
        **kwargs,
    ):
        super().__init__(monitor=monitor_ft, mode=mode_ft, *args, **kwargs)
        self.monitor_ft = monitor_ft
        self.monitor_pt = monitor_pt
        self.mode_ft = mode_ft
        self.mode_pt = mode_pt
        self.wait_count_ft = 0
        self.wait_count_pt = 0
        torch_inf = torch.tensor(np.Inf)
        self.best_score_ft = torch_inf if self.monitor_op_ft == torch.lt else -torch_inf
        self.best_score_pt = torch_inf if self.monitor_op_pt == torch.lt else -torch_inf

        self.min_delta *= 1 if self.monitor_op_pt == torch.gt else -1

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor_ft=self.monitor_ft, monitor_pt=self.monitor_pt, mode=self.mode_ft
        )

    @property
    def monitor_op_ft(self) -> Callable:
        return self.mode_dict[self.mode_ft]

    @property
    def monitor_op_pt(self) -> Callable:
        return self.mode_dict[self.mode_pt]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "wait_count_ft": self.wait_count_ft,
            "wait_count_pt": self.wait_count_pt,
            "stopped_epoch": self.stopped_epoch,
            "best_score_ft": self.best_score_ft,
            "best_score_pt": self.best_score_pt,
            "patience": self.patience,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.wait_count_ft = state_dict["wait_count_ft"]
        self.wait_count_pt = state_dict["wait_count_pt"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.best_score_ft = state_dict["best_score_ft"]
        self.best_score_pt = state_dict["best_score_pt"]
        self.patience = state_dict["patience"]

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def _run_early_stopping_check(
        self,
        trainer: pl.Trainer,
    ) -> None:
        logs = trainer.callback_metrics

        if (
            trainer.fast_dev_run
            or not self._validate_condition_metric(
                logs,
                self.monitor_ft,
            )
            or not self._validate_condition_metric(
                logs,
                self.monitor_pt,
            )
        ):
            return

        #  TODO have different criteria (e.g. patience)?
        current_ft = logs[self.monitor_ft].squeeze()
        current_pt = logs[self.monitor_pt].squeeze()

        (
            should_stop_ft,
            reason_ft,
            self.wait_count_ft,
            self.best_score_ft,
        ) = self._evaluate_stopping_criteria_custom(
            current_ft,
            self.monitor_ft,
            self.best_score_ft,
            self.wait_count_ft,
            min_delta=0.0,
            monitor_op=self.monitor_op_ft,
            mode=self.mode_ft,
        )
        (
            should_stop_pt,
            reason_pt,
            self.wait_count_pt,
            self.best_score_pt,
        ) = self._evaluate_stopping_criteria_custom(
            current_pt,
            self.monitor_pt,
            self.best_score_pt,
            self.wait_count_pt,
            min_delta=self.min_delta,
            monitor_op=self.monitor_op_pt,
            mode=self.mode_pt,
        )

        should_stop = should_stop_ft or should_stop_pt

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop

        if should_stop:
            reason = reason_ft if reason_ft else ""
            reason += reason_pt if reason_pt else ""
            self.stopped_epoch = trainer.current_epoch
        else:
            reason = None
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    def _evaluate_stopping_criteria_custom(
        self,
        current: Tensor,
        monitor: str,
        best_score: Tensor,
        wait_count: int,
        min_delta: float,
        monitor_op: Callable,
        mode: str,
    ) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {monitor} = {current} is not finite."
                f" Previous best value was {best_score:.3f}."
                " Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and monitor_op(
            current, self.stopping_threshold
        ):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {monitor} = {current} {self.order_dict[mode]}"
                f"{self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and monitor_op(
            -current, -self.divergence_threshold
        ):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {monitor} = {current} {self.order_dict[mode]} \
                    {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif monitor_op(current - min_delta, best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current, monitor, min_delta, best_score)
            best_score = current
            wait_count = 0
        else:
            wait_count += 1
            if wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {monitor} did not improve"
                    f" in the last {wait_count} records."
                    f" Best score: {best_score:.3f}."
                    " Signaling Trainer to stop."
                )

        return should_stop, reason, wait_count, best_score

    def _improvement_message(
        self, current: Tensor, monitor: str, min_delta: float, best_score: Tensor
    ) -> str:
        """Formats a log message that informs the user about \
            an improvement in the monitored score."""
        if torch.isfinite(best_score):
            msg = (
                f"Metric {monitor} improved by {abs(best_score - current):.3f} >="
                f" min_delta = {abs(min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {monitor} improved. New best score: {current:.3f}"
        return msg

    def _validate_condition_metric(
        self,
        logs: Dict[str, Tensor],
        monitor: str,
    ) -> bool:
        monitor_val = logs.get(monitor)

        error_msg = (
            f"Early stopping conditioned on metric `{monitor}` which is not available."
            " Pass in or modify your `EarlyStopping` callback"
            f'to use any of the following: `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, category=RuntimeWarning)

            return False

        return True
