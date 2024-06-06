import math
from time import time

import torch
import torch.cuda.amp as amp
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from recbole.trainer import Trainer as RecboleTrainer
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.utils import (
    early_stopping,
    dict2str,
    set_color,
    get_gpu_usage,
    EvaluatorType
)


class Trainer(RecboleTrainer):

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        # Add user sensitive attribute information to the evaluation collector
        if eval_data:
            self.eval_collector.eval_data_collect(eval_data)
        return super(Trainer, self).evaluate(
            eval_data,
            load_best_model=load_best_model,
            model_file=model_file,
            show_progress=show_progress
        )

    @torch.no_grad()
    def evaluate_from_scores(
            self, eval_data, scores, show_progress=False
    ):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            scores (torch.Tensor): scores predicted by model.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return
        else:
            self.eval_collector.eval_data_collect(eval_data)

        if isinstance(eval_data, FullSortEvalDataLoader):
            if self.item_tensor is None:
                self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        else:
            raise NotImplementedError("Evaluation from scores not supported with negative sampling")
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, history_index, positive_u, positive_i = batched_data
            batch_scores = scores.clone()[interaction[eval_data.dataset.uid_field]]
            batch_scores[:, 0] = -torch.inf
            batch_scores[history_index] = -torch.inf
            self.eval_collector.eval_batch_collect(
                batch_scores, interaction, positive_u, positive_i
            )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result
