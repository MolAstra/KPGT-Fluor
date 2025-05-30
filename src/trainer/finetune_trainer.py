from pathlib import Path
import pandas as pd
import torch
from torch import nn
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from loguru import logger


class Trainer:
    def __init__(
        self,
        args,
        optimizer,
        lr_scheduler,
        loss_fn,
        evaluator,
        result_tracker,
        summary_writer,
        device,
        label_mean=None,
        label_std=None,
        ddp=False,
        local_rank=0,
    ):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank

    def _forward_epoch(self, model, batched_data, perturb=None):
        (smiles, solvent, g, ecfp, md, sd, labels) = batched_data
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        sd = sd.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        predictions = model.forward_tune(g, ecfp, md, sd, perturb)
        return predictions, labels

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(
            tqdm(
                train_loader,
                desc=f"Training (epoch {epoch_idx})",
                total=len(train_loader),
            )
        ):
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean) / self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(
                    "Loss/train",
                    loss,
                    (epoch_idx - 1) * len(train_loader) + batch_idx + 1,
                )

    def fit(self, model, train_loader, val_loader, test_loader, save_dir: Path = None):
        best_val_result, best_test_result, best_train_result = (
            self.result_tracker.init(),
            self.result_tracker.init(),
            self.result_tracker.init(),
        )
        best_epoch = 0
        for epoch in range(1, self.args.n_epochs + 1):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.local_rank == 0:
                val_result = self.eval(model, val_loader)
                # test_result = self.eval(model, test_loader)
                # train_result = self.eval(model, train_loader)
                if self.result_tracker.update(
                    np.mean(best_val_result), np.mean(val_result)
                ):
                    best_val_result = val_result
                    # best_test_result = test_result
                    # best_train_result = train_result
                    best_epoch = epoch
                    if save_dir is not None:
                        self.save_model(model, save_dir.parent / f"{save_dir.name}.pth")
                logger.info(
                    f"Epoch {epoch} - Val: {np.mean(val_result)} - Best Val: {np.mean(best_val_result)}",
                )
                if epoch - best_epoch >= 20:
                    break

    def eval(self, model, dataloader):
        model.eval()
        predictions_all = []
        labels_all = []
        for batched_data in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            predictions, labels = self._forward_epoch(model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
        result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        return result

    def predict(self, model, dataloader, save_dir: Path = None):
        model.eval()

        predictions_all = []
        labels_all = []
        smiles_all = []
        solvent_all = []
        for batched_data in tqdm(dataloader, desc="Predicting", total=len(dataloader)):
            predictions, labels = self._forward_epoch(model, batched_data)
            smiles_all.extend(batched_data[0])
            solvent_all.extend(batched_data[1])
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())

        predictions = torch.cat(predictions_all).numpy()
        labels = torch.cat(labels_all).numpy()

        if self.evaluator.mean is not None and self.evaluator.std is not None:
            _predictions = predictions * self.evaluator.std + self.evaluator.mean

        result = self.evaluator.eval(labels, predictions)
        if save_dir is not None:
            save_dir.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                {
                    "smiles": smiles_all,
                    "solvent": solvent_all,
                    "predictions": _predictions.flatten(),
                    "label": labels.flatten(),
                }
            )
            df.to_csv(save_dir.parent / f"{save_dir.name}.csv", index=False)

        return result

    def save_model(self, model, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)


class FLAG_Trainer(Trainer):
    def __init__(
        self,
        args,
        d_node_hidden,
        optimizer,
        lr_scheduler,
        loss_fn,
        evaluator,
        result_tracker,
        summary_writer,
        device,
        label_mean=None,
        label_std=None,
        ddp=False,
        local_rank=0,
    ):
        super().__init__(
            args,
            optimizer,
            lr_scheduler,
            loss_fn,
            evaluator,
            result_tracker,
            summary_writer,
            device,
            label_mean=label_mean,
            label_std=label_std,
            ddp=ddp,
            local_rank=local_rank,
        )
        self.d_node_hidden = d_node_hidden

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(tqdm(list(train_loader))):
            self.optimizer.zero_grad()
            perturb_shape = (batched_data[2].number_of_nodes(), self.d_node_hidden)
            perturb = (
                torch.FloatTensor(*perturb_shape)
                .uniform_(-self.args.flag_step_size, self.args.flag_step_size)
                .to(self.device)
            )
            perturb.requires_grad_()
            predictions, labels = self._forward_epoch(model, batched_data, perturb)

            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean) / self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss /= self.args.flag_m
            for _ in range(self.args.flag_m - 1):
                loss.backward()
                perturb_data = perturb.detach() + self.args.flag_step_size * torch.sign(
                    perturb.grad.detach()
                )
                perturb.data = perturb_data.data
                perturb.grad[:] = 0
                predictions, labels = self._forward_epoch(model, batched_data, perturb)
                is_labeled = (~torch.isnan(labels)).to(torch.float32)
                labels = torch.nan_to_num(labels)
                if (self.label_mean is not None) and (self.label_std is not None):
                    labels = (labels - self.label_mean) / self.label_std
                loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
                loss /= self.args.flag_m
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(
                    "Loss/train",
                    loss,
                    (epoch_idx - 1) * len(train_loader) + batch_idx + 1,
                )


class SPRegularization(nn.Module):
    def __init__(self, source_model: nn.Module, target_model: nn.Module):
        super(SPRegularization, self).__init__()
        self.target_model = target_model
        self.source_weight = {}
        for name, param in source_model.named_parameters():
            if "predictor" not in name:
                self.source_weight[name] = param.detach()
            else:
                pass

    def forward(self):
        output = 0.0
        for name, param in self.target_model.named_parameters():
            if name in self.source_weight.keys():
                output += 0.5 * torch.norm(param - self.source_weight[name]) ** 2
        return output


class L2SP_Trainer(Trainer):
    def __init__(
        self,
        args,
        optimizer,
        lr_scheduler,
        loss_fn,
        evaluator,
        result_tracker,
        summary_writer,
        device,
        label_mean=None,
        label_std=None,
        ddp=False,
        local_rank=0,
    ):
        super().__init__(
            args,
            optimizer,
            lr_scheduler,
            loss_fn,
            evaluator,
            result_tracker,
            summary_writer,
            device,
            label_mean=label_mean,
            label_std=label_std,
            ddp=ddp,
            local_rank=local_rank,
        )
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank

    def train_epoch(self, spr, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(tqdm(list(train_loader))):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean) / self.label_std
            loss = (
                self.loss_fn(predictions, labels) * is_labeled
            ).mean() + self.args.l2sp_weight * spr()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(
                    "Loss/train",
                    loss,
                    (epoch_idx - 1) * len(train_loader) + batch_idx + 1,
                )

    def fit(self, model, train_loader, val_loader, test_loader, save_dir: Path = None):
        source_model = deepcopy(model).to(self.device)
        spr = SPRegularization(source_model, model)
        best_val_result, best_test_result, best_train_result = (
            self.result_tracker.init(),
            self.result_tracker.init(),
            self.result_tracker.init(),
        )
        best_epoch = 0
        for epoch in range(1, self.args.n_epochs + 1):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(spr, model, train_loader, epoch)
            if self.local_rank == 0:
                val_result = self.eval(model, val_loader)
                test_result = self.eval(model, test_loader)
                train_result = self.eval(model, train_loader)
                if self.result_tracker.update(
                    np.mean(best_val_result), np.mean(val_result)
                ):
                    best_val_result = val_result
                    best_test_result = test_result
                    best_train_result = train_result
                    best_epoch = epoch
                    if save_dir is not None:
                        self.save_model(
                            model, save_dir / f"{self.args.config}" / f"checkpoint.pth"
                        )
                if epoch - best_epoch >= 20:  # early stopping
                    break
        return (
            np.mean(best_train_result),
            np.mean(best_val_result),
            np.mean(best_test_result),
        )
