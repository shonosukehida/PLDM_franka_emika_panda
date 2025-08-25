from typing import NamedTuple, List, Any, Optional, Dict
from dataclasses import dataclass, field
import itertools
import os

import torch
from tqdm.auto import tqdm
import numpy as np
import math

from pldm.models.misc import Prober
from pldm.configs import ConfigBase
from pldm.logger import Logger
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from pldm.data.enums import ProbingDatasets, DatasetType
from pldm.data.utils import get_optional_fields
from pldm.optimizers.schedulers import Scheduler, LRSchedule
import glob
import tempfile


from pldm_envs.utils.normalizer import Normalizer
from pldm.models.jepa import JEPA
from pldm.models.hjepa import HJEPA

#probingtest 中でloss を確認
from pldm.objectives import ObjectivesConfig
from pldm.objectives.idm import IDMObjective

from PIL import Image, ImageSequence
from pathlib import Path

@dataclass
class ProbeTargetConfig(ConfigBase):
    arch: Optional[str] = None
    subclass: Optional[str] = None


@dataclass
class ProbingConfig(ConfigBase):
    exe_probing: bool = True
    probe_targets: str = "locations"
    l2_probe_targets: str = "locations"
    locations: ProbeTargetConfig = field(default_factory=ProbeTargetConfig)
    bluebox_locs: ProbeTargetConfig = field(default_factory=ProbeTargetConfig)
    propio_pos: ProbeTargetConfig = field(default_factory=ProbeTargetConfig)
    propio_vel: ProbeTargetConfig = field(default_factory=ProbeTargetConfig)
    full_finetune: bool = False
    lr: float = 1e-3
    epochs: int = 3
    epochs_enc: int = 3
    max_samples: Optional[int] = None
    max_samples_enc: Optional[int] = None
    schedule: LRSchedule = LRSchedule.Constant
    sample_timesteps: Optional[int] = None
    prober_arch: str = ""
    epochs_latent: int = 5
    l1_depth: int = 17
    l2_depth: int = 91
    probe_propio: bool = True
    probe_mpc: bool = False
    probe_wall: bool = True
    probe_border: bool = False
    probe_encoder: bool = True
    probe_preds: bool = True
    probe_expert: bool = False
    visualize_probing: bool = True
    load_prober: bool = False
    arch_subclass: str = "a"
    train_images_path: Optional[str] = None
    train_path: Optional[str] = None
    val_images_path: Optional[str] = None
    val_path: Optional[str] = None
    eval_contrastive: bool = False
    
    use_closed_loss_func: bool = False
    
    vis_dynamics_closed_featuremap: bool = True
    vis_dynamics_open_featuremap: bool = True
    vis_encoder_featruemap: bool = True


class ProbeResult(NamedTuple):
    model: torch.nn.Module
    average_eval_loss: float
    eval_losses_per_step: List[float]
    plots: List[Any]


default_config = ProbingConfig()


def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape
    # Pred and target are both B x T x N_DOTS x 2 or B x N_DOTS x 2.
    # we just avg the batch.
    # mse = (pred - target).pow(2).flatten(end_dim=-4).mean(dim=0)
    mse = (pred - target).pow(2).mean(dim=0)
    return mse


class ProbingEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        probing_datasets: Optional[ProbingDatasets],
        l2_probing_datasets: Optional[ProbingDatasets],
        load_checkpoint_path: str = "",
        output_path: str = "",
        config: ProbingConfig = default_config,
        quick_debug: bool = False,
        objectives_l1: Optional[ObjectivesConfig] = None,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model = model
        self.quick_debug = quick_debug

        self.ds = probing_datasets.ds
        self.val_ds = probing_datasets.val_ds
        self.extra_val_ds = probing_datasets.extra_datasets

        self.load_checkpoint_path = load_checkpoint_path
        self.output_path = output_path

        self.objectives_l1: ObjectivesConfig = (
            objectives_l1 if objectives_l1 is not None else ObjectivesConfig()
        )
        
        
        self.open_objectives_l1 = self.objectives_l1.build_open_objectives_list(
            name_prefix="l1", repr_dim=self.model.level1.spatial_repr_dim
        )

        self.closed_objectives_l1 = None
        if self.config.use_closed_loss_func:
            self.closed_objectives_l1 = self.objectives_l1.build_closed_objectives_list(
                name_prefix="l1", repr_dim=self.model.level1.spatial_repr_dim
            )

        # --- IDM weights をロードする ---
        if self.load_checkpoint_path:
            ckpt = torch.load(self.load_checkpoint_path, map_location="cpu")

            if "idm_open_state_dicts" in ckpt:
                for obj in self.open_objectives_l1:
                    if hasattr(obj, "action_predictor") and isinstance(obj, IDMObjective):
                        key = obj.name_prefix  # 通常 "l1"
                        if key in ckpt["idm_open_state_dicts"]:
                            obj.action_predictor.load_state_dict(ckpt["idm_open_state_dicts"][key])
                            print(f"[ProbingEvaluator] Loaded open IDM weights for {key}")

            if "idm_closed_state_dicts" in ckpt:
                if self.closed_objectives_l1 is not None:
                    for obj in self.closed_objectives_l1:
                        if hasattr(obj, "action_predictor") and isinstance(obj, IDMObjective):
                            key = obj.name_prefix
                            if key in ckpt["idm_closed_state_dicts"]:
                                obj.action_predictor.load_state_dict(ckpt["idm_closed_state_dicts"][key])
                                print(f"[ProbingEvaluator] Loaded closed IDM weights for {key}")


    def _context_manager(self):
        return torch.enable_grad() if self.config.full_finetune else torch.no_grad()

    def _infer_prober_path(self, probe_target, epoch, level, is_open_prober=False):
        if not is_open_prober:
            if self.load_checkpoint_path is not None and self.config.load_prober:
                root_path = "/".join(self.load_checkpoint_path.split("/")[:-1])
                prober_ckpt_paths = glob.glob(f"{root_path}/*{level}*{probe_target}*")
                assert len(prober_ckpt_paths) > 0
                # we get the most recent path (corresponding to latest epoch prober)
                prober_ckpt_path = max(prober_ckpt_paths, key=os.path.getctime)
                return prober_ckpt_path
            else:
                root_path = self.output_path

                prober_ckpt_path = (
                    f"{self.output_path}/{level}_prober-{probe_target}_epoch={epoch}.pt"
                )
                return prober_ckpt_path
        else:
            open_text = "open"
            if self.load_checkpoint_path is not None and self.config.load_prober:
                root_path = "/".join(self.load_checkpoint_path.split("/")[:-1])
                prober_ckpt_paths = glob.glob(f"{root_path}/*{level}*{open_text}*{probe_target}*")
                assert len(prober_ckpt_paths) > 0
                # we get the most recent path (corresponding to latest epoch prober)
                prober_ckpt_path = max(prober_ckpt_paths, key=os.path.getctime)
                return prober_ckpt_path
            else:
                root_path = self.output_path

                prober_ckpt_path = (
                    f"{self.output_path}/{level}_prober_open-{probe_target}_epoch={epoch}.pt"
                )
                return prober_ckpt_path
            

    def _infer_prober_input_dim_for_attr(
        self, probe_target, predictor, conv_input=False
    ):
        if predictor.pred_propio_dim == 0:
            repr_dim = predictor.repr_dim
        elif probe_target == "locations":
            repr_dim = predictor.pred_obs_dim
            print('PREDICTOR.PRED_OBS_DIM:', predictor.pred_obs_dim)
        elif probe_target == "propio_pos" or probe_target == "propio_vel":
            repr_dim = predictor.pred_propio_dim
        elif probe_target == "bluebox_locs": #とりあえず"locations"と同じ処理
            repr_dim = predictor.pred_obs_dim
        else:
            raise ValueError(f"Invalid probe target {probe_target}")

        if not conv_input and isinstance(repr_dim, tuple):
            return math.prod(repr_dim)

        return repr_dim

    def _get_pred_output_for_attr(self, pred_output, probe_target):
        if probe_target == "locations":
            if pred_output.obs_component is not None:
                return pred_output.obs_component
            else:
                return pred_output.predictions
        elif probe_target == "propio_pos" or probe_target == "propio_vel":
            if pred_output.propio_component is not None:
                return pred_output.propio_component
            else:
                return pred_output.predictions
        elif probe_target == "bluebox_locs": #とりあえず"locations"と同じ処理
            if pred_output.obs_component is not None:
                return pred_output.obs_component
            else:
                return pred_output.predictions
            
        else:
            raise ValueError(f"Invalid probe target {probe_target}")

    def _get_enc_output_for_attr(self, enc_output, probe_target):
        if probe_target == "locations":
            if enc_output.obs_component is not None:
                return enc_output.obs_component
            else:
                return enc_output.encodings
        elif probe_target == "propio_pos" or probe_target == "propio_vel":
            if enc_output.propio_component is not None:
                return enc_output.propio_component
            else:
                return enc_output.encodings
        elif probe_target == "bluebox_locs": #とりあえず"locations"と同じ処理
            
            if enc_output.obs_component is not None:
                return enc_output.obs_component
            else:
                return enc_output.encodings
        else:
            raise ValueError(f"Invalid probe target {probe_target}")

    def train_pred_prober(
        self,
        epoch: int,
        extra: Optional[Dict[str, Any]] = None,
        is_open_prober: bool = False,
    ):
        """
        Probes whether the predicted embeddings capture the future locations
        """

        level = "l1"

        plot_prefix = f"{level}_{epoch}"

        model = self.model.level1
        dataset = self.ds

        config = self.config
        epochs = config.epochs

        if self.quick_debug:
            epochs = 1
        test_batch = next(iter(dataset))

        probers = {}
        ckpt_paths = {}
        probe_targets = self.config.probe_targets.split(",") # ['locations', 'propio_vel']

        for probe_target in probe_targets:
            prober_output_shape = getattr(test_batch, probe_target)[0, 0].shape
            prober_output_shape = np.prod(prober_output_shape)

            probe_target_cfg = getattr(config, probe_target)

            prober_input_dim = self._infer_prober_input_dim_for_attr(
                probe_target=probe_target,
                predictor=model.predictor,
                conv_input=probe_target_cfg.arch == "conv",
            )

            prober = Prober(
                prober_input_dim,
                arch=probe_target_cfg.arch,
                output_shape=prober_output_shape,
                input_dim=prober_input_dim,  # to fix
                arch_subclass=probe_target_cfg.subclass,
            )

            probers[probe_target] = prober.to(self.device)

            # load prober logic
            ckpt_path = self._infer_prober_path(
                probe_target=probe_target, epoch=epoch, level=level, is_open_prober=is_open_prober
            )
            if config.load_prober:
                prober_ckpt = torch.load(ckpt_path)
                prober.load_state_dict(prober_ckpt["state_dict"])
                print(f"loaded {level} prober from {ckpt_path}")

            ckpt_paths[probe_target] = ckpt_path

        if config.load_prober:
            return probers

        all_parameters = []
        for probe_target, prober in probers.items():
            all_parameters += list(prober.parameters())

        if config.full_finetune:
            model.train()
            all_parameters += list(model.parameters())

        optimizer_pred_prober = torch.optim.Adam(all_parameters, config.lr)

        sample_step = 0
        step = 0

        batch_size = dataset.config.batch_size
        batch_steps = None

        if config.max_samples is not None:
            batch_steps = config.max_samples // dataset.config.batch_size
            n_epochs = max(1, batch_steps // len(dataset))
            if batch_steps < len(dataset):
                dataset = itertools.islice(dataset, batch_steps)
            epochs = n_epochs

        scheduler = Scheduler(
            schedule=self.config.schedule,
            base_lr=config.lr,
            data_loader=dataset,
            epochs=epochs,
            optimizer=optimizer_pred_prober,
            batch_steps=batch_steps,
            batch_size=batch_size,
        )

        for epoch in tqdm(range(epochs), desc=f"Probe {level} prediction OPEN epochs"):
            for batch in tqdm(dataset, desc="Probe prediction OPEN step"):
                # put time first
                states = batch.states.to(self.device).transpose(0, 1)
                actions = batch.actions.to(self.device).transpose(0, 1)
                optional_fields = get_optional_fields(batch, device=states.device)

                with self._context_manager(): ##
                    if not is_open_prober:
                        forward_result = model.forward_posterior(
                            states, actions, **optional_fields
                        )
                    else:
                        forward_result = model.forward_open(
                            states, actions, **optional_fields
                        )

                pred_output = forward_result.pred_output

                losses_list = []

                for probe_target, prober in probers.items():
                    pred_encs = self._get_pred_output_for_attr(
                        pred_output,
                        probe_target,
                    )

                    n_steps = pred_encs.shape[0]
                    bs = pred_encs.shape[1]

                    if not config.full_finetune:
                        pred_encs = pred_encs.detach()

                    target = getattr(batch, probe_target).to(self.device)
                    target = target[:, :: model.subsampling_ratio()] #model.subsampling_ratio() : 1

                    if (
                        config.sample_timesteps is not None
                        and config.sample_timesteps < n_steps
                    ): #x
                        
                        sample_shape = (config.sample_timesteps,) + pred_encs.shape[1:]
                        # we only randomly sample n timesteps to train prober.
                        # we most likely do this to avoid OOM
                        sampled_pred_encs = torch.empty(
                            sample_shape,
                            dtype=pred_encs.dtype,
                            device=pred_encs.device,
                        )

                        sampled_target_locs = torch.empty(
                            bs, config.sample_timesteps, 1, 2
                        )

                        for i in range(bs):
                            indices = torch.randperm(n_steps)[: config.sample_timesteps]
                            sampled_pred_encs[:, i, :] = pred_encs[indices, i, :]
                            sampled_target_locs[i, :] = target[i, indices]

                        pred_encs = sampled_pred_encs
                        target = sampled_target_locs.to(self.device)

                    pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)

                    losses = location_losses(pred_locs, target)
                    per_probe_loss = losses.mean()

                    if self.quick_debug or True:
                        if not is_open_prober:
                            log_dict = {
                                f"finetune_pred_{plot_prefix}_{probe_target}/loss": per_probe_loss.item(),
                            }
                        else:
                            log_dict = {
                                f"finetune_pred_{plot_prefix}_{probe_target}/open_loss": per_probe_loss.item(),
                            }
                        Logger.run().log(log_dict)

                    losses_list.append(per_probe_loss)

                optimizer_pred_prober.zero_grad()
                loss = sum(losses_list)
                loss.backward()
                optimizer_pred_prober.step()

                scheduler.adjust_learning_rate(step)

                step += 1
                sample_step += states.shape[0]

                if self.quick_debug:
                    break

        if config.full_finetune:  # we save the finetuned model as well
            model_ckpt_path = ckpt_path.replace("prober", "finetuned_model")
            torch.save({"model_state_dict": self.model.state_dict()}, model_ckpt_path)

        for probe_target, prober in probers.items():
            ckpt_path = ckpt_paths[probe_target]
            torch.save({"state_dict": prober.state_dict()}, ckpt_path)

        model.eval()

        return probers

    @torch.no_grad()
    def evaluate_all(
        self,
        probers,
        epoch,
        pixel_mapper=None,
        visualize=True,
        probers_open=None,
        enc_probers=None,
        vis_dynamics_closed_featuremap: bool = True,
        vis_dynamics_open_featuremap: bool = True,
        vis_encoder_featruemap: bool = True,
    ):
        """
        Evaluates on all the different validation datasets
        """

        val_datasets = {"pred_probe": self.val_ds}
        val_datasets.update(self.extra_val_ds)

        for prefix, val_ds in val_datasets.items():
            self.evaluate_pred_prober(
                probers=probers,
                epoch=epoch,
                val_ds=val_ds,
                pixel_mapper=pixel_mapper,
                visualize=visualize,
                probers_open=probers_open,
                enc_probers=enc_probers,
                vis_dynamics_closed_featuremap = vis_dynamics_closed_featuremap,
                vis_dynamics_open_featuremap = vis_dynamics_open_featuremap,
                vis_encoder_featruemap = vis_encoder_featruemap, 
            )

    @torch.no_grad()
    def evaluate_pred_prober(
        self,
        probers,
        epoch,
        val_ds: DatasetType,
        pixel_mapper=None,
        visualize=True,
        probers_open=None,
        enc_probers=None,
        vis_dynamics_closed_featuremap: bool = True,
        vis_dynamics_open_featuremap: bool = True,
        vis_encoder_featruemap: bool = True,
    ):
        level = "l1"

        plot_prefix = f"{level}_{epoch}"

        model = self.model.level1

        quick_debug = self.quick_debug

        eval_repr_losses = []
        target_repr_losses = []

        probing_losses = {}
        for probe_target, prober in probers.items():
            prober.eval()
            probing_losses[probe_target] = []

        for idx, batch in enumerate(tqdm(val_ds, desc="Eval probe pred")):
            # put time first
            states = batch.states.to(self.device).transpose(0, 1)

            actions = batch.actions.to(self.device).transpose(0, 1)

            optional_fields = get_optional_fields(batch, device=states.device)

            forward_result = model.forward_posterior(states, actions, **optional_fields)

            pred_output = forward_result.pred_output
            enc_output = forward_result.backbone_output

            for probe_target, prober in probers.items():
                pred_encs = self._get_pred_output_for_attr(
                    pred_output,
                    probe_target,
                )

                encs = self._get_enc_output_for_attr(enc_output, probe_target)

                target = getattr(batch, probe_target).to(self.device)
                target = target[:, :: model.subsampling_ratio()]

                pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)

                losses = location_losses(pred_locs, target)
                probing_losses[probe_target].append(losses.cpu())

            repr_loss = F.mse_loss(encs, pred_encs, reduction="none")
            reduce_dims = tuple(range(1, encs.ndim))
            repr_loss = repr_loss.mean(dim=reduce_dims)
            eval_repr_losses.append(repr_loss.cpu())

            target_encs = encs[-1]
            target_encs = target_encs.unsqueeze(0).expand(
                encs.shape[0], *[-1] * len(target_encs.shape)
            )

            # permutation = torch.randperm(64)
            # target_encs = target_encs[:, permutation, :, :, :]
            target_repr_loss = F.mse_loss(target_encs, pred_encs, reduction="none")
            target_repr_loss = target_repr_loss.mean(dim=reduce_dims)
            target_repr_losses.append(target_repr_loss.cpu())

            if quick_debug and idx > 2:
                break

        repr_loss = torch.stack(eval_repr_losses).mean(dim=0)
        target_repr_losses = torch.stack(target_repr_losses).mean(dim=0)

        # Plot repr loss over timesteps
        Logger.run().log_line_plot(
            data=[[i, x.item()] for i, x in enumerate(repr_loss)],
            plot_name=f"finetune_pred_val_{plot_prefix}_repr_loss",
        )

        # Plot target repr loss over timesteps
        Logger.run().log_line_plot(
            data=[[i, x.item()] for i, x in enumerate(target_repr_losses)],
            plot_name=f"finetune_pred_val_{plot_prefix}_target_repr_loss",
        )

        log_dict = {}

        for probe_target, eval_losses in probing_losses.items():
            losses_t = torch.stack(eval_losses, dim=0).mean(dim=0)
            losses_t = val_ds.normalizer.unnormalize_mse(losses_t, probe_target)
            losses_t = losses_t.mean(dim=-1)
            average_eval_loss = losses_t.mean().item()
            log_dict[f"finetune_pred_val_{plot_prefix}_{probe_target}/loss_avg"] = (
                average_eval_loss
            )
            log_dict[
                f"finetune_pred_val_{plot_prefix}_{probe_target}/loss_rmse_avg"
            ] = np.sqrt(average_eval_loss)

            # Plot probbing loss over timesteps
            Logger.run().log_line_plot(
                data=[[i, x.item()] for i, x in enumerate(losses_t)],
                plot_name=f"finetune_pred_val_{plot_prefix}_{probe_target}_loss",
            )

        Logger.run().log(log_dict)

        # right now, we only visualize location predictions
        if self.config.visualize_probing and visualize:
            
            # from torch.utils.data import DataLoader, Subset
            # original_dataset = val_ds.dataset 
            # t0_indices = [
            #     i for i, (ep_idx, t) in enumerate(original_dataset.flattened_indices) if t == 0
            # ]
            # t0_dataset = Subset(original_dataset, t0_indices)
            
            # t0_loader = DataLoader(
            #     t0_dataset, 
            #     batch_size=64, 
            #     shuffle=False, 
            #     num_workers=0,
            #     )
            # btc = next(iter(t0_loader))
            
            btc = next(iter(val_ds))
            self.plot_prober_predictions(
                btc,
                model,
                prober = probers["locations"],
                prober_open = probers_open["locations"],
                prober_bluebox_locs = probers["bluebox_locs"],
                prober_bluebox_locs_open = probers_open["bluebox_locs"],
                enc_prober = enc_probers["locations"] if isinstance(enc_probers, dict) else None,
                enc_prober_bluebox = enc_probers.get("bluebox_locs", None) if isinstance(enc_probers, dict) else None,
                normalizer=val_ds.normalizer,
                name_prefix=plot_prefix,
                idxs=None if not quick_debug else list(range(10)),
                pixel_mapper=pixel_mapper,
                vis_dynamics_closed_featuremap = vis_dynamics_closed_featuremap,
                vis_dynamics_open_featuremap = vis_dynamics_open_featuremap,
                vis_encoder_featruemap = vis_encoder_featruemap,
            )
            self.plot_prober_predictions_by_encprober(
                btc,
                model,
                prober = probers["locations"],
                prober_open = probers_open["locations"],
                prober_bluebox_locs = probers["bluebox_locs"],
                prober_bluebox_locs_open = probers_open["bluebox_locs"],
                enc_prober = enc_probers["locations"] if isinstance(enc_probers, dict) else None,
                enc_prober_bluebox = enc_probers.get("bluebox_locs", None) if isinstance(enc_probers, dict) else None,
                normalizer=val_ds.normalizer,
                name_prefix=plot_prefix,
                idxs=None if not quick_debug else list(range(10)),
                pixel_mapper=pixel_mapper,
                vis_dynamics_closed_featuremap = vis_dynamics_closed_featuremap,
                vis_dynamics_open_featuremap = vis_dynamics_open_featuremap,
                vis_encoder_featruemap = vis_encoder_featruemap,
            )

        return

    def train_encoder_prober(self, epoch: int, only_obs_component: bool = True):
        """
        Train a prober to probe whether the encoded embeddings captures the true location
        """
        plot_prefix = str(epoch)

        jepa = self.model.level1
        repr_dim = jepa.repr_dim
        dataset = self.ds
        quick_debug = self.quick_debug
        config = self.config

        test_batch = next(iter(dataset))

        probers = {}
        probe_targets = self.config.probe_targets.split(",")
        for probe_target in probe_targets:
            prober_output_shape = getattr(test_batch, probe_target)[0, 0].shape
            prober_output_shape = np.prod(prober_output_shape)

            probe_target_cfg = getattr(config, probe_target)

            # (1) テストバッチで forward を一度回す
            with torch.no_grad():
                states = test_batch.states.to(self.device).transpose(0, 1)
                actions = test_batch.actions.to(self.device).transpose(0, 1)
                optional_fields = get_optional_fields(test_batch, device=states.device)

                forward_result = jepa.forward_posterior(
                    states, actions, encode_only=True, **optional_fields
                )

                obs_component_shape = forward_result.backbone_output.obs_component[0].shape  # (B, C, H, W)
                obs_channels = obs_component_shape[1:]

            if only_obs_component:
                prober_input_dim = obs_channels
                embedding = obs_channels
            else:
                prober_input_dim = jepa.spatial_repr_dim
                embedding = repr_dim

            prober = Prober(
                embedding,
                arch=probe_target_cfg.arch,
                output_shape=prober_output_shape,
                input_dim=prober_input_dim,
                arch_subclass=probe_target_cfg.subclass,
            )
            #load_prober 用
            ckpt_path = self._infer_prober_path(
                probe_target=probe_target,
                epoch=epoch,
                level="enc",  # encoder prober用
                is_open_prober=False,
            )
            if config.load_prober:
                prober_ckpt = torch.load(ckpt_path)
                prober.load_state_dict(prober_ckpt["state_dict"])
                print(f"loaded encoder prober from {ckpt_path}")
            #;
            
            probers[probe_target] = prober.to(self.device)
        
        if config.load_prober:
            return probers
            

        all_parameters = []
        for probe_target, prober in probers.items():
            all_parameters += list(prober.parameters())

        if config.full_finetune:
            jepa.train()
            all_parameters += list(jepa.backbone.parameters())

        optimizer = torch.optim.Adam(all_parameters, config.lr)

        if quick_debug:
            config.epochs_enc = 1

        batch_size = dataset.config.batch_size
        batch_steps = None
        if config.max_samples_enc is not None:
            batch_steps = config.max_samples_enc // dataset.config.batch_size
            n_epochs = max(1, batch_steps // len(dataset))
            if batch_steps < len(dataset):
                dataset.dataset.config.crop_length = config.max_samples
            config.epochs_enc = n_epochs

        scheduler = Scheduler(
            schedule=self.config.schedule,
            base_lr=config.lr,
            data_loader=dataset,
            epochs=config.epochs_enc,
            optimizer=optimizer,
            batch_steps=batch_steps,
            batch_size=batch_size,
        )

        step = 0
        for epoch in tqdm(range(config.epochs_enc), desc="Eval enc"):
            for batch in dataset:
                states = batch.states.to(self.device).transpose(0, 1)
                actions = batch.actions.to(self.device).transpose(0, 1)

                optional_fields = get_optional_fields(batch, device=states.device)

                with self._context_manager():
                    forward_result = jepa.forward_posterior(
                        states, actions, encode_only=True, **optional_fields
                    )

                losses_list = []
                for t in range(states.shape[0]):
                    if only_obs_component:
                        e = forward_result.backbone_output.obs_component[t]  
                    else:
                        e = forward_result.backbone_output.encodings[t] 

                    for probe_target, prober in probers.items():
                        target = getattr(batch, probe_target)[:, t].to(self.device).float()

                        pred = prober(e)

                        loss = location_losses(pred, target)
                        losses_list.append(loss.mean())

                        if quick_debug or step % 100 == 0:
                            log_dict = {
                                f"finetune_enc_{plot_prefix}_{probe_target}/loss": loss.mean().item(),
                            }
                            Logger.run().log(log_dict)

                optimizer.zero_grad()
                total_loss = sum(losses_list)
                total_loss.backward()
                optimizer.step()

                scheduler.adjust_learning_rate(step)

                step += 1
                if quick_debug:
                    break

            if quick_debug:
                break
        
        for probe_target, prober in probers.items():
            ckpt_path = self._infer_prober_path(
                probe_target=probe_target,
                epoch=epoch,
                level="enc",  
                is_open_prober=False,
            )
            torch.save({"state_dict": prober.state_dict()}, ckpt_path)

        jepa.eval()
        return probers

    @torch.no_grad()
    def eval_probe_enc_position(
        self,
        probers,
        epoch: int,
    ):
        plot_prefix = str(epoch)

        jepa = self.model.level1
        val_dataset = self.val_ds
        quick_debug = self.quick_debug

        probing_losses = {}
        for probe_target, prober in probers.items():
            prober.eval()
            probing_losses[probe_target] = []

        for idx, batch in enumerate(val_dataset):
            states = batch.states.to(self.device).transpose(0, 1)
            actions = batch.actions.to(self.device).transpose(0, 1)

            optional_fields = get_optional_fields(batch, device=states.device)

            forward_result = jepa.forward_posterior(
                states, actions, encode_only=True, **optional_fields
            )

            e = forward_result.backbone_output.encodings[0]

            for probe_target, prober in probers.items():
                target = getattr(batch, probe_target)[:, 0].to(self.device).float()
                pred = prober(e)

                losses = location_losses(pred, target)

                probing_losses[probe_target].append(losses.cpu())

            if idx > 2 and quick_debug:
                break

        log_dict = {}
        for probe_target, eval_losses in probing_losses.items():
            avg_loss = torch.stack(eval_losses, dim=0).mean(dim=0)
            unnormalized_avg_loss = (
                val_dataset.normalizer.unnormalize_mse(avg_loss, probe_target)
                .mean()
                .cpu()
            )
            log_dict = {
                f"avg_eval_enc_{plot_prefix}_{probe_target}_loss": unnormalized_avg_loss,
                f"avg_eval_enc_{plot_prefix}_{probe_target}_loss_rmse": np.sqrt(
                    unnormalized_avg_loss
                ),
            }
        Logger.run().log(log_dict)

        return unnormalized_avg_loss


        
            
    
    def animate_feature_map_sequence(
        self, 
        feature_maps, 
        filename = "111",
        name_prefix=None,
        maps_idx=None,
        save_path=None,
        ):
        
        T, C, H, W = feature_maps.shape 
        n_rows = 4 
        n_cols = (C + n_rows - 1) // n_rows
        
        fig_width = n_cols * 2
        fig_height = n_rows * 2
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axs = axs.flatten()
        
        ims = []
        for t in range(T):
            frame = []
            for c in range(C):
                im = axs[c].imshow(feature_maps[t, c], cmap='viridis', animated = True)
                axs[c].axis('off')
                frame.append(im)
                
            text = fig.text(0.5, 0.02, f"timestep = {t} / {T - 1}", fontsize=14, color='black', ha='center', va='bottom', animated=True)
            frame.append(text)
            
            ims.append(frame)
        
        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=False)

        if save_path is None:
            save_dir = os.path.join(Logger.run().output_path, 'feature_maps')
            os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, f"{name_prefix}-featuremap_{maps_idx}.gif")
            save_path = os.path.join(save_dir, f"{filename}.gif")
            


        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)


        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        ani.save(str(tmp_path), writer="pillow")
        tmp_path.replace(out_path)

        plt.close(fig)
        return save_path, fig_width, fig_height



    def animate_obs_sequence(self, obs, fig_width, fig_height, save_path=None, name_prefix="obs", idx=0):
        """
        obs: Tensor (T, C, H, W) 
        """
        T, C, H, W = obs.shape

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ims = []

        for t in range(T):
            frame = obs[t].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
            frame = frame.clip(0, 255).astype(np.uint8)

            im = ax.imshow(frame, animated=True)
            text = ax.text(0.5, -0.05, f"timestep = {t} / {T - 1}", fontsize=14, color='black', ha='center', transform=ax.transAxes)
            ax.axis("off")
            ims.append([im, text])

        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=False)

        if save_path is None:
            save_dir = os.path.join(Logger.run().output_path, 'observation')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{name_prefix}-obs_{idx}.gif")

        ani.save(save_path, writer='pillow')
        plt.close(fig)

        return save_path

    def concat_gifs(self, gif_path1, gif_path2, save_dir = None, save_file = None, output_path = None, name_prefix: str = "", idx = 0):
        gif1 = Image.open(gif_path1)
        gif2 = Image.open(gif_path2)

        frames = []
        for frame1, frame2 in zip(ImageSequence.Iterator(gif1), ImageSequence.Iterator(gif2)):
            new_frame = Image.new("RGB", (frame1.width + frame2.width, frame1.height))
            new_frame.paste(frame1, (0, 0))
            new_frame.paste(frame2, (frame1.width, 0))
            frames.append(new_frame)
            
        if output_path is None:
            save_dir = os.path.join(Logger.run().output_path, save_dir)
            os.makedirs(save_dir, exist_ok=True)
            # output_path = os.path.join(save_dir, f"{name_prefix}-feature_map_and_obs_{idx}.gif")
            output_path = os.path.join(save_dir, f"{save_file}.gif")

        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=200)

        return output_path


    @torch.no_grad()
    def plot_prober_predictions(
        self,
        batch,
        jepa: JEPA,
        prober: torch.nn.Module,
        prober_open: torch.nn.Module,
        prober_bluebox_locs: torch.nn.Module = None,
        prober_bluebox_locs_open: torch.nn.Module = None,
        enc_prober: torch.nn.Module = None,
        enc_prober_bluebox: torch.nn.Module = None,
        normalizer: Normalizer = None,
        name_prefix: str = "",
        idxs: Optional[List[int]] = None,
        notebook: bool = False,
        pixel_mapper = None,
        vis_dynamics_closed_featuremap: bool = True,
        vis_dynamics_open_featuremap: bool = True,
        vis_encoder_featruemap: bool = True,
        
    ):

        # データバッチ
        states = batch.states.to(self.device).transpose(0, 1) #torch.Size([64, 50, 3, 64, 64])
        actions = batch.actions.to(self.device).transpose(0, 1)

        optional_fields = get_optional_fields(batch, device=states.device)

        # closed-forward 出力列
        pred_output = jepa.forward_posterior(
            states, actions, **optional_fields
        )
        pred_output = pred_output.pred_output

        # open-forward 出力列
        pred_output_open = jepa.forward_open(
            states, actions, **optional_fields
        )
        pred_output_open = pred_output_open.pred_output
        
        
        #encoderの出力列
        enc_output = jepa.forward_posterior(
                states, actions, encode_only=True, **optional_fields
            )
        enc_output = enc_output.backbone_output
        #encoder 出力列から画像ベースに分離
        encoder_encs = enc_output.obs_component
        

        #closed-forward 出力列から画像ベースに分離
        if pred_output.obs_component is not None: ##
            pred_encs = pred_output.obs_component
        else: #x
            pred_encs = pred_output.predictions

        #open-forward 出力列から画像ベースに分離
        if pred_output_open.obs_component is not None: ##
            pred_encs_open = pred_output_open.obs_component
        else: #x
            pred_encs_open = pred_output_open.predictions


        #endeffector
        #closed-forward出力列 --> prober
        pred_locs_clsfwd_clsprb = torch.stack([prober(x) for x in pred_encs], dim=1)
        pred_locs_clsfwd_clsprb = normalizer.unnormalize_location(pred_locs_clsfwd_clsprb).cpu()
        
        #open-forward出力列 --> prober
        pred_locs_opnfwd_clsprb = torch.stack([prober(x) for x in pred_encs_open], dim=1)
        pred_locs_opnfwd_clsprb = normalizer.unnormalize_location(pred_locs_opnfwd_clsprb).cpu()
        
        #closed-forward出力列 --> prober_open
        pred_locs_clsfwd_opnprb = torch.stack([prober_open(x) for x in pred_encs], dim=1)
        pred_locs_clsfwd_opnprb = normalizer.unnormalize_location(pred_locs_clsfwd_opnprb).cpu()
        
        #open-forward出力列 --> prober_open
        pred_locs_opnfwd_opnprb = torch.stack([prober_open(x) for x in pred_encs_open], dim=1)
        pred_locs_opnfwd_opnprb = normalizer.unnormalize_location(pred_locs_opnfwd_opnprb).cpu()
        
        #encoder出力列
        if enc_prober is not None:
            pred_enc_locs = torch.stack([enc_prober(x) for x in encoder_encs], dim=1)
            pred_enc_locs = normalizer.unnormalize_location(pred_enc_locs).cpu()


        #bluebox_locs
        #closed-forward出力列 --> prober
        pred_bluebox_locs_clsfwd_clsprb = torch.stack([prober_bluebox_locs(x) for x in pred_encs], dim=1)
        pred_bluebox_locs_clsfwd_clsprb = normalizer.unnormalize_bluebox_locs(pred_bluebox_locs_clsfwd_clsprb).cpu()
        
        #open-forward出力列 --> prober
        pred_bluebox_locs_opnfwd_clsprb = torch.stack([prober_bluebox_locs(x) for x in pred_encs_open], dim=1)
        pred_bluebox_locs_opnfwd_clsprb = normalizer.unnormalize_bluebox_locs(pred_bluebox_locs_opnfwd_clsprb).cpu()
        
        #closed-forward出力列 --> prober_open
        pred_bluebox_locs_clsfwd_opnprb = torch.stack([prober_bluebox_locs_open(x) for x in pred_encs], dim=1)
        pred_bluebox_locs_clsfwd_opnprb = normalizer.unnormalize_bluebox_locs(pred_bluebox_locs_clsfwd_opnprb).cpu()
        
        #open-forward出力列 --> prober_open
        pred_bluebox_locs_opnfwd_opnprb = torch.stack([prober_bluebox_locs_open(x) for x in pred_encs_open], dim=1)
        pred_bluebox_locs_opnfwd_opnprb = normalizer.unnormalize_bluebox_locs(pred_bluebox_locs_opnfwd_opnprb).cpu()
        
        #encoder出力列
        pred_enc_bluebox_locs = torch.stack([enc_prober_bluebox(x) for x in encoder_encs], dim=1)
        pred_enc_bluebox_locs = normalizer.unnormalize_bluebox_locs(pred_enc_bluebox_locs).cpu()

        # pred_locs is of shape (batch_size, time, 1, 2)
        if idxs is None: ##
            idxs = list(range(min(pred_locs_clsfwd_clsprb.shape[0], 64)))


        gt_locations = normalizer.unnormalize_location(batch.locations).cpu()
        
        gt_bluebox_locations = normalizer.unnormalize_bluebox_locs(batch.bluebox_locs).cpu()



        for i in tqdm(idxs, desc=f"Plotting {name_prefix}"):
            fig, axes = plt.subplots(2, 4, figsize=(18, 12), dpi=200)

            #画像表示
            ax_img = axes[0][0]
            img = normalizer.unnormalize_state(batch.states)
            init_img = img[i, 0].cpu().numpy().transpose(1, 2, 0)
            init_img = init_img.clip(0, 255).astype(np.uint8)
            
            ax_img.imshow(init_img)
            ax_img.set_title("init obs")
            ax_img.axis("off")


            #予測軌跡表示, ee, closed-forward
            ###########################################################################################
            ax_ee_forward = axes[0][1]
            #gt
            ax_ee_forward.plot(
                gt_locations[i, :, 0].cpu(),
                gt_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
                label="endeffector-ground-truth"
            )
            ax_ee_forward.text(
                gt_locations[i, 0, 0].cpu().item(),
                gt_locations[i, 0, 1].cpu().item(),
                "S",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            ax_ee_forward.text(
                gt_locations[i, -1, 0].cpu().item(),
                gt_locations[i, -1, 1].cpu().item(),
                "G",
                color="#3777FF",  
                fontsize=12,
                ha="center",
                va="center",
            )
            
            
            
            #ee, closed_forward, 
            ax_ee_forward.plot(
                pred_locs_clsfwd_clsprb[i, :, 0].cpu(),
                pred_locs_clsfwd_clsprb[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#D62828",
                alpha=0.8,
                label="endeffector_closed_pred"
            )
            ax_ee_forward.text(
                pred_locs_clsfwd_clsprb[i, 0, 0].cpu().item(),
                pred_locs_clsfwd_clsprb[i, 0, 1].cpu().item(),
                "S",
                color="#D62828",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_forward.text(
                pred_locs_clsfwd_clsprb[i, -1, 0].cpu().item(),
                pred_locs_clsfwd_clsprb[i, -1, 1].cpu().item(),
                "G",
                color="#D62828",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            ax_ee_forward.set_aspect("equal", adjustable="box")
            ax_ee_forward.set_xlim(0.315, 0.715)
            ax_ee_forward.set_ylim(-0.2, 0.2)
            ax_ee_forward.set_xlabel("X (meters)")
            ax_ee_forward.set_ylabel("Y (meters)")
            ax_ee_forward.legend()
            ax_ee_forward.set_title("dynamics model vs. groundtruth")
            ###########################################################################################


            #予測軌跡表示, ee, open-forward
            ###########################################################################################
            ax_ee_forward = axes[0][2]
            #gt
            ax_ee_forward.plot(
                gt_locations[i, :, 0].cpu(),
                gt_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
                label="endeffector-ground-truth"
            )
            ax_ee_forward.text(
                gt_locations[i, 0, 0].cpu().item(),
                gt_locations[i, 0, 1].cpu().item(),
                "S",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_forward.text(
                gt_locations[i, -1, 0].cpu().item(),
                gt_locations[i, -1, 1].cpu().item(),
                "G",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            #ee, open_forward, 
            ax_ee_forward.plot(
                pred_locs_opnfwd_opnprb[i, :, 0].cpu(),
                pred_locs_opnfwd_opnprb[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#ff8c00",
                alpha=0.8,
                label="endeffector_open_pred"
            )
            ax_ee_forward.text(
                pred_locs_opnfwd_opnprb[i, 0, 0].cpu().item(),
                pred_locs_opnfwd_opnprb[i, 0, 1].cpu().item(),
                "S",
                color="#ff8c00",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_forward.text(
                pred_locs_opnfwd_opnprb[i, -1, 0].cpu().item(),
                pred_locs_opnfwd_opnprb[i, -1, 1].cpu().item(),
                "G",
                color="#ff8c00",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            ax_ee_forward.set_aspect("equal", adjustable="box")
            ax_ee_forward.set_xlim(0.315, 0.715)
            ax_ee_forward.set_ylim(-0.2, 0.2)
            ax_ee_forward.set_xlabel("X (meters)")
            ax_ee_forward.set_ylabel("Y (meters)")
            ax_ee_forward.legend()
            ax_ee_forward.set_title("dynamics model vs. groundtruth")
            ###########################################################################################


            #予測軌跡表示, ee, encoder
            ###########################################################################################
            ax_ee_enc = axes[0][3]
            
            #gt
            ax_ee_enc.plot(
                gt_locations[i, :, 0].cpu(),
                gt_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
                label="endeffector-ground-truth"
            )
            ax_ee_enc.text(
                gt_locations[i, 0, 0].cpu().item(),
                gt_locations[i, 0, 1].cpu().item(),
                "S",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_enc.text(
                gt_locations[i, -1, 0].cpu().item(),
                gt_locations[i, -1, 1].cpu().item(),
                "G",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            
            #ee, encoder
            ax_ee_enc.plot(
                pred_enc_locs[i, :, 0].cpu(),
                pred_enc_locs[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#008000",
                alpha=0.8,
                label="endeffector_encoder"
            )
            ax_ee_enc.text(
                pred_enc_locs[i, 0, 0].cpu().item(),
                pred_enc_locs[i, 0, 1].cpu().item(),
                "S",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_enc.text(
                pred_enc_locs[i, -1, 0].cpu().item(),
                pred_enc_locs[i, -1, 1].cpu().item(),
                "G",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_enc.set_aspect("equal", adjustable="box")
            ax_ee_enc.set_xlim(0.315, 0.715)
            ax_ee_enc.set_ylim(-0.2, 0.2)
            ax_ee_enc.set_xlabel("X (meters)")
            ax_ee_enc.set_ylabel("Y (meters)")
            ax_ee_enc.legend()
            ax_ee_enc.set_title("encoder vs. groundtruth")
            ###########################################################################################



            #予測軌跡表示, bluebox, closed-forward
            ###########################################################################################
            ax_bluebox_forward = axes[1][1]
            
            #box, gt
            ax_bluebox_forward.plot(
                gt_bluebox_locations[i, :, 0].cpu(),
                gt_bluebox_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
                label="bluebox-ground-truth"
            )
            ax_bluebox_forward.text(
                gt_bluebox_locations[i, 0, 0].cpu().item(),
                gt_bluebox_locations[i, 0, 1].cpu().item(),
                "S",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_forward.text(
                gt_bluebox_locations[i, -1, 0].cpu().item(),
                gt_bluebox_locations[i, -1, 1].cpu().item(),
                "G",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            
            #box, closed_forward, closed_prober
            ax_bluebox_forward.plot(
                pred_bluebox_locs_clsfwd_clsprb[i, :, 0].cpu(),
                pred_bluebox_locs_clsfwd_clsprb[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#D62828",
                alpha=0.8,
                label="bluebox-closed-pred"
            )
            ax_bluebox_forward.text(
                pred_bluebox_locs_clsfwd_clsprb[i, 0, 0].cpu().item(),
                pred_bluebox_locs_clsfwd_clsprb[i, 0, 1].cpu().item(),
                "S",
                color="#D62828",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_forward.text(
                pred_bluebox_locs_clsfwd_clsprb[i, -1, 0].cpu().item(),
                pred_bluebox_locs_clsfwd_clsprb[i, -1, 1].cpu().item(),
                "G",
                color="#D62828",
                fontsize=12,
                ha="center",
                va="center",
            )

            ax_bluebox_forward.set_title("Bluebox Trajectory")
            ax_bluebox_forward.set_xlim(0.315, 0.715)
            ax_bluebox_forward.set_ylim(-0.2, 0.2)
            ax_bluebox_forward.set_aspect("equal")
            ax_bluebox_forward.legend()
            ###########################################################################################


            #予測軌跡表示, bluebox, open-forward
            ###########################################################################################
            ax_bluebox_forward = axes[1][2]
            
            #box, gt
            ax_bluebox_forward.plot(
                gt_bluebox_locations[i, :, 0].cpu(),
                gt_bluebox_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
                label="bluebox-ground-truth"
            )
            ax_bluebox_forward.text(
                gt_bluebox_locations[i, 0, 0].cpu().item(),
                gt_bluebox_locations[i, 0, 1].cpu().item(),
                "S",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_forward.text(
                gt_bluebox_locations[i, -1, 0].cpu().item(),
                gt_bluebox_locations[i, -1, 1].cpu().item(),
                "G",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            
            #box, closed_forward, closed_prober
            ax_bluebox_forward.plot(
                pred_bluebox_locs_opnfwd_opnprb[i, :, 0].cpu(),
                pred_bluebox_locs_opnfwd_opnprb[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#ff8c00",
                alpha=0.8,
                label="bluebox-open-pred"
            )
            ax_bluebox_forward.text(
                pred_bluebox_locs_opnfwd_opnprb[i, 0, 0].cpu().item(),
                pred_bluebox_locs_opnfwd_opnprb[i, 0, 1].cpu().item(),
                "S",
                color="#ff8c00",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_forward.text(
                pred_bluebox_locs_opnfwd_opnprb[i, -1, 0].cpu().item(),
                pred_bluebox_locs_opnfwd_opnprb[i, -1, 1].cpu().item(),
                "G",
                color="#ff8c00",
                fontsize=12,
                ha="center",
                va="center",
            )


            ax_bluebox_forward.set_title("Bluebox Trajectory")
            ax_bluebox_forward.set_xlim(0.315, 0.715)
            ax_bluebox_forward.set_ylim(-0.2, 0.2)
            ax_bluebox_forward.set_aspect("equal")
            ax_bluebox_forward.legend()
            ###########################################################################################


            #予測軌跡表示, bluebox, encoder
            ###########################################################################################
            ax_bluebox_enc = axes[1][3]
            
            #box, gt
            ax_bluebox_enc.plot(
                gt_bluebox_locations[i, :, 0].cpu(),
                gt_bluebox_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
                label="bluebox-ground-truth"
            )
            ax_bluebox_enc.text(
                gt_bluebox_locations[i, 0, 0].cpu().item(),
                gt_bluebox_locations[i, 0, 1].cpu().item(),
                "S",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_enc.text(
                gt_bluebox_locations[i, -1, 0].cpu().item(),
                gt_bluebox_locations[i, -1, 1].cpu().item(),
                "G",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            

            #box, encoder
            ax_bluebox_enc.plot(
                pred_enc_bluebox_locs[i, :, 0].cpu(),
                pred_enc_bluebox_locs[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#008000",
                alpha=0.8,
                label="bluebox-encoder-pred"
            )
            ax_bluebox_enc.text(
                pred_enc_bluebox_locs[i, 0, 0].cpu().item(),
                pred_enc_bluebox_locs[i, 0, 1].cpu().item(),
                "S",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_enc.text(
                pred_enc_bluebox_locs[i, -1, 0].cpu().item(),
                pred_enc_bluebox_locs[i, -1, 1].cpu().item(),
                "G",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )

            ax_bluebox_enc.set_title("Bluebox Trajectory")
            ax_bluebox_enc.set_xlim(0.315, 0.715)
            ax_bluebox_enc.set_ylim(-0.2, 0.2)
            ax_bluebox_enc.set_aspect("equal")
            ax_bluebox_enc.legend()
            ###########################################################################################

            obs_gif_path = None
            
            #ダイナミクスモデル出力(closed-forward)の特徴マップ
            if vis_dynamics_closed_featuremap:
                feature_maps = pred_encs[:, i].detach().cpu()
                filename = f"{name_prefix}-dynamics_closed-featuremap_{i}"
                ft_maps_gif_path, fig_width, fig_height = self.animate_feature_map_sequence(
                    feature_maps, 
                    filename=filename,
                    name_prefix=name_prefix,
                    maps_idx=i
                    )

                obs_gif_path = self.animate_obs_sequence(
                    img[i], 
                    fig_width=fig_width, 
                    fig_height=fig_height,
                    idx = i
                    )

                dynamics_closed_ft_maps_and_obs_gif_path = self.concat_gifs(
                    ft_maps_gif_path, 
                    obs_gif_path, 
                    save_dir="dynamics_closed_ftmap_obs", 
                    save_file=f"{name_prefix}-dynamics_closed_{i}.gif",
                    name_prefix=name_prefix, 
                    idx=i,
                    )
            
            #ダイナミクスモデル出力(open-forward)の特徴マップ
            if vis_dynamics_open_featuremap:
                feature_maps = pred_encs_open[:, i].detach().cpu()
                filename = f"{name_prefix}-dynamics_open-featuremap_{i}"
                ft_maps_gif_path, fig_width, fig_height = self.animate_feature_map_sequence(
                    feature_maps, 
                    filename=filename,
                    name_prefix=name_prefix,
                    maps_idx=i
                    )
                
                if obs_gif_path is None:
                    obs_gif_path = self.animate_obs_sequence(
                        img[i], 
                        fig_width=fig_width, 
                        fig_height=fig_height,
                        idx = i
                        )
                
                dynamics_open_ft_maps_and_obs_gif_path = self.concat_gifs(
                    ft_maps_gif_path, 
                    obs_gif_path, #実観測は作成ずみ
                    save_dir="dynamics_open_ftmap_obs", 
                    save_file=f"{name_prefix}-dynamics_open_{i}.gif",
                    name_prefix=name_prefix, 
                    idx=i,
                    )
                
            #エンコーダ出力の特徴マップ
            if vis_encoder_featruemap:
                feature_maps = encoder_encs[:, i].detach().cpu() 
                filename = f"{name_prefix}-encoder-featuremap_{i}"
                ft_maps_gif_path, fig_width, fig_height = self.animate_feature_map_sequence(
                    feature_maps, 
                    filename=filename,
                    name_prefix=name_prefix,
                    maps_idx=i
                    )
                
                if obs_gif_path is None:
                    obs_gif_path = self.animate_obs_sequence(
                        img[i], 
                        fig_width=fig_width, 
                        fig_height=fig_height,
                        idx = i
                        )
                
                encoder_ft_maps_and_obs_gif_path = self.concat_gifs(
                    ft_maps_gif_path, 
                    obs_gif_path, #実観測は作成ずみ
                    save_dir="encoder_ftmap_obs", 
                    save_file=f"{name_prefix}-encoder_{i}.gif",
                    name_prefix=name_prefix, 
                    idx=i,
                    )
            
            


            if not notebook:
                Logger.run().log_figure(fig, f"{name_prefix}-prober_predictions_{i}")
                # Logger.run().log_video(ft_maps_gif_path, f"{name_prefix}-featuremap_{i}")
                if vis_dynamics_closed_featuremap:
                    Logger.run().log_video(dynamics_closed_ft_maps_and_obs_gif_path, f"{name_prefix}-closed_dynamics_ftmaps_{i}")
                if vis_dynamics_open_featuremap:
                    Logger.run().log_video(dynamics_open_ft_maps_and_obs_gif_path, f"{name_prefix}-open_dynamics_ftmaps_{i}")
                if vis_encoder_featruemap:
                    Logger.run().log_video(encoder_ft_maps_and_obs_gif_path, f"{name_prefix}-encoder_ftmaps_{i}")

                plt.close(fig)
            else:
                plt.show()
        


    # encoder 出力で学習させたprober を共通利用
    @torch.no_grad()
    def plot_prober_predictions_by_encprober(
        self,
        batch,
        jepa: JEPA,
        prober: torch.nn.Module,
        prober_open: torch.nn.Module,
        prober_bluebox_locs: torch.nn.Module = None,
        prober_bluebox_locs_open: torch.nn.Module = None,
        enc_prober: torch.nn.Module = None,
        enc_prober_bluebox: torch.nn.Module = None,
        normalizer: Normalizer = None,
        name_prefix: str = "",
        idxs: Optional[List[int]] = None,
        notebook: bool = False,
        pixel_mapper = None,
        vis_dynamics_closed_featuremap: bool = True,
        vis_dynamics_open_featuremap: bool = True,
        vis_encoder_featruemap: bool = True,
        
    ):
        assert enc_prober is not None, "enc_prober is required"
        assert enc_prober_bluebox is not None, "enc_prober_bluebox is required"


        # データバッチ
        states = batch.states.to(self.device).transpose(0, 1) #torch.Size([64, 50, 3, 64, 64])
        actions = batch.actions.to(self.device).transpose(0, 1)

        optional_fields = get_optional_fields(batch, device=states.device)

        # closed-forward 出力列
        pred_output = jepa.forward_posterior(
            states, actions, **optional_fields
        )
        pred_output = pred_output.pred_output

        # open-forward 出力列
        pred_output_open = jepa.forward_open(
            states, actions, **optional_fields
        )
        pred_output_open = pred_output_open.pred_output
        
        
        #encoderの出力列
        enc_output = jepa.forward_posterior(
                states, actions, encode_only=True, **optional_fields
            )
        enc_output = enc_output.backbone_output
        #encoder 出力列から画像ベースに分離
        encoder_encs = enc_output.obs_component
        

        #closed-forward 出力列から画像ベースに分離
        if pred_output.obs_component is not None: ##
            pred_encs = pred_output.obs_component
        else: #x
            pred_encs = pred_output.predictions

        #open-forward 出力列から画像ベースに分離
        if pred_output_open.obs_component is not None: ##
            pred_encs_open = pred_output_open.obs_component
        else: #x
            pred_encs_open = pred_output_open.predictions


        #TODO endeffector
        # #closed-forward出力列 --> prober
        # pred_locs_clsfwd_clsprb = torch.stack([prober(x) for x in pred_encs], dim=1)
        # pred_locs_clsfwd_clsprb = normalizer.unnormalize_location(pred_locs_clsfwd_clsprb).cpu()
        
        # #open-forward出力列 --> prober
        # pred_locs_opnfwd_clsprb = torch.stack([prober(x) for x in pred_encs_open], dim=1)
        # pred_locs_opnfwd_clsprb = normalizer.unnormalize_location(pred_locs_opnfwd_clsprb).cpu()
        
        # #closed-forward出力列 --> prober_open
        # pred_locs_clsfwd_opnprb = torch.stack([prober_open(x) for x in pred_encs], dim=1)
        # pred_locs_clsfwd_opnprb = normalizer.unnormalize_location(pred_locs_clsfwd_opnprb).cpu()
        
        # #open-forward出力列 --> prober_open
        # pred_locs_opnfwd_opnprb = torch.stack([prober_open(x) for x in pred_encs_open], dim=1)
        # pred_locs_opnfwd_opnprb = normalizer.unnormalize_location(pred_locs_opnfwd_opnprb).cpu()
        
        
        #encoder出力列
        if enc_prober is not None:
            pred_enc_locs = torch.stack([enc_prober(x) for x in encoder_encs], dim=1)
            pred_enc_locs = normalizer.unnormalize_location(pred_enc_locs).cpu()
            
        #closed-forward出力列 --> enc_prober
        pred_locs_clsfwd_encprb = torch.stack([enc_prober(x) for x in pred_encs], dim=1)
        pred_locs_clsfwd_encprb = normalizer.unnormalize_location(pred_locs_clsfwd_encprb).cpu()
        
        #open-forward出力列 --> enc_prober 
        pred_locs_opnfwd_encprb = torch.stack([enc_prober(x) for x in pred_encs_open], dim=1)
        pred_locs_opnfwd_encprb = normalizer.unnormalize_location(pred_locs_opnfwd_encprb).cpu()
        


        #TODO bluebox_locs
        # #closed-forward出力列 --> prober
        # pred_bluebox_locs_clsfwd_clsprb = torch.stack([prober_bluebox_locs(x) for x in pred_encs], dim=1)
        # pred_bluebox_locs_clsfwd_clsprb = normalizer.unnormalize_bluebox_locs(pred_bluebox_locs_clsfwd_clsprb).cpu()
        
        # #open-forward出力列 --> prober
        # pred_bluebox_locs_opnfwd_clsprb = torch.stack([prober_bluebox_locs(x) for x in pred_encs_open], dim=1)
        # pred_bluebox_locs_opnfwd_clsprb = normalizer.unnormalize_bluebox_locs(pred_bluebox_locs_opnfwd_clsprb).cpu()
        
        # #closed-forward出力列 --> prober_open
        # pred_bluebox_locs_clsfwd_opnprb = torch.stack([prober_bluebox_locs_open(x) for x in pred_encs], dim=1)
        # pred_bluebox_locs_clsfwd_opnprb = normalizer.unnormalize_bluebox_locs(pred_bluebox_locs_clsfwd_opnprb).cpu()
        
        # #open-forward出力列 --> prober_open
        # pred_bluebox_locs_opnfwd_opnprb = torch.stack([prober_bluebox_locs_open(x) for x in pred_encs_open], dim=1)
        # pred_bluebox_locs_opnfwd_opnprb = normalizer.unnormalize_bluebox_locs(pred_bluebox_locs_opnfwd_opnprb).cpu()
        
        #encoder出力列
        pred_enc_bluebox_locs = torch.stack([enc_prober_bluebox(x) for x in encoder_encs], dim=1)
        pred_enc_bluebox_locs = normalizer.unnormalize_bluebox_locs(pred_enc_bluebox_locs).cpu()

        
        #closed-forward出力列 --> enc_prober
        pred_bluebox_locs_clsfwd_encprb = torch.stack([enc_prober_bluebox(x) for x in pred_encs], dim=1)
        pred_bluebox_locs_clsfwd_encprb = normalizer.unnormalize_bluebox_locs(pred_bluebox_locs_clsfwd_encprb).cpu()
        
        #open-forward出力列 --> enc_prober
        pred_bluebox_locs_opnfwd_encprb = torch.stack([enc_prober_bluebox(x) for x in pred_encs_open], dim=1)
        pred_bluebox_locs_opnfwd_encprb = normalizer.unnormalize_bluebox_locs(pred_bluebox_locs_opnfwd_encprb).cpu()
        


        # pred_locs is of shape (batch_size, time, 1, 2)
        if idxs is None: ##
            idxs = list(range(min(pred_locs_clsfwd_encprb.shape[0], 64)))


        gt_locations = normalizer.unnormalize_location(batch.locations).cpu()
        
        gt_bluebox_locations = normalizer.unnormalize_bluebox_locs(batch.bluebox_locs).cpu()



        for i in tqdm(idxs, desc=f"Plotting {name_prefix}"):
            fig, axes = plt.subplots(2, 4, figsize=(18, 12), dpi=200)

            #画像表示
            ax_img = axes[0][0]
            img = normalizer.unnormalize_state(batch.states)
            init_img = img[i, 0].cpu().numpy().transpose(1, 2, 0)
            init_img = init_img.clip(0, 255).astype(np.uint8)
            
            ax_img.imshow(init_img)
            ax_img.set_title("init obs")
            ax_img.axis("off")


            #予測軌跡表示, ee, closed-forward
            ###########################################################################################
            ax_ee_forward = axes[0][1]
            #gt
            ax_ee_forward.plot(
                gt_locations[i, :, 0].cpu(),
                gt_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
                label="endeffector-ground-truth"
            )
            ax_ee_forward.text(
                gt_locations[i, 0, 0].cpu().item(),
                gt_locations[i, 0, 1].cpu().item(),
                "S",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            ax_ee_forward.text(
                gt_locations[i, -1, 0].cpu().item(),
                gt_locations[i, -1, 1].cpu().item(),
                "G",
                color="#3777FF",  
                fontsize=12,
                ha="center",
                va="center",
            )

            #ee, encoder
            ax_ee_forward.plot(
                pred_enc_locs[i, :, 0].cpu(),
                pred_enc_locs[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#008000",
                alpha=0.8,
                label="endeffector-encoder"
            )
            ax_ee_forward.text(
                pred_enc_locs[i, 0, 0].cpu().item(),
                pred_enc_locs[i, 0, 1].cpu().item(),
                "S",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_forward.text(
                pred_enc_locs[i, -1, 0].cpu().item(),
                pred_enc_locs[i, -1, 1].cpu().item(),
                "G",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            
            #ee, closed_forward, 
            ax_ee_forward.plot(
                pred_locs_clsfwd_encprb[i, :, 0].cpu(),
                pred_locs_clsfwd_encprb[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#D62828",
                alpha=0.8,
                label="endeffector-closed_pred"
            )
            ax_ee_forward.text(
                pred_locs_clsfwd_encprb[i, 0, 0].cpu().item(),
                pred_locs_clsfwd_encprb[i, 0, 1].cpu().item(),
                "S",
                color="#D62828",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_forward.text(
                pred_locs_clsfwd_encprb[i, -1, 0].cpu().item(),
                pred_locs_clsfwd_encprb[i, -1, 1].cpu().item(),
                "G",
                color="#D62828",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            ax_ee_forward.set_aspect("equal", adjustable="box")
            ax_ee_forward.set_xlim(0.315, 0.715)
            ax_ee_forward.set_ylim(-0.2, 0.2)
            ax_ee_forward.set_xlabel("X (meters)")
            ax_ee_forward.set_ylabel("Y (meters)")
            ax_ee_forward.legend()
            ax_ee_forward.set_title("dynamics model vs. groundtruth")
            ###########################################################################################


            #予測軌跡表示, ee, open-forward
            ###########################################################################################
            ax_ee_forward = axes[0][2]
            #gt
            ax_ee_forward.plot(
                gt_locations[i, :, 0].cpu(),
                gt_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
                label="endeffector-ground-truth"
            )
            ax_ee_forward.text(
                gt_locations[i, 0, 0].cpu().item(),
                gt_locations[i, 0, 1].cpu().item(),
                "S",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_forward.text(
                gt_locations[i, -1, 0].cpu().item(),
                gt_locations[i, -1, 1].cpu().item(),
                "G",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )


            #ee, encoder
            ax_ee_forward.plot(
                pred_enc_locs[i, :, 0].cpu(),
                pred_enc_locs[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#008000",
                alpha=0.8,
                label="endeffector-encoder"
            )
            ax_ee_forward.text(
                pred_enc_locs[i, 0, 0].cpu().item(),
                pred_enc_locs[i, 0, 1].cpu().item(),
                "S",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_forward.text(
                pred_enc_locs[i, -1, 0].cpu().item(),
                pred_enc_locs[i, -1, 1].cpu().item(),
                "G",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            
            #ee, open_forward, 
            ax_ee_forward.plot(
                pred_locs_opnfwd_encprb[i, :, 0].cpu(),
                pred_locs_opnfwd_encprb[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#ff8c00",
                alpha=0.8,
                label="endeffector-open_pred"
            )
            ax_ee_forward.text(
                pred_locs_opnfwd_encprb[i, 0, 0].cpu().item(),
                pred_locs_opnfwd_encprb[i, 0, 1].cpu().item(),
                "S",
                color="#ff8c00",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_forward.text(
                pred_locs_opnfwd_encprb[i, -1, 0].cpu().item(),
                pred_locs_opnfwd_encprb[i, -1, 1].cpu().item(),
                "G",
                color="#ff8c00",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            ax_ee_forward.set_aspect("equal", adjustable="box")
            ax_ee_forward.set_xlim(0.315, 0.715)
            ax_ee_forward.set_ylim(-0.2, 0.2)
            ax_ee_forward.set_xlabel("X (meters)")
            ax_ee_forward.set_ylabel("Y (meters)")
            ax_ee_forward.legend()
            ax_ee_forward.set_title("dynamics model vs. groundtruth")
            ###########################################################################################


            #予測軌跡表示, ee, encoder
            ###########################################################################################
            ax_ee_enc = axes[0][3]
            
            #gt
            ax_ee_enc.plot(
                gt_locations[i, :, 0].cpu(),
                gt_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
                label="endeffector-ground-truth"
            )
            ax_ee_enc.text(
                gt_locations[i, 0, 0].cpu().item(),
                gt_locations[i, 0, 1].cpu().item(),
                "S",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_enc.text(
                gt_locations[i, -1, 0].cpu().item(),
                gt_locations[i, -1, 1].cpu().item(),
                "G",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            
            
            #ee, encoder
            ax_ee_enc.plot(
                pred_enc_locs[i, :, 0].cpu(),
                pred_enc_locs[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#008000",
                alpha=0.8,
                label="endeffector-encoder"
            )
            ax_ee_enc.text(
                pred_enc_locs[i, 0, 0].cpu().item(),
                pred_enc_locs[i, 0, 1].cpu().item(),
                "S",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_enc.text(
                pred_enc_locs[i, -1, 0].cpu().item(),
                pred_enc_locs[i, -1, 1].cpu().item(),
                "G",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_ee_enc.set_aspect("equal", adjustable="box")
            ax_ee_enc.set_xlim(0.315, 0.715)
            ax_ee_enc.set_ylim(-0.2, 0.2)
            ax_ee_enc.set_xlabel("X (meters)")
            ax_ee_enc.set_ylabel("Y (meters)")
            ax_ee_enc.legend()
            ax_ee_enc.set_title("encoder vs. groundtruth")
            ###########################################################################################



            #予測軌跡表示, bluebox, closed-forward
            ###########################################################################################
            ax_bluebox_forward = axes[1][1]
            
            #box, gt
            ax_bluebox_forward.plot(
                gt_bluebox_locations[i, :, 0].cpu(),
                gt_bluebox_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
                label="bluebox-ground-truth"
            )
            ax_bluebox_forward.text(
                gt_bluebox_locations[i, 0, 0].cpu().item(),
                gt_bluebox_locations[i, 0, 1].cpu().item(),
                "S",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_forward.text(
                gt_bluebox_locations[i, -1, 0].cpu().item(),
                gt_bluebox_locations[i, -1, 1].cpu().item(),
                "G",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )

            #box, encoder
            ax_bluebox_forward.plot(
                pred_enc_bluebox_locs[i, :, 0].cpu(),
                pred_enc_bluebox_locs[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#008000",
                alpha=0.8,
                label="bluebox-encoder-pred"
            )
            ax_bluebox_forward.text(
                pred_enc_bluebox_locs[i, 0, 0].cpu().item(),
                pred_enc_bluebox_locs[i, 0, 1].cpu().item(),
                "S",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_forward.text(
                pred_enc_bluebox_locs[i, -1, 0].cpu().item(),
                pred_enc_bluebox_locs[i, -1, 1].cpu().item(),
                "G",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )    
            
            #box, closed_forward, closed_prober
            ax_bluebox_forward.plot(
                pred_bluebox_locs_clsfwd_encprb[i, :, 0].cpu(),
                pred_bluebox_locs_clsfwd_encprb[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#D62828",
                alpha=0.8,
                label="bluebox-closed-pred"
            )
            ax_bluebox_forward.text(
                pred_bluebox_locs_clsfwd_encprb[i, 0, 0].cpu().item(),
                pred_bluebox_locs_clsfwd_encprb[i, 0, 1].cpu().item(),
                "S",
                color="#D62828",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_forward.text(
                pred_bluebox_locs_clsfwd_encprb[i, -1, 0].cpu().item(),
                pred_bluebox_locs_clsfwd_encprb[i, -1, 1].cpu().item(),
                "G",
                color="#D62828",
                fontsize=12,
                ha="center",
                va="center",
            )

            ax_bluebox_forward.set_title("Bluebox Trajectory")
            ax_bluebox_forward.set_xlim(0.315, 0.715)
            ax_bluebox_forward.set_ylim(-0.2, 0.2)
            ax_bluebox_forward.set_aspect("equal")
            ax_bluebox_forward.legend()
            ###########################################################################################


            #予測軌跡表示, bluebox, open-forward
            ###########################################################################################
            ax_bluebox_forward = axes[1][2]
            
            #box, gt
            ax_bluebox_forward.plot(
                gt_bluebox_locations[i, :, 0].cpu(),
                gt_bluebox_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
                label="bluebox-ground-truth"
            )
            ax_bluebox_forward.text(
                gt_bluebox_locations[i, 0, 0].cpu().item(),
                gt_bluebox_locations[i, 0, 1].cpu().item(),
                "S",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_forward.text(
                gt_bluebox_locations[i, -1, 0].cpu().item(),
                gt_bluebox_locations[i, -1, 1].cpu().item(),
                "G",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )

            #box, encoder
            ax_bluebox_forward.plot(
                pred_enc_bluebox_locs[i, :, 0].cpu(),
                pred_enc_bluebox_locs[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#008000",
                alpha=0.8,
                label="bluebox-encoder-pred"
            )
            ax_bluebox_forward.text(
                pred_enc_bluebox_locs[i, 0, 0].cpu().item(),
                pred_enc_bluebox_locs[i, 0, 1].cpu().item(),
                "S",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_forward.text(
                pred_enc_bluebox_locs[i, -1, 0].cpu().item(),
                pred_enc_bluebox_locs[i, -1, 1].cpu().item(),
                "G",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            ) 
            
            #box, closed_forward, closed_prober
            ax_bluebox_forward.plot(
                pred_bluebox_locs_opnfwd_encprb[i, :, 0].cpu(),
                pred_bluebox_locs_opnfwd_encprb[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#ff8c00",
                alpha=0.8,
                label="bluebox-open-pred"
            )
            ax_bluebox_forward.text(
                pred_bluebox_locs_opnfwd_encprb[i, 0, 0].cpu().item(),
                pred_bluebox_locs_opnfwd_encprb[i, 0, 1].cpu().item(),
                "S",
                color="#ff8c00",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_forward.text(
                pred_bluebox_locs_opnfwd_encprb[i, -1, 0].cpu().item(),
                pred_bluebox_locs_opnfwd_encprb[i, -1, 1].cpu().item(),
                "G",
                color="#ff8c00",
                fontsize=12,
                ha="center",
                va="center",
            )


            ax_bluebox_forward.set_title("Bluebox Trajectory")
            ax_bluebox_forward.set_xlim(0.315, 0.715)
            ax_bluebox_forward.set_ylim(-0.2, 0.2)
            ax_bluebox_forward.set_aspect("equal")
            ax_bluebox_forward.legend()
            ###########################################################################################


            #予測軌跡表示, bluebox, encoder
            ###########################################################################################
            ax_bluebox_enc = axes[1][3]
            
            #box, gt
            ax_bluebox_enc.plot(
                gt_bluebox_locations[i, :, 0].cpu(),
                gt_bluebox_locations[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#3777FF",
                alpha=0.8,
                label="bluebox-ground-truth"
            )
            ax_bluebox_enc.text(
                gt_bluebox_locations[i, 0, 0].cpu().item(),
                gt_bluebox_locations[i, 0, 1].cpu().item(),
                "S",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_enc.text(
                gt_bluebox_locations[i, -1, 0].cpu().item(),
                gt_bluebox_locations[i, -1, 1].cpu().item(),
                "G",
                color="#3777FF",
                fontsize=12,
                ha="center",
                va="center",
            )
            

            #box, encoder
            ax_bluebox_enc.plot(
                pred_enc_bluebox_locs[i, :, 0].cpu(),
                pred_enc_bluebox_locs[i, :, 1].cpu(),
                marker="o",
                markersize=2.5,
                linewidth=1,
                c="#008000",
                alpha=0.8,
                label="bluebox-encoder-pred"
            )
            ax_bluebox_enc.text(
                pred_enc_bluebox_locs[i, 0, 0].cpu().item(),
                pred_enc_bluebox_locs[i, 0, 1].cpu().item(),
                "S",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax_bluebox_enc.text(
                pred_enc_bluebox_locs[i, -1, 0].cpu().item(),
                pred_enc_bluebox_locs[i, -1, 1].cpu().item(),
                "G",
                color="#008000",
                fontsize=12,
                ha="center",
                va="center",
            )

            ax_bluebox_enc.set_title("Bluebox Trajectory")
            ax_bluebox_enc.set_xlim(0.315, 0.715)
            ax_bluebox_enc.set_ylim(-0.2, 0.2)
            ax_bluebox_enc.set_aspect("equal")
            ax_bluebox_enc.legend()
            ###########################################################################################
            
            


            if not notebook:
                Logger.run().log_figure(fig, f"{name_prefix}-prober_predictions_by_encprober_{i}", dir_name = 'prober_prediction_by_encprober')
                # Logger.run().log_video(ft_maps_gif_path, f"{name_prefix}-featuremap_{i}")

                plt.close(fig)
            else:
                plt.show()