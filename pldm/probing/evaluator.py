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
import gc


from pldm_envs.utils.normalizer import Normalizer
from pldm.models.jepa import JEPA
from pldm.models.hjepa import HJEPA

#probingtest 中でloss を確認
from pldm.objectives import ObjectivesConfig
from pldm.objectives.idm import IDMObjective

from PIL import Image, ImageSequence
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from matplotlib import cm
from typing import Optional, List

import hashlib
from torch.utils.data import Subset, DataLoader


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
    
    use_opn_loss_func: bool = False
    
    vis_dynamics_closed_featuremap: bool = True
    vis_dynamics_open_featuremap: bool = True
    vis_encoder_featuremap: bool = True


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
        
        
        self.clsd_objectives_l1 = self.objectives_l1.build_clsd_objectives_list(
            name_prefix="l1", repr_dim=self.model.level1.spatial_repr_dim
        )

        self.opn_objectives_l1 = None
        if self.config.use_opn_loss_func:
            self.opn_objectives_l1 = self.objectives_l1.build_opn_objectives_list(
                name_prefix="l1", repr_dim=self.model.level1.spatial_repr_dim
            )
            
        # --- IDM weights をロードする ---
        if self.load_checkpoint_path:
            ckpt = torch.load(self.load_checkpoint_path, map_location="cpu")

            # --- backward compatibility: map old keys to new keys ---
            if "idm_open_state_dicts" in ckpt and "idm_clsd_state_dicts" not in ckpt:
                ckpt["idm_clsd_state_dicts"] = ckpt["idm_open_state_dicts"]
                print("[ProbingEvaluator][BC] mapped idm_open_state_dicts → idm_clsd_state_dicts")

            if "idm_closed_state_dicts" in ckpt and "idm_opn_state_dicts" not in ckpt:
                ckpt["idm_opn_state_dicts"] = ckpt["idm_closed_state_dicts"]
                print("[ProbingEvaluator][BC] mapped idm_closed_state_dicts → idm_opn_state_dicts")

            # --- load closed-loop IDM ---
            if "idm_clsd_state_dicts" in ckpt:
                for obj in self.clsd_objectives_l1:
                    if hasattr(obj, "action_predictor") and isinstance(obj, IDMObjective):
                        key = obj.name_prefix
                        if key in ckpt["idm_clsd_state_dicts"]:
                            obj.action_predictor.load_state_dict(ckpt["idm_clsd_state_dicts"][key])
                            print(f"[ProbingEvaluator] Loaded clsd IDM weights for {key}")
                        else:
                            print(f"[ProbingEvaluator] ⚠ No clsd IDM weights found for {key}")

            # --- load open-loop IDM (if enabled) ---
            if "idm_opn_state_dicts" in ckpt and self.opn_objectives_l1 is not None:
                for obj in self.opn_objectives_l1:
                    if hasattr(obj, "action_predictor") and isinstance(obj, IDMObjective):
                        key = obj.name_prefix
                        if key in ckpt["idm_opn_state_dicts"]:
                            obj.action_predictor.load_state_dict(ckpt["idm_opn_state_dicts"][key])
                            print(f"[ProbingEvaluator] Loaded opn IDM weights for {key}")
                        else:
                            print(f"[ProbingEvaluator] ⚠ No opn IDM weights found for {key}")


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
            print("[prober network]: ", probe_target_cfg.arch)
            print(prober.prober)

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
        vis_encoder_featuremap: bool = True,
    ):
        """
        Evaluates on all the different validation datasets
        """
        # self.model.eval() 

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
                vis_encoder_featuremap = vis_encoder_featuremap, 
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
        vis_encoder_featuremap: bool = True,
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

        # for idx, batch in enumerate(tqdm(val_ds, desc="Eval probe pred")):
        #     # put time first
        #     states = batch.states.to(self.device).transpose(0, 1)

        #     actions = batch.actions.to(self.device).transpose(0, 1)

        #     optional_fields = get_optional_fields(batch, device=states.device)

        #     forward_result = model.forward_posterior(states, actions, **optional_fields)

        #     pred_output = forward_result.pred_output
        #     enc_output = forward_result.backbone_output

        #     for probe_target, prober in probers.items():
        #         pred_encs = self._get_pred_output_for_attr(
        #             pred_output,
        #             probe_target,
        #         )

        #         encs = self._get_enc_output_for_attr(enc_output, probe_target)

        #         target = getattr(batch, probe_target).to(self.device)
        #         target = target[:, :: model.subsampling_ratio()]

        #         pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)

        #         losses = location_losses(pred_locs, target)
        #         probing_losses[probe_target].append(losses.cpu())

        #     repr_loss = F.mse_loss(encs, pred_encs, reduction="none")
        #     reduce_dims = tuple(range(1, encs.ndim))
        #     repr_loss = repr_loss.mean(dim=reduce_dims)
        #     eval_repr_losses.append(repr_loss.cpu())

        #     target_encs = encs[-1]
        #     target_encs = target_encs.unsqueeze(0).expand(
        #         encs.shape[0], *[-1] * len(target_encs.shape)
        #     )

        #     # permutation = torch.randperm(64)
        #     # target_encs = target_encs[:, permutation, :, :, :]
        #     target_repr_loss = F.mse_loss(target_encs, pred_encs, reduction="none")
        #     target_repr_loss = target_repr_loss.mean(dim=reduce_dims)
        #     target_repr_losses.append(target_repr_loss.cpu())

        #     if quick_debug and idx > 2:
        #         break

        # repr_loss = torch.stack(eval_repr_losses).mean(dim=0)
        # target_repr_losses = torch.stack(target_repr_losses).mean(dim=0)

        # # Plot repr loss over timesteps
        # Logger.run().log_line_plot(
        #     data=[[i, x.item()] for i, x in enumerate(repr_loss)],
        #     plot_name=f"finetune_pred_val_{plot_prefix}_repr_loss",
        # )

        # # Plot target repr loss over timesteps
        # Logger.run().log_line_plot(
        #     data=[[i, x.item()] for i, x in enumerate(target_repr_losses)],
        #     plot_name=f"finetune_pred_val_{plot_prefix}_target_repr_loss",
        # )

        # log_dict = {}

        # for probe_target, eval_losses in probing_losses.items():
        #     losses_t = torch.stack(eval_losses, dim=0).mean(dim=0)
        #     losses_t = val_ds.normalizer.unnormalize_mse(losses_t, probe_target)
        #     losses_t = losses_t.mean(dim=-1)
        #     average_eval_loss = losses_t.mean().item()
        #     log_dict[f"finetune_pred_val_{plot_prefix}_{probe_target}/loss_avg"] = (
        #         average_eval_loss
        #     )
        #     log_dict[
        #         f"finetune_pred_val_{plot_prefix}_{probe_target}/loss_rmse_avg"
        #     ] = np.sqrt(average_eval_loss)

        #     # Plot probbing loss over timesteps
        #     Logger.run().log_line_plot(
        #         data=[[i, x.item()] for i, x in enumerate(losses_t)],
        #         plot_name=f"finetune_pred_val_{plot_prefix}_{probe_target}_loss",
        #     )

        # Logger.run().log(log_dict)

        # right now, we only visualize location predictions
        if self.config.visualize_probing and visualize:

            # =======================================================================
            # === 可視化直前の“probe_valチェック”を安全に ===
            from torch.utils.data import RandomSampler
            import types

            def _unwrap_loader(x):
                # NormalizedDataLoader → .dataloader が本物の DataLoader
                return getattr(x, "dataloader", x)

            inner = _unwrap_loader(val_ds)

            # PyTorchの版によって sampler の場所が違うことがあるので両対応
            sampler = getattr(inner, "sampler", None)
            if sampler is None and hasattr(inner, "batch_sampler"):
                sampler = getattr(inner.batch_sampler, "sampler", None)

            sampler_name = type(sampler).__name__ if sampler is not None else "<none>"
            print(f"[VIS] sampler: {sampler_name}")  # 期待: SequentialSampler

            assert not isinstance(sampler, RandomSampler), \
                "可視化で shuffle=True のローダ（probe_ds）を掴んでます！probe_val_ds を渡してください。"


            # ================= 可視化：各バッチの先頭GTをプロット =================


            os.makedirs("vis_debug/val_batches", exist_ok=True)

            # どれくらい見るか（全部なら None に）
            MAX_BATCHES = None

            def _sha(x: np.ndarray, n=16):
                return hashlib.sha256(x.tobytes()).hexdigest()[:n]

            # NOTE:
            #   すでに btc = next(iter(val_ds)) していますが、
            #   for ループは「新しいイテレータ」を作るので**先頭から**始まります👌
            for b_idx, batch in enumerate(itertools.islice(val_ds, 0, MAX_BATCHES)):
                # このバッチの indices をメモ（先頭サンプルの index も）
                idx_np = batch.indices.detach().cpu().numpy().astype(np.int64)
                print(f"[VAL][{b_idx:04d}] head_idx={idx_np[0]}  idx_sha={_sha(idx_np)}")

                # 逆正規化した GT （B, T, 2）想定
                gt_locations = val_ds.normalizer.unnormalize_location(batch.locations).cpu().numpy()

                # 先頭サンプルの軌跡（T,2）
                traj = gt_locations[0]  # 先頭だけ
                x, y = traj[:, 0], traj[:, 1]

                # プロット
                fig = plt.figure(figsize=(5, 5), dpi=160)
                ax = plt.gca()
                ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.0)
                # 始点/終点にマーク
                ax.text(x[0],  y[0],  "S", color="C0", fontsize=10, ha="center", va="center")
                ax.text(x[-1], y[-1], "G", color="C1", fontsize=10, ha="center", va="center")
                ax.set_title(f"val batch {b_idx:04d}  (first sample idx={idx_np[0]})")
                ax.set_xlim(0.315, 0.715)
                ax.set_ylim(-0.2, 0.2)
                ax.set_xlabel("X (meters)")
                ax.set_ylabel("Y (meters)")
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)

                out_path = f"vis_debug/val_batches/val_batch_{b_idx:04d}.png"
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close(fig)
                print(f"[VAL][{b_idx:04d}] saved -> {out_path}")

            print("[VAL] done. wrote images under vis_debug/val_batches/")
            # ===================================================================


            # === 可視化入力btc の作成 === 
            from torch.utils.data import Subset, DataLoader

            def _unwrap_loader(x):
                return getattr(x, "dataloader", x)  

            inner_loader = _unwrap_loader(val_ds)
            root_ds = inner_loader.dataset          
            B = inner_loader.batch_size
            N = len(root_ds)


            head_indices = np.arange(0, (N // B) * B, B, dtype=np.int64)
            os.makedirs("vis_debug", exist_ok=True)
            np.save("vis_debug/vis_indices_heads.npy", head_indices)
            print("[VIS] head_indices sha:", hashlib.sha256(head_indices.tobytes()).hexdigest()[:16])


            vis_subset = Subset(root_ds, head_indices.tolist())
            vis_loader_raw = DataLoader(
                vis_subset,
                batch_size=B,
                shuffle=False,     
                num_workers=0,
                drop_last=False,    
            )
            vis_loader_raw.config = inner_loader.config
            # 既存と同じ正規化を適用（NormalizedDataLoaderでラップ）
            from pldm.data.utils import NormalizedDataLoader
            vis_loader = NormalizedDataLoader(vis_loader_raw, val_ds.normalizer)

            # print("[DBG] len(root_ds) =", len(root_ds))
            # print("[DBG] batch_size =", B)
            # print("[DBG] head_indices =", head_indices, "len =", len(head_indices))
            # print("[DBG] vis_subset len =", len(vis_subset))
            # print("[DBG] len(vis_loader_raw) =", len(vis_loader_raw))
            # print("[DBG] len(vis_loader) =", len(vis_loader))

            vis_batch_idx = 0  
            itr = iter(vis_loader)
            for _ in range(vis_batch_idx + 1):
                btc = next(itr)


            np.save("vis_debug/vis_indices_used.npy", btc.indices.detach().cpu().numpy().astype(int))
            print("[VIS] used_idx sha:", hashlib.sha256(btc.indices.detach().cpu().numpy().astype(np.int64).tobytes()).hexdigest()[:16])
            

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
                vis_encoder_featuremap = vis_encoder_featuremap,
            )
            
            # Encoder の潜在分布が等方ガウスに近いか確認
            self.plot_encoder_latent_gaussianity(
                btc,
                model,
                name_prefix=f"{plot_prefix}_val",
            )
            
            self.log_encoder_latent_variance(
                    btc,
                    model,
                    name_prefix = "",
                    max_points = 5000,
                )
            self.plot_encoder_latent_tsne(
                    btc,
                    model,
                    name_prefix = "",
                    max_points = 5000,
            )

            # self.plot_prober_predictions_by_encprober(
            #     btc,
            #     model,
            #     prober = probers["locations"],
            #     prober_open = probers_open["locations"],
            #     prober_bluebox_locs = probers["bluebox_locs"],
            #     prober_bluebox_locs_open = probers_open["bluebox_locs"],
            #     enc_prober = enc_probers["locations"] if isinstance(enc_probers, dict) else None,
            #     enc_prober_bluebox = enc_probers.get("bluebox_locs", None) if isinstance(enc_probers, dict) else None,
            #     normalizer=val_ds.normalizer,
            #     name_prefix=plot_prefix,
            #     idxs=None if not quick_debug else list(range(10)),
            #     pixel_mapper=pixel_mapper,
            #     vis_dynamics_closed_featuremap = vis_dynamics_closed_featuremap,
            #     vis_dynamics_open_featuremap = vis_dynamics_open_featuremap,
            #     vis_encoder_featuremap = vis_encoder_featuremap,
            # )
            
            self.plot_prober_predictions_by_encprober_FOR_POSTER(
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
                vis_dynamics_closed_featuremap = False,
                vis_dynamics_open_featuremap = False,
                vis_encoder_featuremap = False,     
            )
            
            # self.plot_cca(
            #     btc,
            #     model,
            #     prober = None,
            #     prober_open = None,
            #     prober_bluebox_locs = None,
            #     prober_bluebox_locs_open = None,
            #     enc_prober = None,
            #     enc_prober_bluebox = None,
            #     normalizer = None,
            #     name_prefix = "",
            #     idxs = None,
            #     notebook = False,
            #     pixel_mapper = None,
            #     vis_dynamics_closed_featuremap = False,
            #     vis_dynamics_open_featuremap = False,
            #     vis_encoder_featuremap = False,
            # )
            
            self.plot_pca(
                btc,
                model,     
            )
            
            self.plot_pca_open_closed(
                btc,
                model,
                name_prefix=f"{plot_prefix}_val",
                idxs=None if not quick_debug else list(range(10)),
            )

            
            metrics_latent = self.log_latent_forward_mse(
                batch=btc,
                jepa=model,
                prober=probers["locations"],
                prober_open=probers_open["locations"],
                prober_bluebox_locs=probers["bluebox_locs"],
                prober_bluebox_locs_open=probers_open["bluebox_locs"],
                enc_prober=enc_probers["locations"] if isinstance(enc_probers, dict) else None,
                enc_prober_bluebox=enc_probers.get("bluebox_locs", None) if isinstance(enc_probers, dict) else None,
                normalizer=val_ds.normalizer,
                name_prefix=plot_prefix,
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
        
        import PIL.GifImagePlugin
        
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


        # tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        ani.save(str(out_path), writer="pillow")

        
        try:
            for ax in axs:
                ax.cla()
                if hasattr(ax, "images"):
                    try:
                        ax.images.clear()
                    except Exception:
                        pass
        finally:
            plt.close(fig)            
            del ani
            del ims
            del axs
            del fig
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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


        try:
            ax.cla()
            if hasattr(ax, "images"):
                try:
                    ax.images.clear()
                except Exception:
                    pass
        finally:
            plt.close(fig)
            del ani
            del ims
            del ax
            del fig
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        vis_encoder_featuremap: bool = True,
        
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
            if vis_encoder_featuremap:
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
                if vis_encoder_featuremap:
                    Logger.run().log_video(encoder_ft_maps_and_obs_gif_path, f"{name_prefix}-encoder_ftmaps_{i}")

                plt.close(fig)
            else:
                plt.show()


        plt.close('all')

        try:
            del pred_locs_clsfwd_encprb, pred_locs_opnfwd_encprb
        except Exception:
            pass
        try:
            del pred_bluebox_locs_clsfwd_encprb, pred_bluebox_locs_opnfwd_encprb
        except Exception:
            pass
        try:
            del pred_enc_locs, pred_enc_bluebox_locs
        except Exception:
            pass

        try:
            del pred_encs, pred_encs_open, encoder_encs
        except Exception:
            pass

        try:
            del gt_locations, gt_bluebox_locations
        except Exception:
            pass
        try:
            del img, init_img
        except Exception:
            pass

        try:
            del pred_output, pred_output_open, enc_output
        except Exception:
            pass
        try:
            del states, actions, optional_fields
        except Exception:
            pass

        gc.collect()
        torch.cuda.empty_cache()


    @torch.no_grad()
    def plot_encoder_latent_gaussianity(
        self,
        batch,
        jepa: JEPA,
        name_prefix: str = "",
        max_points: int = 5000,
    ):
        """
        Encoder の潜在表現がどれくらい等方ガウスに近いかを
        可視化する
        """
        from sklearn.decomposition import PCA  

        # ===== 1. Encoder 出力を取得 =====
        states = batch.states.to(self.device).transpose(0, 1)
        actions = batch.actions.to(self.device).transpose(0, 1)
        optional_fields = get_optional_fields(batch, device=states.device)
        
        print("[DBG plot_encoder_latent_gaussianity] states:", states.shape)
        print("[DBG plot_encoder_latent_gaussianity] actions:", actions.shape)

        enc_output = jepa.forward_posterior(
            states, actions, encode_only=True, **optional_fields
        )
        enc_output = enc_output.backbone_output

        encoder_encs = enc_output.obs_component
        print("[DBG plot_encoder_latent_gaussianity] encoder_encs:", encoder_encs.shape)

        # encoder_encs: (T, B, C, H, W) or (T, B, D, ...)
        zs = []
        for x in encoder_encs:
            x_flat = x.reshape(x.shape[0], -1)  # (B, K)
            zs.append(x_flat)

        z_all = torch.cat(zs, dim=0)  # (N, K)
        N, K = z_all.shape
        if N > max_points:
            idx = torch.randperm(N)[:max_points]
            z_all = z_all[idx]
            N = max_points

        # ===== 2. 平均 0 にセンタリング =====
        z_all = z_all.detach().cpu()
        mean = z_all.mean(dim=0, keepdim=True)
        z_centered = z_all - mean
        z_np = z_centered.numpy()  # (N, K)
        
        print("z_centered.mean: ", z_centered.mean().item())
        print("z_centered.std: ", z_centered.std().item())
        print("z_centered.min/max: ", z_centered.min().item(), z_centered.max().item())
        print("z_centered.has_nan: ", torch.isnan(z_centered).any())
        print("z_centered.has_inf: ", torch.isinf(z_centered).any())



        # ===== 3. PCA で 4 次元に落とす =====
        pca = PCA(n_components=min(4, K))
        z_pca = pca.fit_transform(z_np)
        # ===== 3.5 主成分ごとの寄与率を確認 =====
        
        explained_var = pca.explained_variance_
        explained_ratio = pca.explained_variance_ratio_

        print("[PCA] explained_variance:", explained_var)
        print("[PCA] explained_variance_ratio:", explained_ratio)

        # もし 4 つすべてを見たい場合（n_components が 4 のとき）
        for i, (ev, r) in enumerate(zip(explained_var, explained_ratio), start=1):
            print(f"[PCA] PC{i}: variance={ev:.4f}, ratio={r:.4%}")


        # ===== 4. 共分散の固有値も見て等方性をざっくり計算 =====
        cov = np.cov(z_np.T)
        eigvals = np.linalg.eigvalsh(cov)
        iso_ratio = float(eigvals.max() / (eigvals.min() + 1e-12))
        print("eigvals min/max:", eigvals.min(), eigvals.max())

        # ===== 5. プロット =====
        run = Logger.run()


        from pathlib import Path
        if run.output_path is not None:
            run_dir = Path(run.output_path)
        else:
            run_dir = Path(".")

        out_dir = run_dir / "latent_gaussianity"
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=200)

        # 左: PC1 vs PC2
        axes[0].scatter(z_pca[:, 0], z_pca[:, 1], s=8, alpha=0.6)
        axes[0].set_xlabel("PC 1")
        axes[0].set_ylabel("PC 2")
        axes[0].set_title(f"{name_prefix} encoder latent (PC1 vs PC2)")

        # 右: PC3 vs PC4
        if z_pca.shape[1] >= 4:
            axes[1].scatter(z_pca[:, 2], z_pca[:, 3], s=8, alpha=0.6)
            axes[1].set_xlabel("PC 3")
            axes[1].set_ylabel("PC 4")
            axes[1].set_title("PC3 vs PC4")
        else:
            axes[1].axis("off")

        plt.tight_layout()
        out_path = out_dir / f"{name_prefix}_encoder_latent.png"
        plt.savefig(out_path)
        plt.close(fig)

        abs_path = out_path.resolve()
        print(f"[LATENT] saved encoder latent plot -> {abs_path}")

        print(f"[LATENT] iso_ratio (max_eig / min_eig) = {iso_ratio:.3f}")

        # Logger にも記録
        try:
            run.log(
                {
                    f"{name_prefix}/latent_iso_ratio": iso_ratio,
                    f"{name_prefix}/latent_eig_max": float(eigvals.max()),
                    f"{name_prefix}/latent_eig_min": float(eigvals.min()),
                }
            )
        except Exception as e:
            print("[LATENT] Logger logging skipped:", e)
            
            
        # ===== ログをファイルに保存 =====
        log_path = out_dir / f"{name_prefix}_stats.txt"
        with open(log_path, "w") as f:
            f.write("=== z_centered stats ===\n")
            f.write(f"mean: {z_centered.mean().item()}\n")
            f.write(f"std: {z_centered.std().item()}\n")
            f.write(f"min: {z_centered.min().item()}\n")
            f.write(f"max: {z_centered.max().item()}\n")
            f.write(f"has_nan: {torch.isnan(z_centered).any()}\n")
            f.write(f"has_inf: {torch.isinf(z_centered).any()}\n\n")

            f.write("=== PCA explained variance ===\n")
            for i, (ev, r) in enumerate(zip(explained_var, explained_ratio), start=1):
                f.write(f"PC{i}: variance={ev:.4f}, ratio={r:.4%}\n")
            f.write("\n")

            f.write("=== covariance eigenvalues ===\n")
            f.write(f"eigvals: {eigvals.tolist()}\n")
            f.write(f"iso_ratio (max/min): {iso_ratio}\n")
            

        try:
            del z_all, z_centered, z_np
        except:
            pass
        try:
            del z_pca, explained_var, explained_ratio
        except:
            pass
        try:
            del cov, eigvals, iso_ratio
        except:
            pass
        try:
            del enc_output, encoder_encs
        except:
            pass
        try:
            del states, actions, optional_fields
        except:
            pass

        gc.collect()
        torch.cuda.empty_cache()

    # import gc
    # import numpy as np
    # import torch
    # from pathlib import Path
    # from pldm.data.utils import get_optional_fields
    # from pldm.models.jepa import JEPA
    # from pldm.logger import Logger
    # from matplotlib import pyplot as plt


    @torch.no_grad()
    def log_encoder_latent_variance(
        self,
        batch,
        jepa: JEPA,
        name_prefix: str = "",
        max_points: int = 5000,
    ):
        """
        学習後の Encoder 潜在の「分散・標準偏差」がどうなっているかを確認する。

        - 入力 shape / 流れは plot_encoder_latent_gaussianity() と同じ
        - 時刻 T × バッチ B を全部まとめて N = T*B サンプルとして扱い、
        各潜在次元の mean / std を集計する。
        """

        # ===== 1. Encoder 出力を取得 =====
        states = batch.states.to(self.device).transpose(0, 1)   # [T, B, ...]
        actions = batch.actions.to(self.device).transpose(0, 1) # [T, B, ...]
        optional_fields = get_optional_fields(batch, device=states.device)

        print("[DBG log_encoder_latent_variance] states:", states.shape)
        print("[DBG log_encoder_latent_variance] actions:", actions.shape)

        enc_output = jepa.forward_posterior(
            states, actions, encode_only=True, **optional_fields
        )
        enc_output = enc_output.backbone_output

        # ここでは obs_component を対象にする（必要なら propio_component などに変えて OK）
        encoder_encs = enc_output.obs_component
        # encoder_encs: (T, B, C, H, W) or (T, B, D, ...)
        print("[DBG log_encoder_latent_variance] encoder_encs:", encoder_encs.shape)

        # ===== 2. [T,B,...] → (N, K) に flatten =====
        zs = []
        for x in encoder_encs:             # x: [B, C, H, W] or [B, D, ...] （時間 t ごと）
            x_flat = x.reshape(x.shape[0], -1)  # (B, K)
            zs.append(x_flat)

        z_all = torch.cat(zs, dim=0)  # (N, K), N = T * B
        N, K = z_all.shape
        if N > max_points:
            idx = torch.randperm(N)[:max_points]
            z_all = z_all[idx]
            N = max_points

        # ===== 3. 全サンプル方向 (N) での mean / std を計算 =====
        # ここでは「各次元 k の mean / std」を見る
        z_all = z_all.detach().cpu()              # (N, K)
        mean_per_dim = z_all.mean(dim=0)          # (K,)
        std_per_dim  = z_all.std(dim=0, unbiased=False)  # (K,)

        # 全次元まとめたスカラーの統計量も出しておく
        global_mean_of_means = float(mean_per_dim.mean().item())
        global_std_of_means  = float(mean_per_dim.std().item())
        global_mean_std      = float(std_per_dim.mean().item())
        global_min_std       = float(std_per_dim.min().item())
        global_max_std       = float(std_per_dim.max().item())

        print("=== Encoder latent variance stats ===")
        print(f"[mean_per_dim]   mean={global_mean_of_means:.6f}, std={global_std_of_means:.6f}")
        print(f"[std_per_dim]    mean={global_mean_std:.6f}, "
            f"min={global_min_std:.6f}, max={global_max_std:.6f}")
        print(f"[N, K] = ({N}, {K})")

        # ===== 4. ログ & ファイル保存 =====
        run = Logger.run()

        if run.output_path is not None:
            run_dir = Path(run.output_path)
        else:
            run_dir = Path(".")

        out_dir = run_dir / "latent_variance"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ヒストグラム（std の分布）をプロットして保存
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=200)
        ax.hist(std_per_dim.numpy(), bins=40, alpha=0.8)
        ax.set_xlabel("std per dim")
        ax.set_ylabel("count")
        ax.set_title(f"{name_prefix} encoder latent std distribution")
        fig.tight_layout()
        out_path_fig = out_dir / f"{name_prefix}_encoder_latent_std_hist.png"
        plt.savefig(out_path_fig)
        plt.close(fig)

        # テキストログも保存
        out_path_txt = out_dir / f"{name_prefix}_encoder_latent_var_stats.txt"
        with open(out_path_txt, "w") as f:
            f.write(f"N = {N}, K = {K}\n")
            f.write("=== mean_per_dim ===\n")
            f.write(f"global mean(mean_per_dim): {global_mean_of_means}\n")
            f.write(f"global std(mean_per_dim):  {global_std_of_means}\n\n")
            f.write("=== std_per_dim ===\n")
            f.write(f"mean(std_per_dim): {global_mean_std}\n")
            f.write(f"min(std_per_dim):  {global_min_std}\n")
            f.write(f"max(std_per_dim):  {global_max_std}\n\n")

        print(f"[LATENT VAR] saved std histogram -> {out_path_fig.resolve()}")
        print(f"[LATENT VAR] saved stats txt    -> {out_path_txt.resolve()}")

        # wandb / Logger にも保存（数値だけ）
        try:
            run.log(
                {
                    f"{name_prefix}/latent_std_mean": global_mean_std,
                    f"{name_prefix}/latent_std_min": global_min_std,
                    f"{name_prefix}/latent_std_max": global_max_std,
                    f"{name_prefix}/latent_mean_of_means": global_mean_of_means,
                }
            )
        except Exception as e:
            print("[LATENT VAR] Logger logging skipped:", e)


        try:
            del z_all, mean_per_dim, std_per_dim
        except:
            pass
        try:
            del enc_output, encoder_encs, states, actions, optional_fields
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def plot_encoder_latent_tsne(
        self,
        batch,
        jepa: JEPA,
        name_prefix: str = "",
        max_points: int = 5000,
    ):
        """
        学習後の Encoder 潜在を t-SNE で 2 次元に可視化する。

        - 入力 shape / 流れは log_encoder_latent_variance() と同じ
        - 時刻 T × バッチ B を全部まとめて N = T*B サンプルとして扱い、
        flatten した潜在ベクトルに対して t-SNE を適用する。
        """

        from pathlib import Path
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        # ===== 1. Encoder 出力を取得 =====
        states = batch.states.to(self.device).transpose(0, 1)   # [T, B, ...]
        actions = batch.actions.to(self.device).transpose(0, 1) # [T, B, ...]
        optional_fields = get_optional_fields(batch, device=states.device)

        print("[DBG plot_encoder_latent_tsne] states:", states.shape)
        print("[DBG plot_encoder_latent_tsne] actions:", actions.shape)

        enc_output = jepa.forward_posterior(
            states, actions, encode_only=True, **optional_fields
        )
        enc_output = enc_output.backbone_output

        encoder_encs = enc_output.obs_component
        # encoder_encs: (T, B, C, H, W) or (T, B, D, ...)
        print("[DBG plot_encoder_latent_tsne] encoder_encs:", encoder_encs.shape)

        # ===== 2. [T,B,...] → (N, K) に flatten =====
        zs = []
        for x in encoder_encs:  # x: [B, C, H, W] or [B, D, ...]
            x_flat = x.reshape(x.shape[0], -1)  # (B, K)
            zs.append(x_flat)

        z_all = torch.cat(zs, dim=0)  # (N, K), N = T * B
        N, K = z_all.shape
        if N > max_points:
            idx = torch.randperm(N)[:max_points]
            z_all = z_all[idx]
            N = max_points

        print(f"[TSNE] using N = {N}, K = {K}")

        # ===== 3. t-SNE 入力用に前処理 =====
        # mean 0 / var 1 に軽く正規化しておくと t-SNE が安定しやすい
        z_all = z_all.detach().cpu()
        z_mean = z_all.mean(dim=0, keepdim=True)
        z_std = z_all.std(dim=0, keepdim=True)
        z_std[z_std < 1e-6] = 1.0  # ほぼ一定の次元で発散しないように
        z_norm = (z_all - z_mean) / z_std
        z_np = z_norm.numpy()

        # ===== 4. t-SNE の実行 =====
        # パラメータは標準的な設定（必要ならあとで調整）
        tsne = TSNE(
            n_components=2,
            perplexity=min(30.0, max(5.0, N / 50.0)),  # サンプル数に応じて少しだけ調整
            max_iter=1000,
            init="random",
            learning_rate="auto",
            metric="euclidean",
        )
        print("[TSNE] fitting...")
        z_2d = tsne.fit_transform(z_np)  # (N, 2)
        print("[TSNE] done.")

        # ===== 5. プロット & 保存 =====
        run = Logger.run()
        if run.output_path is not None:
            run_dir = Path(run.output_path)
        else:
            run_dir = Path(".")

        out_dir = run_dir / "latent_tsne"  # latent_variance と同列
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=200)
        ax.scatter(z_2d[:, 0], z_2d[:, 1], s=6, alpha=0.6)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.set_title(f"{name_prefix} encoder latent t-SNE")
        fig.tight_layout()

        out_path_fig = out_dir / f"{name_prefix}_encoder_latent_tsne.png"
        plt.savefig(out_path_fig)
        plt.close(fig)

        print(f"[LATENT TSNE] saved t-SNE plot -> {out_path_fig.resolve()}")

        # （スカラーはないので Logger への log は省略 or 必要なら追加）

        # ===== 6. メモリ開放 =====
        try:
            del z_all, z_norm, z_np, z_2d
        except:
            pass
        try:
            del enc_output, encoder_encs, states, actions, optional_fields
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()



            

    @torch.no_grad()
    def plot_cca(
        self,
        batch,
        jepa: JEPA,
        prober: torch.nn.Module,
        prober_open: torch.nn.Module,
        prober_bluebox_locs: torch.nn.Module = None,
        prober_bluebox_locs_open: torch.nn.Module = None,
        enc_prober: torch.nn.Module = None,
        enc_prober_bluebox: torch.nn.Module = None,
        normalizer: "Normalizer" = None,
        name_prefix: str = "",
        idxs: Optional[List[int]] = None,
        notebook: bool = False,
        pixel_mapper = None,
        vis_dynamics_closed_featuremap: bool = False,
        vis_dynamics_open_featuremap: bool = False,
        vis_encoder_featuremap: bool = False,
    ):
        """
        encoder の潜在列と closed-forward の潜在列を CCA で共通空間に射影し、
        上位 k=3 の正準変数で可視化（2D/3D）します。
        - 学習/推論は JEPA から取得した latent そのもの（prober 不使用）
        - CCA は batch×time をまとめて fit し、全サンプルで共通軸に射影
        """
        import matplotlib as mpl
        from matplotlib.lines import Line2D


        device = self.device


        states = batch.states.to(device).transpose(0, 1)  # [T, B, C, H, W] 想定
        actions = batch.actions.to(device).transpose(0, 1)

        optional_fields = get_optional_fields(batch, device=states.device)


        pred_output = jepa.forward_posterior(states, actions, **optional_fields).pred_output
        
        
        if getattr(pred_output, "obs_component", None) is not None:
            closed_lat_seq = pred_output.obs_component  # [T,B,C,H,W] or [T,B,D]
        else:
            closed_lat_seq = pred_output.predictions   # [T,B,C,H,W] or [T,B,D]

        print("pred_output.obs_component", None if pred_output.obs_component is None else pred_output.obs_component.shape)
        print("pred_output.predictions", None if pred_output.predictions is None else pred_output.predictions.shape)


        enc_output = jepa.forward_posterior(states, actions, encode_only=True, **optional_fields).backbone_output
        encoder_lat_seq = enc_output.obs_component    # [T,B,C,H,W] or [T,B,D]

        print("[DEGUB: plot_cca] closed_lat_seq.shape", closed_lat_seq.shape)
        print("[DEGUB: plot_cca] encoder_lat_seq", encoder_lat_seq.shape)
        
        
        
        # [B,T,D] に正規化
        T_ref, B_ref = states.shape[0], states.shape[1]  # states は [T,B,...]

        def _to_BTD(x, pool="flat"):
            """
            任意の x を [B,T,D] に正規化:
            - 3D: [T,B,D] or [B,T,D] -> [B,T,D]
            - 5D: [T,B,C,H,W] or [B,T,C,H,W] -> [B,T,D]（D=C*H*W or GAPでC）
            - それ以外は先頭2軸を [B,T] に揃え、残りを flatten
            pool: "flat" なら D=C*H*W、"gap" なら D=C（Global Avg Pool）
            """
            assert torch.is_tensor(x)

            if x.dim() == 3:
                # [T,B,D] or [B,T,D]
                if x.shape[0] == T_ref and x.shape[1] == B_ref:
                    x = x.transpose(0, 1).contiguous()       # -> [B,T,D]
                return x

            if x.dim() == 5:
                # [T,B,C,H,W] or [B,T,C,H,W]
                if x.shape[0] == T_ref and x.shape[1] == B_ref:
                    x = x.permute(1, 0, 2, 3, 4).contiguous()  # -> [B,T,C,H,W]
                elif not (x.shape[0] == B_ref and x.shape[1] == T_ref):
                    x = x.transpose(0, 1).contiguous()         # ヒューリスティック
                B, T = x.shape[:2]
                if pool == "gap":
                    x = x.mean(dim=(-2, -1)).contiguous()      # -> [B,T,C]
                else:
                    C, H, W = x.shape[2], x.shape[3], x.shape[4]
                    x = x.reshape(B, T, C * H * W).contiguous()
                return x

            # 4Dなど一般形：先頭2軸を [B,T] に揃えて残り flatten
            if x.shape[0] == T_ref and x.shape[1] == B_ref:
                x = x.transpose(0, 1).contiguous()             # -> [B,T,...]
            elif not (x.shape[0] == B_ref and x.shape[1] == T_ref):
                x = x.transpose(0, 1).contiguous()
            B, T = x.shape[:2]
            D = int(np.prod(x.shape[2:]))
            return x.reshape(B, T, D).contiguous()


        closed_lat_seq  = _to_BTD(closed_lat_seq,  pool="gap")  # or "gap"
        encoder_lat_seq = _to_BTD(encoder_lat_seq, pool="gap")  # or "gap"
        print("[CCA plot]closed_lat_seq final shape:", closed_lat_seq.shape)
        print("[CCA plot]encoder_lat_seq final shape:", encoder_lat_seq.shape)
        print("[CCA plot]closed_lat_seq std:", closed_lat_seq.std().item())
        print("[CCA plot]encoder_lat_seq std:", encoder_lat_seq.std().item())



        print('[CCA plot] normalized:',
            tuple(closed_lat_seq.shape), tuple(encoder_lat_seq.shape))

        B, T, Dc = closed_lat_seq.shape
        _, _, De = encoder_lat_seq.shape

        # ===== 2) CCA を学習（batch×time をまとめて fit）=====
        Z_enc = encoder_lat_seq.reshape(B * T, De).detach().cpu().numpy()
        Z_clo = closed_lat_seq.reshape(B * T, Dc).detach().cpu().numpy()


        k = 3  # 上位3成分を可視化
        k_eff = min(k, De, Dc) 

        cca = CCA(n_components=k_eff, max_iter=5000)  # 反復回数は余裕を持たせる
        U, V = cca.fit_transform(Z_enc, Z_clo)  # U, V: [B*T, k_eff]

        # 正準相関係数（各成分ペアの相関）
        corrs = []
        for i in range(k_eff):
            ui, vi = U[:, i], V[:, i]

            if np.std(ui) < 1e-8 or np.std(vi) < 1e-8:
                corrs.append(0.0)
            else:
                corrs.append(np.corrcoef(ui, vi)[0, 1])
        mean_corr = float(np.mean(corrs)) if len(corrs) > 0 else 0.0

        # [B, T, k_eff] に戻す
        U_bt = U.reshape(B, T, k_eff)
        V_bt = V.reshape(B, T, k_eff)


        
        if idxs is None:
            idxs = list(range(B))  


        color_u = "navy"      # Encoder (U)
        color_v = "firebrick" # Closed (V)

        # --- 2D (CC1-CC2) per-trajectory ---
        if k_eff >= 2:
            U2 = U_bt[:, :, :2]
            V2 = V_bt[:, :, :2]
            lim = float(max(abs(U2).max(), abs(V2).max()))  



            for i in idxs:
                fig2d, ax2d = plt.subplots(1, 1, figsize=(6, 6), dpi=140)

                u2d = U_bt[i, :, :2]
                v2d = V_bt[i, :, :2]


                for t in range(T - 1):
                    ax2d.plot(u2d[t:t+2, 0], u2d[t:t+2, 1], color=color_u, alpha=0.95)
                    ax2d.plot(v2d[t:t+2, 0], v2d[t:t+2, 1], color=color_v, alpha=0.95)

                # 始点・終点
                ax2d.scatter(u2d[0, 0], u2d[0, 1], s=12, c=color_u, label="Encoder (U)")
                ax2d.text(u2d[0, 0], u2d[0, 1], "S", fontsize=9, ha="center", va="center", color=color_u)
                ax2d.text(u2d[-1, 0], u2d[-1, 1], "G", fontsize=9, ha="center", va="center", color=color_u)
                
                ax2d.scatter(v2d[0, 0], v2d[0, 1], s=12, c=color_v, label="Closed (V)")
                ax2d.text(v2d[0,0],  v2d[0,1],  "S", fontsize=9, ha="center", va="center", color=color_v,   zorder=4)
                ax2d.text(v2d[-1,0], v2d[-1,1], "G", fontsize=9, ha="center", va="center", color=color_v, zorder=4)

                ax2d.set_xlim(-lim, lim)
                ax2d.set_ylim(-lim, lim)
                ax2d.set_aspect("equal", adjustable="box")
                ax2d.set_xlabel("CC1")
                ax2d.set_ylabel("CC2")
                ax2d.set_title(f"{name_prefix} | idx={i} | mean corr={mean_corr:.3f}")

                handles = [
                    Line2D([0], [0], color=color_u, lw=2, label="Encoder (U)"),
                    Line2D([0], [0], color=color_v, lw=2, label="Closed (V)")
                ]
                ax2d.legend(handles=handles, loc="best", frameon=True)

                # sm = mpl.cm.ScalarMappable(cmap=cmap_e, norm=norm)
                # sm.set_array([])
                # cbar = fig2d.colorbar(sm, ax=ax2d, fraction=0.046, pad=0.04)
                # cbar.set_label("Time step", fontsize=10)
                # cbar.ax.tick_params(labelsize=8)
                
                if not notebook:
                    Logger.run().log_figure(fig2d, f"{name_prefix}-cca-2d-i{i}", dir_name="cca/cca2d_pertraj")
                    plt.close(fig2d)
                else:
                    plt.show()




        # --- 3D (CC1-CC2-CC3) ---
        if k_eff >= 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa

            for i in idxs:
                fig3d = plt.figure(figsize=(8, 8), dpi=140)
                ax3d = fig3d.add_subplot(111, projection="3d")

                u3d = U_bt[i, :, :3]
                v3d = V_bt[i, :, :3]

                for t in range(T - 1):
                    ax3d.plot(u3d[t:t+2, 0], u3d[t:t+2, 1], u3d[t:t+2, 2], color=color_u, alpha=0.95)
                    ax3d.plot(v3d[t:t+2, 0], v3d[t:t+2, 1], v3d[t:t+2, 2], color=color_v, alpha=0.95)

                # --- Start/Goal markers for 3D (U=Encoder, V=Closed) ---
                # 目印サイズとオフセット（重なり回避用）
                s_size = 24
                off = 0.02 * float(max(abs(U_bt[:, :, :3]).max(), abs(V_bt[:, :, :3]).max()))

                # U (青)
                ax3d.scatter(u3d[0,0],  u3d[0,1],  u3d[0,2],  s=s_size, c=color_u,   depthshade=False, zorder=5)
                ax3d.scatter(u3d[-1,0], u3d[-1,1], u3d[-1,2], s=s_size, c=color_u, depthshade=False, zorder=5)
                ax3d.text(u3d[0,0]+off,  u3d[0,1]+off,  u3d[0,2]+off,  "S",
                        color=color_u,   fontsize=9, zorder=6)
                ax3d.text(u3d[-1,0]+off, u3d[-1,1]+off, u3d[-1,2]+off, "G",
                        color=color_u, fontsize=9, zorder=6)

                # V (赤)
                ax3d.scatter(v3d[0,0],  v3d[0,1],  v3d[0,2],  s=s_size, c=color_v,   depthshade=False, zorder=5)
                ax3d.scatter(v3d[-1,0], v3d[-1,1], v3d[-1,2], s=s_size, c=color_v, depthshade=False, zorder=5)
                ax3d.text(v3d[0,0]+off,  v3d[0,1]+off,  v3d[0,2]+off,  "S",
                        color=color_v,   fontsize=9, zorder=6)
                ax3d.text(v3d[-1,0]+off, v3d[-1,1]+off, v3d[-1,2]+off, "G",
                        color=color_v, fontsize=9, zorder=6)




                ax3d.set_xlabel("CC1")
                ax3d.set_ylabel("CC2")
                ax3d.set_zlabel("CC3")
                ax3d.set_title(
                    f"CCA (3D) — {name_prefix} | idx={i} | "
                    f"mean corr={mean_corr:.3f}, "
                    f"per-comp={','.join(f'{c:.2f}' for c in corrs)}"
                )
                
                handles = [
                    Line2D([0], [0], color=color_u, lw=2, label="Encoder (U)"),
                    Line2D([0], [0], color=color_v, lw=2, label="Closed (V)")
                ]
                ax3d.legend(handles=handles, loc="upper left", frameon=True)


                if not notebook:
                    Logger.run().log_figure(fig3d, f"{name_prefix}-cca-3d-i{i}", dir_name="cca/cca3d_pertraj")
                    plt.close(fig3d)
                else:
                    plt.show()



        # τごとの平均コサイン類似度（線形整合なしの素のU-V距離ではなく、U vs V の同次元での相関を簡易に）
        # ここでは U,V はすでに「対応空間」なので、各τでの cos を出すのも有用
        try:
            import torch.nn.functional as F
            U_t = torch.from_numpy(U_bt)  # [B,T,k]
            V_t = torch.from_numpy(V_bt)
            # 正規化して cos 類似度
            def _cos_mean(a, b, eps=1e-8):
                a = a / (a.norm(dim=-1, keepdim=True) + eps)
                b = b / (b.norm(dim=-1, keepdim=True) + eps)
                return (a * b).sum(-1).mean().item()
            cos_over_time = []
            for t in range(T):
                cos_over_time.append(_cos_mean(U_t[:, t, :k_eff], V_t[:, t, :k_eff]))

            figc, axc = plt.subplots(1, 1, figsize=(7, 3), dpi=140)
            axc.plot(range(T), cos_over_time, marker='o', linewidth=1.5)
            axc.set_xlabel("t (horizon)")
            axc.set_ylabel("cosine(U_t, V_t)")
            axc.set_title(f"Timewise Cosine in CCA space — mean={np.mean(cos_over_time):.3f}")
            
            figc.tight_layout()
            if not notebook:
                Logger.run().log_figure(figc, f"{name_prefix}-cca-timewise-cosine", dir_name="cca/cca_cos")
                plt.close(figc)
            else:
                plt.show()
        except Exception:
            pass
    
        plt.close('all')
        try:
            del U_bt, V_bt, U, V, Z_enc, Z_clo
        except Exception:
            pass
        try:
            del closed_lat_seq, encoder_lat_seq
        except Exception:
            pass
        try:
            del pred_output, enc_output, states, actions, optional_fields
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()





    @torch.no_grad()
    def plot_pca(
        self,
        batch,
        jepa: "JEPA",
        name_prefix: str = "",
        idxs: Optional[List[int]] = None,
        notebook: bool = False,
        pool: str = "gap",              # "gap"/"flat"
        k: int = 3,                     
        align_closed: bool = True,     
    ):
        """
        encoder の潜在列で PCA 軸を学習し、同じ PCA 基底へ
        - encoder 列（基準）
        - closed-forward 列
        を射影して 2D/3D 可視化する。

        * 入力:
            - batch: states/actions を含む
            - jepa : forward_posterior から encoder/closed の潜在を得る
        * 出力:
            - Logger.run().log_figure(...) で 2D/3D 図を保存（notebook=True なら plt.show）
            - 主成分寄与率や時間方向の cos 類似度も表示
        """
        import matplotlib as mpl
        from matplotlib.lines import Line2D
        from sklearn.decomposition import PCA

        device = self.device
        states = batch.states.to(device).transpose(0, 1)  # [T,B,...]
        actions = batch.actions.to(device).transpose(0, 1)
        optional_fields = get_optional_fields(batch, device=states.device)

        # --- 予測（closed）とエンコード（encoder）潜在の取得 ---
        pred_output = jepa.forward_posterior(states, actions, **optional_fields).pred_output
        if getattr(pred_output, "obs_component", None) is not None:
            closed_lat_seq = pred_output.obs_component  # [T,B,C,H,W] or [T,B,D]
        else:
            closed_lat_seq = pred_output.predictions

        enc_output = jepa.forward_posterior(states, actions, encode_only=True, **optional_fields).backbone_output
        encoder_lat_seq = enc_output.obs_component  # [T,B,C,H,W] or [T,B,D]

        # --- [B,T,D] に正規化 ---
        T_ref, B_ref = states.shape[0], states.shape[1]

        def _to_BTD(x, pool="flat"):
            assert torch.is_tensor(x)
            if x.dim() == 3:
                # [T,B,D] or [B,T,D]
                if x.shape[0] == T_ref and x.shape[1] == B_ref:
                    x = x.transpose(0, 1).contiguous()
                return x
            if x.dim() == 5:
                # [T,B,C,H,W] or [B,T,C,H,W]
                if x.shape[0] == T_ref and x.shape[1] == B_ref:
                    x = x.permute(1, 0, 2, 3, 4).contiguous()
                elif not (x.shape[0] == B_ref and x.shape[1] == T_ref):
                    x = x.transpose(0, 1).contiguous()
                B, T = x.shape[:2]
                if pool == "gap":
                    x = x.mean(dim=(-2, -1)).contiguous()     # -> [B,T,C]
                else:
                    C, H, W = x.shape[2:]
                    x = x.reshape(B, T, C*H*W).contiguous()
                return x
            # 一般形
            if x.shape[0] == T_ref and x.shape[1] == B_ref:
                x = x.transpose(0, 1).contiguous()
            elif not (x.shape[0] == B_ref and x.shape[1] == T_ref):
                x = x.transpose(0, 1).contiguous()
            B, T = x.shape[:2]
            D = int(np.prod(x.shape[2:]))
            return x.reshape(B, T, D).contiguous()

        closed_lat_seq  = _to_BTD(closed_lat_seq,  pool=pool)
        encoder_lat_seq = _to_BTD(encoder_lat_seq, pool=pool)

        B, T, Dc = closed_lat_seq.shape
        _, _, De = encoder_lat_seq.shape
        if idxs is None:
            idxs = list(range(B))

        # --- PCA を encoder で学習（scikit-learn は内部でセンタリングする）---
        Z_enc = encoder_lat_seq.reshape(B*T, De).detach().cpu().numpy()
        pca = PCA(n_components=min(k, De))
        U_enc = pca.fit_transform(Z_enc)  # [B*T, k_eff]
        expl = pca.explained_variance_ratio_
        k_eff = U_enc.shape[1]

        # --- closed を同一基底へ射影 ---
        Z_clo = closed_lat_seq.reshape(B*T, Dc).detach().cpu().numpy()
        if Dc == De:
            Z_clo_in_enc_dim = Z_clo
        else:
            if not align_closed:
                raise ValueError(
                    f"Dc({Dc}) != De({De}). align_closed=False の場合、次元を揃える必要があります。"
                )

            A, *_ = np.linalg.lstsq(Z_clo, Z_enc, rcond=None)
            Z_clo_in_enc_dim = Z_clo @ A

        V_clo = pca.transform(Z_clo_in_enc_dim)  # [B*T, k_eff]

        # [B,T,k_eff] へ戻す
        U_bt = U_enc.reshape(B, T, k_eff)
        V_bt = V_clo.reshape(B, T, k_eff)

        #2D可視化
        color_u = "navy"       # Encoder
        color_v = "firebrick"  # Closed
        if k_eff >= 2:
            U2 = U_bt[:, :, :2]
            V2 = V_bt[:, :, :2]
            lim = float(max(abs(U2).max(), abs(V2).max()))

            for i in idxs:
                fig2d, ax2d = plt.subplots(1, 1, figsize=(6, 6), dpi=140)
                u2d = U_bt[i, :, :2]
                v2d = V_bt[i, :, :2]
                for t in range(T - 1):
                    ax2d.plot(u2d[t:t+2, 0], u2d[t:t+2, 1], color=color_u, alpha=0.95)
                    ax2d.plot(v2d[t:t+2, 0], v2d[t:t+2, 1], color=color_v, alpha=0.95)
                ax2d.scatter(u2d[0,0], u2d[0,1], s=12, c=color_u, label="Encoder")
                ax2d.text(u2d[0,0], u2d[0,1], "S", fontsize=9, ha="center", va="center", color=color_u)
                ax2d.text(u2d[-1,0], u2d[-1,1], "G", fontsize=9, ha="center", va="center", color=color_u)

                ax2d.scatter(v2d[0,0], v2d[0,1], s=12, c=color_v, label="Closed")
                ax2d.text(v2d[0,0], v2d[0,1], "S", fontsize=9, ha="center", va="center", color=color_v)
                ax2d.text(v2d[-1,0], v2d[-1,1], "G", fontsize=9, ha="center", va="center", color=color_v)

                ax2d.set_xlim(-lim, lim); ax2d.set_ylim(-lim, lim)
                ax2d.set_aspect("equal", adjustable="box")
                ax2d.set_xlabel("PC1"); ax2d.set_ylabel("PC2")
                ax2d.set_title(
                    f"{name_prefix} | idx={i} | PCA (top-{k_eff}): "
                    + ", ".join(f"{v:.2f}" for v in expl[:k_eff])
                )
                handles = [
                    Line2D([0],[0], color=color_u, lw=2, label="Encoder"),
                    Line2D([0],[0], color=color_v, lw=2, label="Closed"),
                ]
                ax2d.legend(handles=handles, loc="best", frameon=True)
                if not notebook:
                    Logger.run().log_figure(fig2d, f"{name_prefix}-pca-2d-i{i}", dir_name="pca/pca2d_pertraj")
                    plt.close(fig2d)
                else:
                    plt.show()

        #3D可視化
        if k_eff >= 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            for i in idxs:
                fig3d = plt.figure(figsize=(8, 8), dpi=140)
                ax3d = fig3d.add_subplot(111, projection="3d")
                u3d = U_bt[i, :, :3]; v3d = V_bt[i, :, :3]

                for t in range(T - 1):
                    ax3d.plot(u3d[t:t+2,0], u3d[t:t+2,1], u3d[t:t+2,2], color=color_u, alpha=0.95)
                    ax3d.plot(v3d[t:t+2,0], v3d[t:t+2,1], v3d[t:t+2,2], color=color_v, alpha=0.95)

                s_size = 24
                off = 0 #0.02 * float(max(abs(U_bt[:, :, :3]).max(), abs(V_bt[:, :, :3]).max()))
                ax3d.scatter(u3d[0,0], u3d[0,1], u3d[0,2], s=s_size, c=color_u, depthshade=False)
                ax3d.scatter(u3d[-1,0], u3d[-1,1], u3d[-1,2], s=s_size, c=color_u, depthshade=False)
                ax3d.text(u3d[0,0]+off,  u3d[0,1]+off,  u3d[0,2]+off,  "S", color=color_u, fontsize=9)
                ax3d.text(u3d[-1,0]+off, u3d[-1,1]+off, u3d[-1,2]+off, "G", color=color_u, fontsize=9)

                ax3d.scatter(v3d[0,0], v3d[0,1], v3d[0,2], s=s_size, c=color_v, depthshade=False)
                ax3d.scatter(v3d[-1,0], v3d[-1,1], v3d[-1,2], s=s_size, c=color_v, depthshade=False)
                ax3d.text(v3d[0,0]+off,  v3d[0,1]+off,  v3d[0,2]+off,  "S", color=color_v, fontsize=9)
                ax3d.text(v3d[-1,0]+off, v3d[-1,1]+off, v3d[-1,2]+off, "G", color=color_v, fontsize=9)

                ax3d.set_xlabel("PC1"); ax3d.set_ylabel("PC2"); ax3d.set_zlabel("PC3")
                ax3d.set_title(
                    f"PCA (3D) — {name_prefix} | var exp: "
                    + ", ".join(f"{v:.2f}" for v in expl[:3])
                )
                handles = [
                    Line2D([0],[0], color=color_u, lw=2, label="Encoder"),
                    Line2D([0],[0], color=color_v, lw=2, label="Closed"),
                ]
                ax3d.legend(handles=handles, loc="upper left", frameon=True)
                if not notebook:
                    Logger.run().log_figure(fig3d, f"{name_prefix}-pca-3d-i{i}", dir_name="pca/pca3d_pertraj")
                    plt.close(fig3d)
                else:
                    plt.show()


        try:
            import torch.nn.functional as F
            U_t = torch.from_numpy(U_bt)  # [B,T,k_eff]
            V_t = torch.from_numpy(V_bt)
            def _cos_mean(a, b, eps=1e-8):
                a = a / (a.norm(dim=-1, keepdim=True) + eps)
                b = b / (b.norm(dim=-1, keepdim=True) + eps)
                return (a * b).sum(-1).mean().item()
            cos_over_time = []
            for t in range(T):
                cos_over_time.append(_cos_mean(U_t[:, t, :k_eff], V_t[:, t, :k_eff]))
            figc, axc = plt.subplots(1, 1, figsize=(7,3), dpi=140)
            axc.plot(range(T), cos_over_time, marker='o', linewidth=1.5)
            axc.set_xlabel("t (horizon)")
            axc.set_ylabel("cosine in PCA space")
            axc.set_title(
                f"Timewise Cosine — mean={np.mean(cos_over_time):.3f} | var exp (top-{k_eff}): "
                + ", ".join(f"{v:.2f}" for v in expl[:k_eff])
            )
            figc.tight_layout()
            if not notebook:
                Logger.run().log_figure(figc, f"{name_prefix}-pca-timewise-cosine", dir_name="pca/pca_cos")
                plt.close(figc)
            else:
                plt.show()
        except Exception:
            pass

        plt.close('all')
        try:
            del U_bt, V_bt, U_enc, V_clo, Z_enc, Z_clo
        except Exception:
            pass
        try:
            del closed_lat_seq, encoder_lat_seq, pred_output, enc_output, states, actions, optional_fields
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()



    @torch.no_grad()
    def plot_pca_open_closed(
        self,
        batch,
        jepa: "JEPA",
        name_prefix: str = "",
        idxs: Optional[List[int]] = None,
        notebook: bool = False,
        pool: str = "gap",      # "gap"/"flat"
        k: int = 3,             # PCA 次元
    ):
        """
        open-forward 列を基準に PCA 軸を学習し、
        - open-forward 列（基準）
        - closed-forward 列
        を「同じ PCA 空間」に射影して 2D/3D で可視化する。

        encoder ではなく open を anchor にしている点だけが元の plot_pca と異なる。
        """
        import matplotlib as mpl
        from matplotlib.lines import Line2D
        from sklearn.decomposition import PCA
        import numpy as np
        import gc
        import torch

        device = self.device
        states = batch.states.to(device).transpose(0, 1)  # [T,B,...]
        actions = batch.actions.to(device).transpose(0, 1)
        optional_fields = get_optional_fields(batch, device=states.device)

        # ---- closed-forward 潜在列 ----
        closed_res = jepa.forward_posterior(states, actions, **optional_fields)
        pred_closed = closed_res.pred_output
        if getattr(pred_closed, "obs_component", None) is not None:
            closed_lat_seq = pred_closed.obs_component    # [T,B,C,H,W] or [T,B,D]
        else:
            closed_lat_seq = pred_closed.predictions      # [T,B,D]

        # ---- open-forward 潜在列 ----
        open_res = jepa.forward_open(states, actions, **optional_fields)
        pred_open = open_res.pred_output
        if getattr(pred_open, "obs_component", None) is not None:
            open_lat_seq = pred_open.obs_component        # [T,B,C,H,W] or [T,B,D]
        else:
            open_lat_seq = pred_open.predictions          # [T,B,D]

        T_ref, B_ref = states.shape[0], states.shape[1]

        def _to_BTD(x, pool="flat"):
            """
            [T,B,...] or [B,T,...] -> [B,T,D]
            （元の plot_pca と同じヘルパ）
            """
            assert torch.is_tensor(x)
            if x.dim() == 3:
                if x.shape[0] == T_ref and x.shape[1] == B_ref:
                    x = x.transpose(0, 1).contiguous()   # [B,T,D]
                return x
            if x.dim() == 5:
                # [T,B,C,H,W] or [B,T,C,H,W]
                if x.shape[0] == T_ref and x.shape[1] == B_ref:
                    x = x.permute(1, 0, 2, 3, 4).contiguous()  # [B,T,C,H,W]
                elif not (x.shape[0] == B_ref and x.shape[1] == T_ref):
                    x = x.transpose(0, 1).contiguous()
                B, T = x.shape[:2]
                if pool == "gap":
                    x = x.mean(dim=(-2, -1)).contiguous()      # -> [B,T,C]
                else:
                    C, H, W = x.shape[2:]
                    x = x.reshape(B, T, C * H * W).contiguous()
                return x
            # fallback
            if x.shape[0] == T_ref and x.shape[1] == B_ref:
                x = x.transpose(0, 1).contiguous()
            elif not (x.shape[0] == B_ref and x.shape[1] == T_ref):
                x = x.transpose(0, 1).contiguous()
            B, T = x.shape[:2]
            D = int(np.prod(x.shape[2:]))
            return x.reshape(B, T, D).contiguous()

        # [B,T,D] にそろえる（まだGPU上）
        open_lat_seq   = _to_BTD(open_lat_seq,   pool=pool)
        closed_lat_seq = _to_BTD(closed_lat_seq, pool=pool)

        B, T, Do = open_lat_seq.shape
        B2, T2, Dc = closed_lat_seq.shape
        assert B == B2 and T == T2, f"[PCA-open] shape mismatch: open {open_lat_seq.shape}, closed {closed_lat_seq.shape}"

        if idxs is None:
            idxs = list(range(B))

        # ===== ここで CPU に移して GPU を早めに解放 =====
        open_lat_seq_cpu   = open_lat_seq.detach().to("cpu")
        closed_lat_seq_cpu = closed_lat_seq.detach().to("cpu")

        try:
            del open_lat_seq, closed_lat_seq
            del closed_res, open_res, pred_open, pred_closed
            del states, actions, optional_fields
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        # ---- PCA を open-forward で学習 (CPU) ----
        Z_open = open_lat_seq_cpu.reshape(B * T, Do).numpy()
        pca = PCA(n_components=min(k, Do))
        U_open = pca.fit_transform(Z_open)  # [B*T, k_eff]
        expl = pca.explained_variance_ratio_
        k_eff = U_open.shape[1]

        # ---- closed-forward を同じ基底へ射影 (CPU) ----
        if Dc != Do:
            # open/closed の次元が違う場合は線形射影で合わせる
            Z_closed = closed_lat_seq_cpu.reshape(B * T, Dc).numpy()
            A, *_ = np.linalg.lstsq(Z_closed, Z_open, rcond=None)
            Z_closed_in_open_dim = Z_closed @ A
        else:
            Z_closed_in_open_dim = closed_lat_seq_cpu.reshape(B * T, Dc).numpy()

        V_closed = pca.transform(Z_closed_in_open_dim)  # [B*T, k_eff]

        # [B,T,k_eff] に戻す (still CPU, numpy → torch)
        U_bt = U_open.reshape(B, T, k_eff)
        V_bt = V_closed.reshape(B, T, k_eff)

        # ---- 2D 可視化 ----
        color_open   = "tab:orange"
        color_closed = "tab:blue"

        import matplotlib.pyplot as plt

        if k_eff >= 2:
            U2 = U_bt[:, :, :2]
            V2 = V_bt[:, :, :2]
            lim = float(max(abs(U2).max(), abs(V2).max()))

            for i in idxs:
                fig2d, ax2d = plt.subplots(1, 1, figsize=(6, 6), dpi=140)
                u2d = U_bt[i, :, :2]
                v2d = V_bt[i, :, :2]

                for t in range(T - 1):
                    ax2d.plot(u2d[t:t+2, 0], u2d[t:t+2, 1], color=color_open,   alpha=0.95)
                    ax2d.plot(v2d[t:t+2, 0], v2d[t:t+2, 1], color=color_closed, alpha=0.95)

                # start / goal マーク（Open）
                ax2d.scatter(u2d[0,0],  u2d[0,1],  s=12, c=color_open,   label="Open")
                ax2d.text(u2d[0,0],  u2d[0,1],  "S", fontsize=9, ha="center", va="center", color=color_open)
                ax2d.text(u2d[-1,0], u2d[-1,1], "G", fontsize=9, ha="center", va="center", color=color_open)

                # start / goal マーク（Closed）
                ax2d.scatter(v2d[0,0],  v2d[0,1],  s=12, c=color_closed, label="Closed")
                ax2d.text(v2d[0,0],  v2d[0,1],  "S", fontsize=9, ha="center", va="center", color=color_closed)
                ax2d.text(v2d[-1,0], v2d[-1,1], "G", fontsize=9, ha="center", va="center", color=color_closed)

                ax2d.set_xlim(-lim, lim); ax2d.set_ylim(-lim, lim)
                ax2d.set_aspect("equal", adjustable="box")
                ax2d.set_xlabel("PC1"); ax2d.set_ylabel("PC2")
                ax2d.set_title(
                    f"{name_prefix} | idx={i} | PCA(open-anchor) (top-{k_eff}): "
                    + ", ".join(f"{v:.2f}" for v in expl[:k_eff])
                )
                handles = [
                    Line2D([0],[0], color=color_open,   lw=2, label="Open"),
                    Line2D([0],[0], color=color_closed, lw=2, label="Closed"),
                ]
                ax2d.legend(handles=handles, loc="best", frameon=True)

                if not notebook:
                    Logger.run().log_figure(
                        fig2d,
                        f"{name_prefix}-pca_openanchor-2d-i{i}",
                        dir_name="pca_open_anchor/pca2d_pertraj",
                    )
                    plt.close(fig2d)
                else:
                    plt.show()

        # ---- 3D 可視化 ----
        if k_eff >= 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa

            for i in idxs:
                fig3d = plt.figure(figsize=(8, 8), dpi=140)
                ax3d = fig3d.add_subplot(111, projection="3d")
                u3d = U_bt[i, :, :3]
                v3d = V_bt[i, :, :3]

                for t in range(T - 1):
                    ax3d.plot(u3d[t:t+2,0], u3d[t:t+2,1], u3d[t:t+2,2], color=color_open,   alpha=0.95)
                    ax3d.plot(v3d[t:t+2,0], v3d[t:t+2,1], v3d[t:t+2,2], color=color_closed, alpha=0.95)

                s_size = 24
                off = 0.0
                ax3d.scatter(u3d[0,0],  u3d[0,1],  u3d[0,2],  s=s_size, c=color_open,   depthshade=False)
                ax3d.scatter(u3d[-1,0], u3d[-1,1], u3d[-1,2], s=s_size, c=color_open,   depthshade=False)
                ax3d.text(u3d[0,0]+off,  u3d[0,1]+off,  u3d[0,2]+off,  "S", color=color_open,   fontsize=9)
                ax3d.text(u3d[-1,0]+off, u3d[-1,1]+off, u3d[-1,2]+off, "G", color=color_open,   fontsize=9)

                ax3d.scatter(v3d[0,0],  v3d[0,1],  v3d[0,2],  s=s_size, c=color_closed, depthshade=False)
                ax3d.scatter(v3d[-1,0], v3d[-1,1], v3d[-1,2], s=s_size, c=color_closed, depthshade=False)
                ax3d.text(v3d[0,0]+off,  v3d[0,1]+off,  v3d[0,2]+off,  "S", color=color_closed, fontsize=9)
                ax3d.text(v3d[-1,0]+off, v3d[-1,1]+off, v3d[-1,2]+off, "G", color=color_closed, fontsize=9)

                ax3d.set_xlabel("PC1"); ax3d.set_ylabel("PC2"); ax3d.set_zlabel("PC3")
                ax3d.set_title(
                    f"PCA(open-anchor,3D) — {name_prefix} | var exp: "
                    + ", ".join(f"{v:.2f}" for v in expl[:3])
                )
                handles = [
                    Line2D([0],[0], color=color_open,   lw=2, label="Open"),
                    Line2D([0],[0], color=color_closed, lw=2, label="Closed"),
                ]
                ax3d.legend(handles=handles, loc="upper left", frameon=True)
                if not notebook:
                    Logger.run().log_figure(
                        fig3d,
                        f"{name_prefix}-pca_openanchor-3d-i{i}",
                        dir_name="pca_open_anchor/pca3d_pertraj",
                    )
                    plt.close(fig3d)
                else:
                    plt.show()

        # ---- 時間方向の cos 類似度（open vs closed） ----
        try:
            U_t = torch.from_numpy(U_bt)   # [B,T,k_eff]
            V_t = torch.from_numpy(V_bt)

            def _cos_mean(a, b, eps=1e-8):
                a = a / (a.norm(dim=-1, keepdim=True) + eps)
                b = b / (b.norm(dim=-1, keepdim=True) + eps)
                return (a * b).sum(-1).mean().item()

            cos_over_time = []
            for t in range(T):
                cos_over_time.append(_cos_mean(U_t[:, t, :k_eff], V_t[:, t, :k_eff]))

            figc, axc = plt.subplots(1, 1, figsize=(7, 3), dpi=140)
            axc.plot(range(T), cos_over_time, marker='o', linewidth=1.5)
            axc.set_xlabel("t (horizon)")
            axc.set_ylabel("cosine in PCA(open) space")
            axc.set_title(
                f"Timewise Cosine (open vs closed) — mean={np.mean(cos_over_time):.3f} | var exp (top-{k_eff}): "
                + ", ".join(f"{v:.2f}" for v in expl[:k_eff])
            )
            figc.tight_layout()
            if not notebook:
                Logger.run().log_figure(
                    figc,
                    f"{name_prefix}-pca_openanchor-timewise-cosine",
                    dir_name="pca_open_anchor/pca_cos",
                )
                plt.close(figc)
            else:
                plt.show()
        except Exception:
            pass

        plt.close("all")
        try:
            del U_bt, V_bt, U_open, V_closed, Z_open, Z_closed_in_open_dim
        except Exception:
            pass
        try:
            del open_lat_seq_cpu, closed_lat_seq_cpu
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()








    # encoder 出力で学習させたprober を共通利用
    # ポスター用
    @torch.no_grad()
    def plot_prober_predictions_by_encprober_FOR_POSTER(
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
        vis_encoder_featuremap: bool = True,
        
    ):
        import gc
        import torch

        assert enc_prober is not None, "enc_prober is required"
        assert enc_prober_bluebox is not None, "enc_prober_bluebox is required"

        device = self.device

        # データバッチ
        states = batch.states.to(device).transpose(0, 1) #torch.Size([64, 50, 3, 64, 64])
        actions = batch.actions.to(device).transpose(0, 1)

        optional_fields = get_optional_fields(batch, device=states.device)

        # closed-forward 出力列
        closed_res = jepa.forward_posterior(
            states, actions, **optional_fields
        )
        pred_output = closed_res.pred_output

        # open-forward 出力列
        open_res = jepa.forward_open(
            states, actions, **optional_fields
        )
        pred_output_open = open_res.pred_output
        
        # encoderの出力列 (encode_only で余計なものを出さない)
        enc_res = jepa.forward_posterior(
            states, actions, encode_only=True, **optional_fields
        )
        enc_output = enc_res.backbone_output

        # encoder 出力列から画像ベースに分離
        encoder_encs = enc_output.obs_component  # [T,B,C,H,W] 想定
        
        # closed-forward 出力列から画像ベースに分離
        if pred_output.obs_component is not None: ##
            pred_encs = pred_output.obs_component
        else: #x
            pred_encs = pred_output.predictions

        # open-forward 出力列から画像ベースに分離
        if pred_output_open.obs_component is not None: ##
            pred_encs_open = pred_output_open.obs_component
        else: #x
            pred_encs_open = pred_output_open.predictions

        # detach（逆伝播グラフを完全に切る）
        encoder_encs   = encoder_encs.detach()
        pred_encs      = pred_encs.detach()
        pred_encs_open = pred_encs_open.detach()

        # ===== プロバー通過（まだGPU上） =====
        # encoder出力列
        if enc_prober is not None:
            pred_enc_locs = torch.stack([enc_prober(x) for x in encoder_encs], dim=1)
            pred_enc_locs = normalizer.unnormalize_location(pred_enc_locs)
            
        # closed-forward出力列 --> enc_prober
        pred_locs_clsfwd_encprb = torch.stack([enc_prober(x) for x in pred_encs], dim=1)
        pred_locs_clsfwd_encprb = normalizer.unnormalize_location(pred_locs_clsfwd_encprb)
        
        # open-forward出力列 --> enc_prober 
        pred_locs_opnfwd_encprb = torch.stack([enc_prober(x) for x in pred_encs_open], dim=1)
        pred_locs_opnfwd_encprb = normalizer.unnormalize_location(pred_locs_opnfwd_encprb)
        
        # TODO bluebox_locs
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
        
        # encoder出力列 (bluebox)
        pred_enc_bluebox_locs = torch.stack([enc_prober_bluebox(x) for x in encoder_encs], dim=1)
        pred_enc_bluebox_locs = normalizer.unnormalize_bluebox_locs(pred_enc_bluebox_locs)

        # closed-forward出力列 --> enc_prober (bluebox)
        pred_bluebox_locs_clsfwd_encprb = torch.stack([enc_prober_bluebox(x) for x in pred_encs], dim=1)
        pred_bluebox_locs_clsfwd_encprb = normalizer.unnormalize_bluebox_locs(pred_bluebox_locs_clsfwd_encprb)
        
        # open-forward出力列 --> enc_prober (bluebox)
        pred_bluebox_locs_opnfwd_encprb = torch.stack([enc_prober_bluebox(x) for x in pred_encs_open], dim=1)
        pred_bluebox_locs_opnfwd_encprb = normalizer.unnormalize_bluebox_locs(pred_bluebox_locs_opnfwd_encprb)

        # ===== ここで全部 CPU に移し、GPU テンソルを早めに解放 =====
        pred_enc_locs              = pred_enc_locs.detach().cpu()
        pred_locs_clsfwd_encprb    = pred_locs_clsfwd_encprb.detach().cpu()
        pred_locs_opnfwd_encprb    = pred_locs_opnfwd_encprb.detach().cpu()
        pred_enc_bluebox_locs      = pred_enc_bluebox_locs.detach().cpu()
        pred_bluebox_locs_clsfwd_encprb = pred_bluebox_locs_clsfwd_encprb.detach().cpu()
        pred_bluebox_locs_opnfwd_encprb = pred_bluebox_locs_opnfwd_encprb.detach().cpu()

        # pred_encs 系はもう不要
        try:
            del encoder_encs, pred_encs, pred_encs_open
        except Exception:
            pass

        # pred_output 系もここで解放
        try:
            del closed_res, open_res, enc_res
            del pred_output, pred_output_open, enc_output
        except Exception:
            pass

        # pred_locs is of shape (batch_size, time, 1, 2)
        if idxs is None: ##
            idxs = list(range(min(pred_locs_clsfwd_encprb.shape[0], 64)))

        # GT は CPU でOK
        gt_locations = normalizer.unnormalize_location(batch.locations).cpu()
        gt_bluebox_locations = normalizer.unnormalize_bluebox_locs(batch.bluebox_locs).cpu()

        # states/actions/optional_fields はここまでで用は済んだので解放
        try:
            del states, actions, optional_fields
        except Exception:
            pass

        import matplotlib.pyplot as plt
        import numpy as np
        from tqdm import tqdm

        for i in tqdm(idxs, desc=f"Plotting {name_prefix}"):
            fig, axes = plt.subplots(1, 2, figsize=(18, 12), dpi=200)

            #画像表示
            # ax_img = axes[0][0]
            # img = normalizer.unnormalize_state(batch.states)
            # init_img = img[i, 0].cpu().numpy().transpose(1, 2, 0)
            # init_img = init_img.clip(0, 255).astype(np.uint8)
            
            # ax_img.imshow(init_img)
            # ax_img.set_title("init obs")
            # ax_img.axis("off")


            #予測軌跡表示, ee, closed-forward, openforward, encoder-ouput
            ###########################################################################################
            ax_ee_forward = axes[0]
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
            ax_ee_forward.set_title("Endeffector Trajectory")
            ###########################################################################################


            #予測軌跡表示, ee, open-forward
            ###########################################################################################
            # ax_ee_forward = axes[0][2]
            # #gt
            # ax_ee_forward.plot(
            #     gt_locations[i, :, 0].cpu(),
            #     gt_locations[i, :, 1].cpu(),
            #     marker="o",
            #     markersize=2.5,
            #     linewidth=1,
            #     c="#3777FF",
            #     alpha=0.8,
            #     label="endeffector-ground-truth"
            # )
            # ax_ee_forward.text(
            #     gt_locations[i, 0, 0].cpu().item(),
            #     gt_locations[i, 0, 1].cpu().item(),
            #     "S",
            #     color="#3777FF",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            # ax_ee_forward.text(
            #     gt_locations[i, -1, 0].cpu().item(),
            #     gt_locations[i, -1, 1].cpu().item(),
            #     "G",
            #     color="#3777FF",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )


            # #ee, encoder
            # ax_ee_forward.plot(
            #     pred_enc_locs[i, :, 0].cpu(),
            #     pred_enc_locs[i, :, 1].cpu(),
            #     marker="o",
            #     markersize=2.5,
            #     linewidth=1,
            #     c="#008000",
            #     alpha=0.8,
            #     label="endeffector-encoder"
            # )
            # ax_ee_forward.text(
            #     pred_enc_locs[i, 0, 0].cpu().item(),
            #     pred_enc_locs[i, 0, 1].cpu().item(),
            #     "S",
            #     color="#008000",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            # ax_ee_forward.text(
            #     pred_enc_locs[i, -1, 0].cpu().item(),
            #     pred_enc_locs[i, -1, 1].cpu().item(),
            #     "G",
            #     color="#008000",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            
            
            # #ee, open_forward, 
            # ax_ee_forward.plot(
            #     pred_locs_opnfwd_encprb[i, :, 0].cpu(),
            #     pred_locs_opnfwd_encprb[i, :, 1].cpu(),
            #     marker="o",
            #     markersize=2.5,
            #     linewidth=1,
            #     c="#ff8c00",
            #     alpha=0.8,
            #     label="endeffector-open_pred"
            # )
            # ax_ee_forward.text(
            #     pred_locs_opnfwd_encprb[i, 0, 0].cpu().item(),
            #     pred_locs_opnfwd_encprb[i, 0, 1].cpu().item(),
            #     "S",
            #     color="#ff8c00",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            # ax_ee_forward.text(
            #     pred_locs_opnfwd_encprb[i, -1, 0].cpu().item(),
            #     pred_locs_opnfwd_encprb[i, -1, 1].cpu().item(),
            #     "G",
            #     color="#ff8c00",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            
            # ax_ee_forward.set_aspect("equal", adjustable="box")
            # ax_ee_forward.set_xlim(0.315, 0.715)
            # ax_ee_forward.set_ylim(-0.2, 0.2)
            # ax_ee_forward.set_xlabel("X (meters)")
            # ax_ee_forward.set_ylabel("Y (meters)")
            # ax_ee_forward.legend()
            # ax_ee_forward.set_title("dynamics model vs. groundtruth")
            ###########################################################################################


            #予測軌跡表示, ee, encoder
            ###########################################################################################
            # ax_ee_enc = axes[0][3]
            
            # #gt
            # ax_ee_enc.plot(
            #     gt_locations[i, :, 0].cpu(),
            #     gt_locations[i, :, 1].cpu(),
            #     marker="o",
            #     markersize=2.5,
            #     linewidth=1,
            #     c="#3777FF",
            #     alpha=0.8,
            #     label="endeffector-ground-truth"
            # )
            # ax_ee_enc.text(
            #     gt_locations[i, 0, 0].cpu().item(),
            #     gt_locations[i, 0, 1].cpu().item(),
            #     "S",
            #     color="#3777FF",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            # ax_ee_enc.text(
            #     gt_locations[i, -1, 0].cpu().item(),
            #     gt_locations[i, -1, 1].cpu().item(),
            #     "G",
            #     color="#3777FF",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            
            
            # #ee, encoder
            # ax_ee_enc.plot(
            #     pred_enc_locs[i, :, 0].cpu(),
            #     pred_enc_locs[i, :, 1].cpu(),
            #     marker="o",
            #     markersize=2.5,
            #     linewidth=1,
            #     c="#008000",
            #     alpha=0.8,
            #     label="endeffector-encoder"
            # )
            # ax_ee_enc.text(
            #     pred_enc_locs[i, 0, 0].cpu().item(),
            #     pred_enc_locs[i, 0, 1].cpu().item(),
            #     "S",
            #     color="#008000",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            # ax_ee_enc.text(
            #     pred_enc_locs[i, -1, 0].cpu().item(),
            #     pred_enc_locs[i, -1, 1].cpu().item(),
            #     "G",
            #     color="#008000",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            # ax_ee_enc.set_aspect("equal", adjustable="box")
            # ax_ee_enc.set_xlim(0.315, 0.715)
            # ax_ee_enc.set_ylim(-0.2, 0.2)
            # ax_ee_enc.set_xlabel("X (meters)")
            # ax_ee_enc.set_ylabel("Y (meters)")
            # ax_ee_enc.legend()
            # ax_ee_enc.set_title("encoder vs. groundtruth")
            ###########################################################################################



            #予測軌跡表示, bluebox, closed-forward
            ###########################################################################################
            ax_bluebox_forward = axes[1]
            
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


            #予測軌跡表示, bluebox, open-forward
            ###########################################################################################
            # ax_bluebox_forward = axes[1][2]
            
            # #box, gt
            # ax_bluebox_forward.plot(
            #     gt_bluebox_locations[i, :, 0].cpu(),
            #     gt_bluebox_locations[i, :, 1].cpu(),
            #     marker="o",
            #     markersize=2.5,
            #     linewidth=1,
            #     c="#3777FF",
            #     alpha=0.8,
            #     label="bluebox-ground-truth"
            # )
            # ax_bluebox_forward.text(
            #     gt_bluebox_locations[i, 0, 0].cpu().item(),
            #     gt_bluebox_locations[i, 0, 1].cpu().item(),
            #     "S",
            #     color="#3777FF",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            # ax_bluebox_forward.text(
            #     gt_bluebox_locations[i, -1, 0].cpu().item(),
            #     gt_bluebox_locations[i, -1, 1].cpu().item(),
            #     "G",
            #     color="#3777FF",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )

            # #box, encoder
            # ax_bluebox_forward.plot(
            #     pred_enc_bluebox_locs[i, :, 0].cpu(),
            #     pred_enc_bluebox_locs[i, :, 1].cpu(),
            #     marker="o",
            #     markersize=2.5,
            #     linewidth=1,
            #     c="#008000",
            #     alpha=0.8,
            #     label="bluebox-encoder-pred"
            # )
            # ax_bluebox_forward.text(
            #     pred_enc_bluebox_locs[i, 0, 0].cpu().item(),
            #     pred_enc_bluebox_locs[i, 0, 1].cpu().item(),
            #     "S",
            #     color="#008000",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            # ax_bluebox_forward.text(
            #     pred_enc_bluebox_locs[i, -1, 0].cpu().item(),
            #     pred_enc_bluebox_locs[i, -1, 1].cpu().item(),
            #     "G",
            #     color="#008000",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # ) 
            
            # #box, closed_forward, closed_prober
            # ax_bluebox_forward.plot(
            #     pred_bluebox_locs_opnfwd_encprb[i, :, 0].cpu(),
            #     pred_bluebox_locs_opnfwd_encprb[i, :, 1].cpu(),
            #     marker="o",
            #     markersize=2.5,
            #     linewidth=1,
            #     c="#ff8c00",
            #     alpha=0.8,
            #     label="bluebox-open-pred"
            # )
            # ax_bluebox_forward.text(
            #     pred_bluebox_locs_opnfwd_encprb[i, 0, 0].cpu().item(),
            #     pred_bluebox_locs_opnfwd_encprb[i, 0, 1].cpu().item(),
            #     "S",
            #     color="#ff8c00",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            # ax_bluebox_forward.text(
            #     pred_bluebox_locs_opnfwd_encprb[i, -1, 0].cpu().item(),
            #     pred_bluebox_locs_opnfwd_encprb[i, -1, 1].cpu().item(),
            #     "G",
            #     color="#ff8c00",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )


            # ax_bluebox_forward.set_title("Bluebox Trajectory")
            # ax_bluebox_forward.set_xlim(0.315, 0.715)
            # ax_bluebox_forward.set_ylim(-0.2, 0.2)
            # ax_bluebox_forward.set_aspect("equal")
            # ax_bluebox_forward.legend()
            ###########################################################################################


            #予測軌跡表示, bluebox, encoder
            ###########################################################################################
            # ax_bluebox_enc = axes[1][3]
            
            # #box, gt
            # ax_bluebox_enc.plot(
            #     gt_bluebox_locations[i, :, 0].cpu(),
            #     gt_bluebox_locations[i, :, 1].cpu(),
            #     marker="o",
            #     markersize=2.5,
            #     linewidth=1,
            #     c="#3777FF",
            #     alpha=0.8,
            #     label="bluebox-ground-truth"
            # )
            # ax_bluebox_enc.text(
            #     gt_bluebox_locations[i, 0, 0].cpu().item(),
            #     gt_bluebox_locations[i, 0, 1].cpu().item(),
            #     "S",
            #     color="#3777FF",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            # ax_bluebox_enc.text(
            #     gt_bluebox_locations[i, -1, 0].cpu().item(),
            #     gt_bluebox_locations[i, -1, 1].cpu().item(),
            #     "G",
            #     color="#3777FF",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            

            # #box, encoder
            # ax_bluebox_enc.plot(
            #     pred_enc_bluebox_locs[i, :, 0].cpu(),
            #     pred_enc_bluebox_locs[i, :, 1].cpu(),
            #     marker="o",
            #     markersize=2.5,
            #     linewidth=1,
            #     c="#008000",
            #     alpha=0.8,
            #     label="bluebox-encoder-pred"
            # )
            # ax_bluebox_enc.text(
            #     pred_enc_bluebox_locs[i, 0, 0].cpu().item(),
            #     pred_enc_bluebox_locs[i, 0, 1].cpu().item(),
            #     "S",
            #     color="#008000",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )
            # ax_bluebox_enc.text(
            #     pred_enc_bluebox_locs[i, -1, 0].cpu().item(),
            #     pred_enc_bluebox_locs[i, -1, 1].cpu().item(),
            #     "G",
            #     color="#008000",
            #     fontsize=12,
            #     ha="center",
            #     va="center",
            # )

            # ax_bluebox_enc.set_title("Bluebox Trajectory")
            # ax_bluebox_enc.set_xlim(0.315, 0.715)
            # ax_bluebox_enc.set_ylim(-0.2, 0.2)
            # ax_bluebox_enc.set_aspect("equal")
            # ax_bluebox_enc.legend()
            ###########################################################################################
            
            if not notebook:
                Logger.run().log_figure(fig, f"{name_prefix}-prober_predictions_by_encprober_{i}", dir_name = 'prober_prediction_by_encprober_FOR_POSTER')
                # Logger.run().log_video(ft_maps_gif_path, f"{name_prefix}-featuremap_{i}")

                plt.close(fig)
            else:
                plt.show()
                
            plt.close('all')

        # ===== 末尾: 後始末（元の del 群は維持） =====
        try:
            del pred_locs_clsfwd_encprb, pred_locs_opnfwd_encprb
        except Exception:
            pass
        try:
            del pred_bluebox_locs_clsfwd_encprb, pred_bluebox_locs_opnfwd_encprb
        except Exception:
            pass
        try:
            del pred_enc_locs, pred_enc_bluebox_locs
        except Exception:
            pass

        try:
            del pred_encs, pred_encs_open, encoder_encs
        except Exception:
            pass

        try:
            del gt_locations, gt_bluebox_locations
        except Exception:
            pass
        try:
            del img, init_img
        except Exception:
            pass

        try:
            del pred_output, pred_output_open, enc_output
        except Exception:
            pass
        try:
            del states, actions, optional_fields
        except Exception:
            pass

        gc.collect()
        torch.cuda.empty_cache()





    @torch.no_grad()
    def log_latent_forward_mse(
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
        vis_encoder_featuremap: bool = True,
    ):
        """
        latent 上での一貫性を測る評価:

        - closed vs open
        - open   vs encoder(Z)
        - closed vs encoder(Z)

        を MSE で計算し，

        - 全時刻・全バッチ平均のスカラー値
        - 時刻ごとの MSE 推移の line plot

        を Logger に記録する。
        """

        device = self.device

        # ===== 1. バッチをデバイスに載せる & optional fields =====
        states = batch.states.to(device).transpose(0, 1)   # [T, B, ...]
        actions = batch.actions.to(device).transpose(0, 1) # [T, B, ...]
        optional_fields = get_optional_fields(batch, device=states.device)

        # ===== 2. closed-forward 出力列 =====
        closed_res = jepa.forward_posterior(
            states, actions, **optional_fields
        )
        closed_pred = closed_res.pred_output       # ForwardResult.pred_output
        enc_output  = closed_res.backbone_output   # ForwardResult.backbone_output

        encoder_encs = enc_output.obs_component    # [T, B, C, H, W] 想定
        if closed_pred.obs_component is not None:
            closed_encs = closed_pred.obs_component
        else:
            closed_encs = closed_pred.predictions  # fallback

        # ---- まず encoder / closed を CPU に退避して GPU を空ける ----
        encoder_encs_cpu = encoder_encs.detach().float().cpu()
        closed_encs_cpu  = closed_encs.detach().float().cpu()

        # GPU 上の参照を削除
        del encoder_encs, closed_encs, enc_output, closed_pred, closed_res
        # states / actions は open-forward でも使うので残す
        torch.cuda.empty_cache()

        # ===== 3. open-forward 出力列 =====
        open_res = jepa.forward_open(
            states, actions, **optional_fields
        )
        open_pred = open_res.pred_output

        if open_pred.obs_component is not None:
            open_encs = open_pred.obs_component
        else:
            open_encs = open_pred.predictions

        # open もすぐ CPU へ
        open_encs_cpu = open_encs.detach().float().cpu()

        # GPU 上の参照を削除
        del open_encs, open_pred, open_res
        del states, actions, optional_fields
        torch.cuda.empty_cache()

        # ここから先は encoder_encs_cpu / closed_encs_cpu / open_encs_cpu だけを使う（全部 CPU）✨
        #   encoder_encs_cpu: [T, B, ...]  (CPU)
        #   closed_encs_cpu : [T, B, ...]  (CPU)
        #   open_encs_cpu   : [T, B, ...]  (CPU)

        # ===== 4. helper: 全体平均 MSE / 時刻ごとの MSE =====
        def mse_all(a: torch.Tensor, b: torch.Tensor) -> float:
            # 全次元に対して平均を取ったスカラー
            return F.mse_loss(a, b).item()

        def mse_per_timestep(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # a,b: [T, B, ...] (CPU 上)
            diff = F.mse_loss(a, b, reduction="none")    # [T, B, ...]
            # 時刻以外の次元で平均 -> [T]
            reduce_dims = tuple(range(1, diff.ndim))
            return diff.mean(dim=reduce_dims).detach().cpu()

        # ===== 5. スカラー MSE を計算 =====
        mse_closed_open = mse_all(closed_encs_cpu, open_encs_cpu)
        mse_open_enc    = mse_all(open_encs_cpu,  encoder_encs_cpu)
        mse_closed_enc  = mse_all(closed_encs_cpu, encoder_encs_cpu)

        # ===== 6. 時刻ごとの MSE 推移も計算 =====
        mse_t_closed_open = mse_per_timestep(closed_encs_cpu, open_encs_cpu)    # [T]
        mse_t_open_enc    = mse_per_timestep(open_encs_cpu,  encoder_encs_cpu)  # [T]
        mse_t_closed_enc  = mse_per_timestep(closed_encs_cpu, encoder_encs_cpu) # [T]

        # ===== 7. ログ: スカラー値 =====
        log_dict = {
            f"{name_prefix}/latent_mse_closed_vs_open": mse_closed_open,
            f"{name_prefix}/latent_mse_open_vs_enc"  : mse_open_enc,
            f"{name_prefix}/latent_mse_closed_vs_enc": mse_closed_enc,
        }
        Logger.run().log(log_dict)

        # ===== 8. ログ: 時刻ごとの line plot =====
        Logger.run().log_line_plot(
            data=[[int(t), float(mse_t_closed_open[t])] for t in range(mse_t_closed_open.shape[0])],
            plot_name=f"{name_prefix}_latent_mse_closed_vs_open_per_t"
        )
        Logger.run().log_line_plot(
            data=[[int(t), float(mse_t_open_enc[t])] for t in range(mse_t_open_enc.shape[0])],
            plot_name=f"{name_prefix}_latent_mse_open_vs_enc_per_t"
        )
        Logger.run().log_line_plot(
            data=[[int(t), float(mse_t_closed_enc[t])] for t in range(mse_t_closed_enc.shape[0])],
            plot_name=f"{name_prefix}_latent_mse_closed_vs_enc_per_t"
        )

        # コンソールにも一応出しておくとデバッグしやすいかも
        print(
            f"[LATENT MSE] {name_prefix} | "
            f"closed-open={mse_closed_open:.6e}, "
            f"open-Z={mse_open_enc:.6e}, "
            f"closed-Z={mse_closed_enc:.6e}"
        )

        # ===== ローカルファイルにも保存する =====
        run = Logger.run()
        from pathlib import Path

        if run.output_path is not None:
            out_dir = Path(run.output_path) / "latent_mse"
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{name_prefix}_latent_mse.txt"
            with open(out_path, "w") as f:
                f.write(f"[LATENT MSE] {name_prefix}\n")
                f.write(f"closed-open={mse_closed_open:.6e}\n")
                f.write(f"open-Z    ={mse_open_enc:.6e}\n")
                f.write(f"closed-Z  ={mse_closed_enc:.6e}\n\n")

                f.write("=== timestep MSE ===\n")
                f.write("closed-open: " + ", ".join(f"{v:.6e}" for v in mse_t_closed_open.tolist()) + "\n")
                f.write("open-enc   : " + ", ".join(f"{v:.6e}" for v in mse_t_open_enc.tolist()) + "\n")
                f.write("closed-enc : " + ", ".join(f"{v:.6e}" for v in mse_t_closed_enc.tolist()) + "\n")

            print(f"[LATENT MSE] saved → {out_path.resolve()}")

        # ===== CPU テンソルも掃除（お好みで） =====
        del encoder_encs_cpu, closed_encs_cpu, open_encs_cpu
        del mse_t_closed_open, mse_t_open_enc, mse_t_closed_enc
        gc.collect()

        return {
            "closed_vs_open": mse_closed_open,
            "open_vs_z": mse_open_enc,
            "closed_vs_z": mse_closed_enc,
        }