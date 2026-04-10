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
from pldm.data.enums import ProbingDatasets, Datasets, DatasetType
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
from typing import Optional, List, Tuple

import hashlib
from torch.utils.data import Subset, DataLoader

from datetime import datetime
import os
import imageio

import matplotlib as mpl
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import gc


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import gc


import matplotlib.pyplot as plt
from tqdm import tqdm

# === 可視化入力 btc の作成（vis_sample_length に応じて長いTを生成） ===
from torch.utils.data import DataLoader
from pldm.data.utils import NormalizedDataLoader
from pldm_envs.franka.franka_dataset import FrankaDataset
import dataclasses

from pldm_envs.franka.franka_dataset import FrankaEpisodeDataset

from pldm_envs.franka.envs import FrankaSimEnv
from pldm_envs.franka.evaluation.envs_generator import FrankaEnvsGenerator

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
    visualize_dynamics: bool = True
    vis_long_horizon_eval: bool = False
    vis_long_horizon_train: bool = False
    load_prober: bool = False
    train_prober: bool = True
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
    
    vis_sample_length: int = 50
    
    model_path: str = ""
    max_dq: float = 1000 
    camera_name: str = ""
    
    


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
        train_ds: Optional[Datasets] = None,
        normalizer = None, 
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        # print("[pldm/probing/evaluator.py] self.config.model_path:", self.config.model_path)
        # print("[pldm/probing/evaluator.py] self.config.max_dq:", self.config.max_dq)
        # print("[pldm/probing/evaluator.py] self.config.camera_name:", self.config.camera_name)
        
        
        self.model = model
        self.quick_debug = quick_debug

        self.ds = probing_datasets.ds
        self.val_ds = probing_datasets.val_ds
        self.extra_val_ds = probing_datasets.extra_datasets
        
        self.train_ds = train_ds

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
            
        self.normalizer = normalizer

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
            
        if (not config.load_prober) and (not config.train_prober):
            print("[ProbingEvaluator] load_prober=False & train_prober=False: skip prober entirely.")
            return {}
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
        if not config.train_prober:
            print("[ProbingEvaluator] train_prober=False: skip training probers.")
            return {} 

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
        print("probe evaluating start!")
        for prefix, val_ds in val_datasets.items():
            self.evaluate_pred_prober(
                probers=probers,
                epoch=epoch,
                val_ds=val_ds,
                train_ds=self.train_ds,
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
        train_ds: DatasetType = None,
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
        
        
        if visualize:

            # # === 可視化入力btc の作成 === 
            # from torch.utils.data import Subset, DataLoader

            # def _unwrap_loader(x):
            #     return getattr(x, "dataloader", x)  

            # inner_loader = _unwrap_loader(val_ds)
            # root_ds = inner_loader.dataset          
            # B = inner_loader.batch_size
            # N = len(root_ds)


            # head_indices = np.arange(0, (N // B) * B, B, dtype=np.int64)
            # os.makedirs("vis_debug", exist_ok=True)
            # np.save("vis_debug/vis_indices_heads.npy", head_indices)
            # print("[VIS] head_indices sha:", hashlib.sha256(head_indices.tobytes()).hexdigest()[:16])


            # vis_subset = Subset(root_ds, head_indices.tolist())
            # vis_loader_raw = DataLoader(
            #     vis_subset,
            #     batch_size=B,
            #     shuffle=False,     
            #     num_workers=0,
            #     drop_last=False,    
            # )
            # vis_loader_raw.config = inner_loader.config
            # # 既存と同じ正規化を適用（NormalizedDataLoaderでラップ）
            # from pldm.data.utils import NormalizedDataLoader
            # vis_loader = NormalizedDataLoader(vis_loader_raw, val_ds.normalizer)
            

            # vis_batch_idx = 0  
            # itr = iter(vis_loader)
            # for _ in range(vis_batch_idx + 1):
            #     btc = next(itr)




            ##追加
            def _unwrap_loader(x):
                return getattr(x, "dataloader", x)

            inner_loader = _unwrap_loader(val_ds)   # DataLoader or NormalizedDataLoader
            base_cfg = inner_loader.config          # FrankaDatasetのconfig相当（sample_length含む）

            vis_T = self.config.vis_sample_length
            if vis_T is None:
                vis_T = getattr(base_cfg, "sample_length", None)
            print("[DBG][pldm/probing/evaluator.py] vis_T:", vis_T)
            
            vis_cfg = dataclasses.replace(
                base_cfg,
                sample_length=vis_T,
                path=self.config.val_path if self.config.val_path is not None else base_cfg.path,
                images_path=self.config.val_images_path if self.config.val_images_path is not None else base_cfg.images_path,
            )

            vis_root_ds = FrankaDataset(vis_cfg)

            # DataLoader（batch_sizeはbase_cfgに入っている想定）
            vis_loader_raw = DataLoader(
                vis_root_ds,
                batch_size=base_cfg.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            vis_loader_raw.config = base_cfg

            # 正規化は既存の val_ds.normalizer を使う（再推定しない）
            vis_loader = NormalizedDataLoader(vis_loader_raw, val_ds.normalizer)

            vis_batch_idx = getattr(self.config, "vis_batch_idx", 0)
            itr = iter(vis_loader)
            for _ in range(vis_batch_idx + 1):
                btc = next(itr)
            print("[DBG][pldm/probing/evaluator.py] btc.shape:", btc.states.shape)
            print(f"[VIS] visualize sample_length={vis_T}, batch_idx={vis_batch_idx}")


            inner_train_loader = _unwrap_loader(train_ds)   # DataLoader or NormalizedDataLoader
            print("[DBG][pldm/probing/evaluator.py] train_ds:", train_ds)
            base_train_cfg = inner_train_loader.config          # FrankaDatasetのconfig相当（sample_length含む）
            vis_train_cfg = dataclasses.replace(
                base_train_cfg,
                sample_length=vis_T,
                path=base_train_cfg.path,
                images_path=base_train_cfg.images_path,
            )
            
            vis_root_train_ds = FrankaDataset(vis_train_cfg)
            vis_loader_raw_train = DataLoader(
                vis_root_train_ds,
                batch_size=base_train_cfg.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            vis_loader_raw_train.config = base_train_cfg
            vis_train_loader = NormalizedDataLoader(vis_loader_raw_train, train_ds.normalizer)
            vis_train_batch_idx = getattr(self.config, "vis_train_batch_idx", 0)
            itr = iter(vis_train_loader)
            for _ in range(vis_train_batch_idx + 1):
                train_btc = next(itr)
            
            
            ###
            
            





            # ===== DEBUG: btc.states 上位5本を mp4 として保存 =====


            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_root = os.path.join("debug_btc_states_videos", f"{ts}")
            os.makedirs(save_root, exist_ok=True)

            st = btc.states.detach().cpu()  # まずCPUへ

            # btc.states が [B,T,...] の場合が多いので [T,B,...] に揃える
            # もし既に [T,B,...] ならこの変換は不要だが、形で判定して安全に処理する
            if st.dim() >= 2:
                # 典型: [B,T,C,H,W] -> [T,B,C,H,W]
                # 典型: [T,B,C,H,W] の可能性もあるので、Tっぽい方を見て判断（ここは簡易）
                # あなたの可視化コードと合わせて、まずは "transpose(0,1)" を標準にするのが無難
                if st.shape[0] == btc.states.shape[0]:  # 元のテンソルと同じならそのまま（ダミー条件）
                    pass
            # ここではあなたの他の関数に合わせて「transpose(0,1)でT先頭」を採用
            # （もし既に [T,B,...] なら結果が逆になるので、下のログで形を確認して必要なら外してOK）
            print("[DBG][pldm/probing/evaluator.py] st.shape:", st.shape)
            st = st.transpose(0, 1).contiguous()  # [T,B,...] を期待

            print("[DBG] btc.states (as T,B,...) shape:", tuple(st.shape))
            print("[DBG] btc.states raw  min/max/mean/std:",
                st.min().item(), st.max().item(), st.mean().item(), st.std().item())

            # 画像観測のみ対応: [T,B,C,H,W]
            if st.dim() == 5:
                # 逆正規化（できれば）
                st_vis = st
                try:
                    st_vis = val_ds.normalizer.unnormalize_state(st_vis)
                    # もし normalize_mode が minmax 等で外に出る場合もありうるのでここではそのまま
                except Exception as e:
                    print("[DBG] unnormalize_state failed:", repr(e))
                    st_vis = st

                print("[DBG] btc.states unnorm min/max/mean/std:",
                    st_vis.min().item(), st_vis.max().item(), st_vis.mean().item(), st_vis.std().item())

                T_, B_, C_, H_, W_ = st_vis.shape
                n_save = min(5, B_)
                fps = 10

                for bi in range(n_save):
                    out_path = os.path.join(save_root, f"traj{bi:03d}.mp4")

                    with imageio.get_writer(
                        out_path,
                        fps=fps,
                        codec="libx264",
                        quality=8,
                        pixelformat="yuv420p",
                    ) as writer:
                        for t in range(T_):
                            img = st_vis[t, bi].numpy()              # (C,H,W)
                            img = np.transpose(img, (1, 2, 0))       # (H,W,C)

                            # 0-1 / 0-255 両対応
                            if img.max() <= 1.0:
                                img = img * 255.0
                            img = np.clip(img, 0, 255).astype(np.uint8)

                            writer.append_data(img)

                    print(f"[DBG] saved video: {out_path}")

            else:
                print("[DBG] btc.states are not image-shaped. skip video saving.")
            # ==================================================



            
            if self.config.visualize_probing:

                #use_prober
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

                # #use prober
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
                
                
                #use prober
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



            if self.config.visualize_dynamics:
                
                print('[DBG][pldm/probing/evaluator.py] vis_long_horizon_eval:', self.config.vis_long_horizon_eval)
                if self.config.vis_long_horizon_eval: 
                    # 疑似ロング軌道としてPCA可視化 ###
                    #episodeを抜き取るデータセットを作成
                    vis_root_ds_per_epi = FrankaEpisodeDataset(
                        vis_cfg,
                        pick_mode="first",   # "middle"/"random"/"last" もOK
                    )

                    vis_loader_raw_per_epi = DataLoader(
                        vis_root_ds_per_epi,
                        batch_size=5,        # ← ここはお好み
                        shuffle=False,       # episode順固定が良いならFalse
                        num_workers=0,
                        drop_last=False,
                    )
                    vis_loader_raw_per_epi.config = base_cfg

                    vis_loader_per_epi = NormalizedDataLoader(vis_loader_raw_per_epi, val_ds.normalizer)
                    
                    self.plot_pca_long_horizon(
                        loader=vis_loader_per_epi,
                        jepa=model,
                        name_prefix=f"{plot_prefix}_val_full",
                        pool="flat",
                        k=3,
                        take_per_batch=999999,  # ★ batch内(B=8)を全部拾う
                        pick_mode="first",      # ここは関係薄い（take_per_batch>=Bなら使われない）
                        max_batches=None,       # ★ 全batch回す → 全episode拾う
                    )
                    #######
                print("[DBG][pldm/probing/evaluator.py] self.config.vis_long_horizon_train:", self.config.vis_long_horizon_train)
                if self.config.vis_long_horizon_train:
                    
                    vis_root_train_ds_per_epi = FrankaEpisodeDataset(
                        vis_train_cfg,
                        pick_mode="first",   # "middle"/"random"/"last" もOK
                    )
                    
                    vis_train_loader_raw_per_epi = DataLoader(
                        vis_root_train_ds_per_epi,
                        batch_size=5,        # ← ここはお好み
                        shuffle=False,       # episode順固定が良いならFalse
                        num_workers=0,
                        drop_last=False,
                    )
                    
                    vis_train_loader_raw_per_epi.config = base_train_cfg
                    vis_train_loader_per_epi = NormalizedDataLoader(vis_train_loader_raw_per_epi, train_ds.normalizer)
                    self.plot_pca_long_horizon(
                        loader=vis_train_loader_per_epi,
                        jepa=model,
                        name_prefix=f"{plot_prefix}_train_full",
                        pool="flat",
                        k=3,
                        take_per_batch=999999,  # ★ batch内(B=8)を全部拾う
                        pick_mode="first",      # ここは関係薄い（take_per_batch>=Bなら使われない）
                        max_batches=None,       # ★ 全batch回す → 全episode拾う
                        max_num_plot = 25,
                    ) 
                    
                self.pca_visual_encoder_bluebox(
                    btc,
                    model,
                    pool = "flat",
                )
                
                self.pca_visual_encoder_bluebox_3d(
                    btc,
                    model,
                    pool = "flat",
                )
                
                self.pca_visual_encoder_franka(
                    btc,
                    model,
                    pool = "flat",
                )
                
                self.pca_visual_encoder_franka_3d(
                    btc,
                    model, 
                    pool = "flat",
                )
                
                self.plot_pca_cartesian_action_rollout(
                    btc,
                    model,
                    pool = "flat",
                )
                  
                #dont use prober
                # Encoder の潜在分布が等方ガウスに近いか確認
                self.plot_encoder_latent_gaussianity(
                    btc,
                    model,
                    name_prefix=f"{plot_prefix}_val",
                )
                
                #dont use prober
                self.log_encoder_latent_variance(
                        btc,
                        model,
                        name_prefix = "",
                        max_points = 5000,
                    )
                #dont use prober
                self.plot_encoder_latent_tsne(
                        btc,
                        model,
                        name_prefix = "",
                        max_points = 5000,
                )

                
                #dont use prober
                self.plot_pca(
                    btc,
                    model,  
                    idxs=None if not quick_debug else list(range(5)),
                    pool = 'flat',   
                )
                
                #dont use prober
                self.plot_pca_open_closed(
                    btc,
                    model,
                    name_prefix=f"{plot_prefix}_val",
                    idxs=None if not quick_debug else list(range(5)),
                    pool = 'flat', 
                )
                
                #dont use prober
                self.plot_pca_encoder_open(
                    btc,
                    model,
                    name_prefix=f"{plot_prefix}_val",
                    idxs=None if not quick_debug else list(range(5)),
                    pool = 'flat',      
                )

                #train_ds
                self.plot_pca(
                    train_btc,
                    model,  
                    idxs=None if not quick_debug else list(range(5)),
                    pool = 'flat',   
                    is_train=True,
                ) 
                self.plot_pca_encoder_open(
                    train_btc,
                    model,
                    name_prefix=f"{plot_prefix}_val",
                    idxs=None if not quick_debug else list(range(5)),
                    pool = 'flat',   
                    is_train=True   
                )
                
                self.plot_pca_rgb(
                    btc,
                    model,
                    name_prefix=f"{plot_prefix}_val",
                    idxs=None if not quick_debug else list(range(5)),
                    is_train=False,
                    upsample=(224, 224),
                )

                
                
                
                

                #dont use prober
                metrics_latent = self.log_latent_forward_rmse(
                    batch=btc,
                    jepa=model,
                    normalizer=val_ds.normalizer,
                    name_prefix=plot_prefix,
                )
                
                # self.plot_pca_rgb_featuremaps


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
        if (not config.load_prober) and (not config.train_prober):
            print("[ProbingEvaluator] load_prober=False & train_prober=False: skip prober entirely.")
            return {}
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
            
            
            probers[probe_target] = prober.to(self.device)
        
        if config.load_prober:
            return probers
        
        if not config.train_prober:
            print("[ProbingEvaluator] train_prober=False: skip training probers.")
            return {} 
            

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
        pool: str = "gap",   # "gap" or "flat"
    ):
        """
        Encoder の潜在表現がどれくらい等方ガウスに近いかを可視化する。
        - pool="gap": feature map (C,H,W) を空間平均して (C) にしてから PCA
        - pool="flat": (C,H,W) をフラット化して (C*H*W) で PCA（従来）
        センタリングは sklearn PCA に任せる（明示的に引かない）。
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from pathlib import Path
        import gc
        import torch

        # ===== 1. Encoder 出力を取得 =====
        states = batch.states.to(self.device).transpose(0, 1)   # [T,B,...]
        actions = batch.actions.to(self.device).transpose(0, 1) # [T,B,...]
        optional_fields = get_optional_fields(batch, device=states.device)

        print("[DBG plot_encoder_latent_gaussianity] states:", states.shape)
        print("[DBG plot_encoder_latent_gaussianity] actions:", actions.shape)

        enc_out = jepa.forward_posterior(
            states, actions, encode_only=True, **optional_fields
        ).backbone_output

        encoder_encs = enc_out.obs_component
        print("[DBG plot_encoder_latent_gaussianity] encoder_encs:", encoder_encs.shape)

        # ===== 2. [N, K] を作る =====
        # encoder_encs: [T,B,C,H,W] or [T,B,D] etc.
        zs = []
        for x in encoder_encs:  # x: [B,...]
            if x.dim() == 4:
                # [B,C,H,W]
                if pool == "gap":
                    x_vec = x.mean(dim=(-2, -1))          # [B,C]
                elif pool == "flat":
                    x_vec = x.reshape(x.shape[0], -1)     # [B,C*H*W]
                else:
                    raise ValueError(f"pool must be 'gap' or 'flat', got {pool}")
            else:
                # [B,D] などはそのまま flatten
                x_vec = x.reshape(x.shape[0], -1)         # [B,K]
            zs.append(x_vec)

        z_all = torch.cat(zs, dim=0).detach().cpu()  # [N,K]
        N, K = z_all.shape

        # subsample
        if N > max_points:
            idx = torch.randperm(N)[:max_points]
            z_all = z_all[idx]
            N = max_points

        # ===== 3. stats（PCA前の実値スケール確認用）=====
        print("[LATENT] pool:", pool)
        print("[LATENT] z_all.mean:", z_all.mean().item())
        print("[LATENT] z_all.std :", z_all.std().item())
        print("[LATENT] z_all.min/max:", z_all.min().item(), z_all.max().item())
        print("[LATENT] has_nan:", torch.isnan(z_all).any().item())
        print("[LATENT] has_inf:", torch.isinf(z_all).any().item())

        # numpy (センタリングはしない。PCAが内部でcenterする)
        z_np = z_all.numpy()

        # ===== 4. PCA (最大4次元) =====
        pca = PCA(n_components=min(4, K))
        z_pca = pca.fit_transform(z_np)

        explained_var = pca.explained_variance_
        explained_ratio = pca.explained_variance_ratio_

        print("[PCA] explained_variance:", explained_var)
        print("[PCA] explained_variance_ratio:", explained_ratio)
        for i, (ev, r) in enumerate(zip(explained_var, explained_ratio), start=1):
            print(f"[PCA] PC{i}: variance={ev:.6f}, ratio={r:.4%}")

        # ===== 5. 共分散の固有値（等方性の目安）=====
        cov = np.cov(z_np.T)  # PCA前の空間で
        eigvals = np.linalg.eigvalsh(cov)
        iso_ratio = float(eigvals.max() / (eigvals.min() + 1e-12))
        print("[LATENT] eigvals min/max:", float(eigvals.min()), float(eigvals.max()))
        print(f"[LATENT] iso_ratio (max/min) = {iso_ratio:.3f}")

        # ===== 6. 保存先 =====
        run = Logger.run()
        run_dir = Path(run.output_path) if (run.output_path is not None) else Path(".")
        out_dir = run_dir / "latent_gaussianity"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ===== 7. プロット =====
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=200)

        axes[0].scatter(z_pca[:, 0], z_pca[:, 1], s=8, alpha=0.6)
        axes[0].set_xlabel("PC 1")
        axes[0].set_ylabel("PC 2")
        axes[0].set_title(f"{name_prefix} encoder latent ({pool}) PC1 vs PC2")

        if z_pca.shape[1] >= 4:
            axes[1].scatter(z_pca[:, 2], z_pca[:, 3], s=8, alpha=0.6)
            axes[1].set_xlabel("PC 3")
            axes[1].set_ylabel("PC 4")
            axes[1].set_title("PC3 vs PC4")
        else:
            axes[1].axis("off")

        plt.tight_layout()
        out_path = out_dir / f"{name_prefix}_encoder_latent_{pool}.png"
        plt.savefig(out_path)
        plt.close(fig)
        print(f"[LATENT] saved encoder latent plot -> {out_path.resolve()}")

        # ===== 8. Logger にも記録 =====
        try:
            run.log(
                {
                    f"{name_prefix}/latent_iso_ratio_{pool}": iso_ratio,
                    f"{name_prefix}/latent_eig_max_{pool}": float(eigvals.max()),
                    f"{name_prefix}/latent_eig_min_{pool}": float(eigvals.min()),
                    f"{name_prefix}/latent_std_{pool}": float(z_all.std().item()),
                }
            )
        except Exception as e:
            print("[LATENT] Logger logging skipped:", e)

        # ===== 9. txt 保存 =====
        log_path = out_dir / f"{name_prefix}_stats_{pool}.txt"
        with open(log_path, "w") as f:
            f.write(f"=== z_all stats (pool={pool}) ===\n")
            f.write(f"mean: {z_all.mean().item()}\n")
            f.write(f"std: {z_all.std().item()}\n")
            f.write(f"min: {z_all.min().item()}\n")
            f.write(f"max: {z_all.max().item()}\n")
            f.write(f"has_nan: {torch.isnan(z_all).any().item()}\n")
            f.write(f"has_inf: {torch.isinf(z_all).any().item()}\n\n")

            f.write("=== PCA explained variance ===\n")
            for i, (ev, r) in enumerate(zip(explained_var, explained_ratio), start=1):
                f.write(f"PC{i}: variance={ev:.8f}, ratio={r:.6%}\n")
            f.write("\n")

            f.write("=== covariance eigenvalues ===\n")
            f.write(f"eigvals: {eigvals.tolist()}\n")
            f.write(f"iso_ratio (max/min): {iso_ratio}\n")

        print(f"[LATENT] saved stats -> {log_path.resolve()}")

        # ===== cleanup =====
        try:
            del z_all, z_np, z_pca, cov, eigvals, encoder_encs, enc_out, states, actions, optional_fields
        except Exception:
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


    #K (=CHW)個のサンプルのN (=T * B)方向の分散のヒストグラムをとっている
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
        is_train: bool = False,   
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
        print("[pldm/probing/evaluator.py]states.shape:", states.shape)
        print("[pldm/probing/evaluator.py]actions.shape:", actions.shape)
        print("[pldm/probing/evaluator.py]pred_output.shape:", pred_output.predictions.shape)
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
        color_u = "black"       # Encoder
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
                ax2d.text(u2d[0,0], u2d[0,1], "S", fontsize=13, ha="center", va="center", color=color_u)
                ax2d.text(u2d[-1,0], u2d[-1,1], "G", fontsize=13, ha="center", va="center", color=color_u)

                ax2d.scatter(v2d[0,0], v2d[0,1], s=12, c=color_v, label="Closed")
                ax2d.text(v2d[0,0], v2d[0,1], "S", fontsize=13, ha="center", va="center", color=color_v)
                ax2d.text(v2d[-1,0], v2d[-1,1], "G", fontsize=13, ha="center", va="center", color=color_v)

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
                
                #2dは可視化しない
                # if not notebook:
                #     if not is_train:
                #         Logger.run().log_figure(fig2d, f"{name_prefix}-pca-2d-i{i}", dir_name="pca_val/pca2d_pertraj")
                #     else:
                #         Logger.run().log_figure(fig2d, f"{name_prefix}-pca-2d-i{i}", dir_name="pca_train/pca2d_pertraj")
                #     plt.close(fig2d)
                # else:
                #     plt.show()
                plt.close(fig2d)

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
                    if not is_train:
                        Logger.run().log_figure(fig3d, f"{name_prefix}-pca-3d-i{i}", dir_name="pca_val/pca3d_pertraj")
                    else:
                        Logger.run().log_figure(fig3d, f"{name_prefix}-pca-3d-i{i}", dir_name="pca_train/pca3d_pertraj")
                    plt.close(fig3d)
                else:
                    plt.show()
                    plt.close(fig3d)


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
                if not is_train:
                    Logger.run().log_figure(figc, f"{name_prefix}-pca-timewise-cosine", dir_name="pca_val/pca_cos")
                else:
                    Logger.run().log_figure(figc, f"{name_prefix}-pca-timewise-cosine", dir_name="pca_train/pca_cos")
                plt.close(figc)
            else:
                plt.show()
            plt.close(fig2d)
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
        is_train: bool = False,
    ):
        """
        open-forward 列を基準に PCA 軸を学習し、
        - open-forward 列（基準）
        - closed-forward 列
        を「同じ PCA 空間」に射影して 2D/3D で可視化する。

        encoder ではなく open を anchor にしている点だけが元の plot_pca と異なる。
        """


        device = self.device
        states = batch.states.to(device).transpose(0, 1)  # [T,B,...]
        actions = batch.actions.to(device).transpose(0, 1)
        optional_fields = get_optional_fields(batch, device=states.device)
        print("[DBG][pldm/probing/evaluator.py] states range:", states.min().item(), states.max().item())
        # print("[DBG][pldm/probing/evaluator.py] normalizer mode:", self.normalizer.normalize_mode)


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

                #2dは可視化しない
                if not notebook:
                    # if not is_train:
                    #     Logger.run().log_figure(
                    #         fig2d,
                    #         f"{name_prefix}-pca_openanchor-2d-i{i}",
                    #         dir_name="pca_open_closed_val/pca2d_pertraj",
                    #     )
                    # else:
                    #     Logger.run().log_figure(
                    #         fig2d,
                    #         f"{name_prefix}-pca_openanchor-2d-i{i}",
                    #         dir_name="pca_open_closed_train/pca2d_pertraj",
                    #     )
                    plt.close(fig2d)
                else:
                    plt.show()
                    plt.close(fig2d)
                plt.close(fig2d)

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
                    if not is_train:
                        Logger.run().log_figure(
                            fig3d,
                            f"{name_prefix}-pca_openanchor-3d-i{i}",
                            dir_name="pca_open_closed_val/pca3d_pertraj",
                        )
                    else:
                        Logger.run().log_figure(
                            fig3d,
                            f"{name_prefix}-pca_openanchor-3d-i{i}",
                            dir_name="pca_open_close_train/pca3d_pertraj",
                        )
                    plt.close(fig3d)
                else:
                    plt.show()
                    plt.close(fig3d)

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
                if not is_train:
                    Logger.run().log_figure(
                        figc,
                        f"{name_prefix}-pca_openanchor-timewise-cosine",
                        dir_name="pca_open_closed_val/pca_cos",
                    )
                else:
                    Logger.run().log_figure(
                        figc,
                        f"{name_prefix}-pca_openanchor-timewise-cosine",
                        dir_name="pca_open_closed_train/pca_cos",
                    )
                plt.close(figc)
            else:
                plt.show()
                plt.close(figc)
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


    @torch.no_grad()
    def plot_pca_encoder_open(
        self,
        batch,
        jepa: "JEPA",
        name_prefix: str = "",
        idxs=None,
        notebook: bool = False,
        pool: str = "gap",      # "gap"/"flat"
        k: int = 3,             # PCA 次元
        is_train: bool = False,
    ):
        """
        encoder 出力を基準に PCA 軸を学習し、
        - encoder 潜在列（基準）
        - open-forward 潜在列
        を同一 PCA 空間に射影して 2D/3D で可視化する。
        """


        device = self.device
        states  = batch.states.to(device).transpose(0, 1)    # [T,B,...]
        actions = batch.actions.to(device).transpose(0, 1)   # [T,B,...]
        optional_fields = get_optional_fields(batch, device=states.device)

        # ===== encoder 潜在列（encode_only）=====
        enc_out = jepa.forward_posterior(
            states, actions, encode_only=True, **optional_fields
        ).backbone_output
        encoder_lat_seq = enc_out.obs_component  # [T,B,C,H,W] or [T,B,D]

        # ===== open-forward 潜在列 =====
        open_res = jepa.forward_open(states, actions, **optional_fields)
        pred_open = open_res.pred_output
        if getattr(pred_open, "obs_component", None) is not None:
            open_lat_seq = pred_open.obs_component  # [T+1,B,...]
        else:
            open_lat_seq = pred_open.predictions    # [T+1,B,D] など

        # open は [0]=z0(=enc0), [t]=z_hat_t なので encoder と長さ合わせ
        # encoder_lat_seq が [T,B,...] なら open_lat_seq[:T] で揃える
        open_lat_seq = open_lat_seq[: encoder_lat_seq.shape[0]]

        T_ref, B_ref = states.shape[0], states.shape[1]

        def _to_BTD(x, pool="flat"):
            """[T,B,...] or [B,T,...] -> [B,T,D]"""
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
                    x = x.mean(dim=(-2, -1)).contiguous()  # -> [B,T,C]
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
            return x.reshape(B, T, -1).contiguous()

        encoder_lat_seq = _to_BTD(encoder_lat_seq, pool=pool)
        open_lat_seq    = _to_BTD(open_lat_seq,    pool=pool)

        B, T, De = encoder_lat_seq.shape
        _, _, Do = open_lat_seq.shape
        if idxs is None:
            idxs = list(range(B))

        # ===== PCA を encoder で学習 =====
        Z_enc = encoder_lat_seq.reshape(B * T, De).detach().cpu().numpy()
        pca = PCA(n_components=min(k, De))
        U_enc = pca.fit_transform(Z_enc)  # [B*T, k_eff]
        expl = pca.explained_variance_ratio_
        k_eff = U_enc.shape[1]

        # ===== open を encoder 次元へ合わせて同一基底に射影 =====
        Z_open = open_lat_seq.reshape(B * T, Do).detach().cpu().numpy()
        if Do != De:
            A, *_ = np.linalg.lstsq(Z_open, Z_enc, rcond=None)
            Z_open_in_enc = Z_open @ A
        else:
            Z_open_in_enc = Z_open

        V_open = pca.transform(Z_open_in_enc)  # [B*T, k_eff]

        # [B,T,k] に戻す
        U_bt = U_enc.reshape(B, T, k_eff)
        V_bt = V_open.reshape(B, T, k_eff)

        # ===== 2D 可視化 =====
        if k_eff >= 2:
            color_enc  = "black"
            color_open = "darkorange"
            lim = float(max(abs(U_bt[:, :, :2]).max(), abs(V_bt[:, :, :2]).max()))

            for i in idxs:
                fig2d, ax2d = plt.subplots(1, 1, figsize=(6, 6), dpi=140)
                ue = U_bt[i, :, :2]
                vo = V_bt[i, :, :2]

                for t in range(T - 1):
                    ax2d.plot(ue[t:t+2, 0], ue[t:t+2, 1], color=color_enc,  alpha=0.95)
                    ax2d.plot(vo[t:t+2, 0], vo[t:t+2, 1], color=color_open, alpha=0.95)

                ax2d.scatter(ue[0,0], ue[0,1], s=12, c=color_enc)
                ax2d.scatter(vo[0,0], vo[0,1], s=12, c=color_open)
                ax2d.text(ue[0,0], ue[0,1], "S", color=color_enc,  fontsize=9)
                ax2d.text(ue[-1,0], ue[-1,1], "G", color=color_enc, fontsize=9)
                ax2d.text(vo[0,0], vo[0,1], "S", color=color_open, fontsize=9)
                ax2d.text(vo[-1,0], vo[-1,1], "G", color=color_open, fontsize=9)

                ax2d.set_xlim(-lim, lim); ax2d.set_ylim(-lim, lim)
                ax2d.set_aspect("equal", adjustable="box")
                ax2d.set_xlabel("PC1"); ax2d.set_ylabel("PC2")
                ax2d.set_title(
                    f"{name_prefix} | idx={i} | PCA(encoder anchor) top-{k_eff}: "
                    + ", ".join(f"{v:.2f}" for v in expl[:k_eff])
                )
                ax2d.legend(handles=[
                    Line2D([0],[0], color=color_enc,  lw=2, label="Encoder"),
                    Line2D([0],[0], color=color_open, lw=2, label="Open"),
                ], loc="best", frameon=True)

                #2d は可視化しない
                # if not notebook:
                #     if not is_train:
                #         Logger.run().log_figure(fig2d, f"{name_prefix}-pca-enc-open-2d-i{i}", dir_name="pca_encoder_open_val/pca2d")
                #     else:
                #         Logger.run().log_figure(fig2d, f"{name_prefix}-pca-enc-open-2d-i{i}", dir_name="pca_encoder_open_train/pca2d")
                #     plt.close(fig2d)
                # else:
                #     plt.show()
                plt.close(fig2d)

        # ===== 3D 可視化 =====
        if k_eff >= 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa

            color_enc  = "black"
            color_open = "darkorange"

            for i in idxs:
                fig3d = plt.figure(figsize=(8, 8), dpi=140)
                ax3d = fig3d.add_subplot(111, projection="3d")

                ue = U_bt[i, :, :3]
                vo = V_bt[i, :, :3]

                for t in range(T - 1):
                    ax3d.plot(ue[t:t+2, 0], ue[t:t+2, 1], ue[t:t+2, 2], color=color_enc,  alpha=0.95)
                    ax3d.plot(vo[t:t+2, 0], vo[t:t+2, 1], vo[t:t+2, 2], color=color_open, alpha=0.95)

                s_size = 24
                ax3d.scatter(ue[0,0],  ue[0,1],  ue[0,2],  s=s_size, c=color_enc,  depthshade=False)
                ax3d.scatter(ue[-1,0], ue[-1,1], ue[-1,2], s=s_size, c=color_enc,  depthshade=False)
                ax3d.text(ue[0,0],  ue[0,1],  ue[0,2],  "S", color=color_enc,  fontsize=9)
                ax3d.text(ue[-1,0], ue[-1,1], ue[-1,2], "G", color=color_enc,  fontsize=9)

                ax3d.scatter(vo[0,0],  vo[0,1],  vo[0,2],  s=s_size, c=color_open, depthshade=False)
                ax3d.scatter(vo[-1,0], vo[-1,1], vo[-1,2], s=s_size, c=color_open, depthshade=False)
                ax3d.text(vo[0,0],  vo[0,1],  vo[0,2],  "S", color=color_open, fontsize=9)
                ax3d.text(vo[-1,0], vo[-1,1], vo[-1,2], "G", color=color_open, fontsize=9)

                ax3d.set_xlabel("PC1"); ax3d.set_ylabel("PC2"); ax3d.set_zlabel("PC3")
                ax3d.set_title(
                    f"PCA(3D, encoder anchor) — {name_prefix} | var exp: "
                    + ", ".join(f"{v:.2f}" for v in expl[:3])
                )
                ax3d.legend(handles=[
                    Line2D([0],[0], color=color_enc,  lw=2, label="Encoder"),
                    Line2D([0],[0], color=color_open, lw=2, label="Open"),
                ], loc="upper left", frameon=True)

                if not notebook:
                    if not is_train:
                        Logger.run().log_figure(fig3d, f"{name_prefix}-pca-enc-open-3d-i{i}", dir_name="pca_encoder_open_val/pca3d")
                    else:
                        Logger.run().log_figure(fig3d, f"{name_prefix}-pca-enc-open-3d-i{i}", dir_name="pca_encoder_open_train/pca3d")
                        
                else:
                    plt.show()
                plt.close(fig3d)

        # ===== 後始末 =====
        plt.close("all")
        try:
            del U_bt, V_bt, Z_enc, Z_open, Z_open_in_enc
        except Exception:
            pass
        try:
            del encoder_lat_seq, open_lat_seq, enc_out, open_res, pred_open, states, actions, optional_fields
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()




    @torch.no_grad()
    def log_latent_forward_rmse(
        self,
        batch,
        jepa: JEPA,
        normalizer: Normalizer = None,
        name_prefix: str = "",
        idxs: Optional[List[int]] = None,   # Noneなら全traj
    ):
        """
        latent 上での一貫性:
        - closed vs open
        - open   vs encoder(Z)
        - closed vs encoder(Z)
        を trajectory（batch内サンプル）ごとに MSE 計算し、
        1 traj = 1 txt で保存する。
        """

        device = self.device

        # ===== 1. deviceへ =====
        states  = batch.states.to(device).transpose(0, 1)   # [T,B,...]
        actions = batch.actions.to(device).transpose(0, 1)  # [T,B,...]
        optional_fields = get_optional_fields(batch, device=states.device)

        # ===== 2. closed + encoder =====
        closed_res = jepa.forward_posterior(states, actions, **optional_fields)
        closed_pred = closed_res.pred_output
        enc_output  = closed_res.backbone_output

        encoder_encs = enc_output.obs_component
        closed_encs  = closed_pred.obs_component if (getattr(closed_pred, "obs_component", None) is not None) else closed_pred.predictions

        # CPUへ退避
        encoder_encs_cpu = encoder_encs.detach().float().cpu()  # [T,B,...]
        closed_encs_cpu  = closed_encs.detach().float().cpu()   # [T,B,...]

        # GPU参照削除
        del encoder_encs, closed_encs, enc_output, closed_pred, closed_res
        torch.cuda.empty_cache()

        # ===== 3. open =====
        open_res  = jepa.forward_open(states, actions, **optional_fields)
        open_pred = open_res.pred_output
        open_encs = open_pred.obs_component if (getattr(open_pred, "obs_component", None) is not None) else open_pred.predictions

        open_encs_cpu = open_encs.detach().float().cpu()        # [T,B,...]

        # GPU参照削除
        del open_encs, open_pred, open_res
        del states, actions, optional_fields
        torch.cuda.empty_cache()

        # ===== 4. helper: 1 traj 用 =====
        def mse_all_traj(a: torch.Tensor, b: torch.Tensor) -> float:
            # a,b: [T,...]
            return F.mse_loss(a, b).item()
        def rmse_all_traj(a: torch.Tensor, b: torch.Tensor) -> float:
            # a,b: [T,...]
            return torch.sqrt(F.mse_loss(a, b)).item()

        def rms_all_traj(x: torch.Tensor) -> float:
            # x: [T, ...]
            return torch.sqrt((x.float().pow(2)).mean()).item()

        def nrmse_all_traj_pct(pred: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
            # pred, ref: [T, ...]
            rmse = rmse_all_traj(pred, ref)
            scale = rms_all_traj(ref)   # ★ここが分母
            return 100.0 * rmse / (scale + eps)

        def mse_per_timestep_traj(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # a,b: [T,...]
            diff = F.mse_loss(a, b, reduction="none")      # [T,...]
            reduce_dims = tuple(range(1, diff.ndim))       # 時刻以外
            return diff.mean(dim=reduce_dims).detach().cpu()  # [T]
        def rmse_per_timestep_traj(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # a,b: [T,...]
            diff = F.mse_loss(a, b, reduction="none")      # [T,...]  = (a-b)^2
            reduce_dims = tuple(range(1, diff.ndim))       # 時刻以外
            mse_t = diff.mean(dim=reduce_dims)             # [T]
            rmse_t = torch.sqrt(mse_t)                     # [T]
            return rmse_t.detach().cpu()
        
        def rms_per_timestep_traj(x: torch.Tensor) -> torch.Tensor:
            # x: [T, ...] -> [T]
            x2 = x.float().pow(2)
            reduce_dims = tuple(range(1, x2.ndim))
            return torch.sqrt(x2.mean(dim=reduce_dims)).detach().cpu()

        def nrmse_per_timestep_traj_pct(pred: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            rmse_t = rmse_per_timestep_traj(pred, ref)   # [T]
            scale_t = rms_per_timestep_traj(ref)         # ★ ref の RMS
            return (100.0 * rmse_t / (scale_t + eps)).detach().cpu()


        # ===== 5. trajごとに計算 =====
        T, B = encoder_encs_cpu.shape[:2]
        if idxs is None:
            idxs = list(range(B))

        run = Logger.run()
        from pathlib import Path
        out_dir = None
        if run.output_path is not None:
            out_dir = Path(run.output_path) / "latent_mse_per_traj" / name_prefix
            out_dir.mkdir(parents=True, exist_ok=True)

        for bi in idxs:
            enc_i    = encoder_encs_cpu[:, bi]  # [T,...]
            closed_i = closed_encs_cpu[:, bi]
            open_i   = open_encs_cpu[:, bi]

            rmse_closed_open = rmse_all_traj(closed_i, open_i)
            rmse_open_enc    = rmse_all_traj(open_i, enc_i)
            rmse_closed_enc  = rmse_all_traj(closed_i, enc_i)

            # ===== Error rate (%) : NRMSE =====
            nrmse_closed_open_pct = nrmse_all_traj_pct(closed_i, open_i)   # ref=open
            nrmse_open_enc_pct    = nrmse_all_traj_pct(open_i, enc_i)      # ref=enc
            nrmse_closed_enc_pct  = nrmse_all_traj_pct(closed_i, enc_i)    # ref=enc

            rmse_t_closed_open = rmse_per_timestep_traj(closed_i, open_i)  # [T]
            rmse_t_open_enc    = rmse_per_timestep_traj(open_i, enc_i)     # [T]
            rmse_t_closed_enc  = rmse_per_timestep_traj(closed_i, enc_i)   # [T]

            # ===== txt 保存 =====
            if out_dir is not None:
                out_path = out_dir / f"traj{bi:03d}_latent_mse.txt"
                with open(out_path, "w") as f:
                    f.write(f"[LATENT RMSE PER TRAJ] {name_prefix} | traj={bi}\n")
                    f.write(f"closed-open={rmse_closed_open:.6e}\n")
                    f.write(f"open-enc   ={rmse_open_enc:.6e}\n")
                    f.write(f"closed-enc ={rmse_closed_enc:.6e}\n\n")

                    f.write("\n=== NRMSE (%) PER TRAJ ===\n")
                    f.write(f"closed-open={nrmse_closed_open_pct:.3f}%\n")
                    f.write(f"open-enc   ={nrmse_open_enc_pct:.3f}%\n")
                    f.write(f"closed-enc ={nrmse_closed_enc_pct:.3f}%\n\n")

                    f.write("=== timestep RMSE (len=T) ===\n")
                    f.write("closed-open: " + ", ".join(f"{v:.6e}" for v in rmse_t_closed_open.tolist()) + "\n")
                    f.write("open-enc   : " + ", ".join(f"{v:.6e}" for v in rmse_t_open_enc.tolist()) + "\n")
                    f.write("closed-enc : " + ", ".join(f"{v:.6e}" for v in rmse_t_closed_enc.tolist()) + "\n")

                print(f"[LATENT RMSE PER TRAJ] saved → {out_path.resolve()}")

        # ===== 掃除 =====
        del encoder_encs_cpu, closed_encs_cpu, open_encs_cpu
        gc.collect()



    @torch.no_grad()
    def plot_pca_open_closed_anchor_closed(
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
        closed-forward 列を基準 (anchor) に PCA 軸を学習し、
        - closed-forward 列（基準）
        - open-forward 列
        を「同じ PCA 空間」に射影して 2D/3D で可視化する。

        注意:
        - open/closed の潜在次元が異なる場合は、open -> closed への線形射影を最小二乗で求めて合わせる。
        - 可視化は軸の “向き/符号” が PCA の任意性で反転しうるので、形状比較が主。
        """

        import numpy as np
        import gc
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from matplotlib.lines import Line2D

        device = self.device

        # ===== バッチをデバイスへ & optional fields =====
        states  = batch.states.to(device).transpose(0, 1)   # [T,B,...]
        actions = batch.actions.to(device).transpose(0, 1)  # [T,B,...]
        optional_fields = get_optional_fields(batch, device=states.device)

        T_ref, B_ref = states.shape[0], states.shape[1]

        # ===== closed-forward 潜在列 =====
        closed_res = jepa.forward_posterior(states, actions, **optional_fields)
        pred_closed = closed_res.pred_output
        if getattr(pred_closed, "obs_component", None) is not None:
            closed_lat_seq = pred_closed.obs_component    # [T,B,C,H,W] or [T,B,D]
        else:
            closed_lat_seq = pred_closed.predictions      # [T,B,D]

        # ===== open-forward 潜在列 =====
        open_res = jepa.forward_open(states, actions, **optional_fields)
        pred_open = open_res.pred_output
        if getattr(pred_open, "obs_component", None) is not None:
            open_lat_seq = pred_open.obs_component        # [T,B,C,H,W] or [T,B,D]
        else:
            open_lat_seq = pred_open.predictions          # [T,B,D]

        # ===== helper: [T,B,...] or [B,T,...] -> [B,T,D] =====
        def _to_BTD(x, pool="flat"):
            assert torch.is_tensor(x)
            # [T,B,D] or [B,T,D]
            if x.dim() == 3:
                if x.shape[0] == T_ref and x.shape[1] == B_ref:
                    x = x.transpose(0, 1).contiguous()   # [B,T,D]
                return x
            # [T,B,C,H,W] or [B,T,C,H,W]
            if x.dim() == 5:
                if x.shape[0] == T_ref and x.shape[1] == B_ref:
                    x = x.permute(1, 0, 2, 3, 4).contiguous()  # [B,T,C,H,W]
                elif not (x.shape[0] == B_ref and x.shape[1] == T_ref):
                    x = x.transpose(0, 1).contiguous()
                B, T = x.shape[:2]
                if pool == "gap":
                    x = x.mean(dim=(-2, -1)).contiguous()      # [B,T,C]
                else:
                    C, H, W = x.shape[2:]
                    x = x.reshape(B, T, C * H * W).contiguous()
                return x

            # fallback: flatten
            if x.shape[0] == T_ref and x.shape[1] == B_ref:
                x = x.transpose(0, 1).contiguous()
            elif not (x.shape[0] == B_ref and x.shape[1] == T_ref):
                x = x.transpose(0, 1).contiguous()
            B, T = x.shape[:2]
            D = int(np.prod(x.shape[2:]))
            return x.reshape(B, T, D).contiguous()

        # ===== [B,T,D] に揃える（GPU上）=====
        closed_lat_btd = _to_BTD(closed_lat_seq, pool=pool)
        open_lat_btd   = _to_BTD(open_lat_seq,   pool=pool)

        Bc, Tc, Dc = closed_lat_btd.shape
        Bo, To, Do = open_lat_btd.shape
        assert Bc == Bo and Tc == To, f"[PCA-closed-anchor] shape mismatch: closed {closed_lat_btd.shape}, open {open_lat_btd.shape}"
        B, T = Bc, Tc

        if idxs is None:
            idxs = list(range(B))

        # ===== CPUへ移してGPU解放 =====
        closed_cpu = closed_lat_btd.detach().to("cpu")
        open_cpu   = open_lat_btd.detach().to("cpu")

        try:
            del closed_lat_seq, open_lat_seq, closed_lat_btd, open_lat_btd
            del closed_res, open_res, pred_closed, pred_open
            del states, actions, optional_fields
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        # ===== PCA を closed で学習 =====
        Z_closed = closed_cpu.reshape(B * T, Dc).numpy()
        pca = PCA(n_components=min(k, Dc))
        U_closed = pca.fit_transform(Z_closed)  # [B*T, k_eff]
        expl = pca.explained_variance_ratio_
        k_eff = U_closed.shape[1]

        # ===== open を closed 次元へ合わせて同一基底へ射影 =====
        Z_open = open_cpu.reshape(B * T, Do).numpy()
        if Do != Dc:
            # open @ A ≈ closed となる A (Do->Dc) を最小二乗で推定
            A, *_ = np.linalg.lstsq(Z_open, Z_closed, rcond=None)  # (Do,Dc)
            Z_open_in_closed = Z_open @ A                          # (B*T,Dc)
        else:
            Z_open_in_closed = Z_open

        V_open = pca.transform(Z_open_in_closed)  # [B*T, k_eff]

        # ===== [B,T,k] に戻す =====
        U_bt = U_closed.reshape(B, T, k_eff)  # Closed in closed-PCA space
        V_bt = V_open.reshape(B, T, k_eff)    # Open projected into same space

        # ===== 可視化設定 =====
        color_closed = "tab:blue"
        color_open   = "tab:orange"

        # ===== 2D =====
        if k_eff >= 2:
            U2 = U_bt[:, :, :2]
            V2 = V_bt[:, :, :2]
            lim = float(max(abs(U2).max(), abs(V2).max()))

            for i in idxs:
                fig2d, ax2d = plt.subplots(1, 1, figsize=(6, 6), dpi=140)

                u2d = U_bt[i, :, :2]  # closed
                v2d = V_bt[i, :, :2]  # open

                for t in range(T - 1):
                    ax2d.plot(u2d[t:t+2, 0], u2d[t:t+2, 1], color=color_closed, alpha=0.95)
                    ax2d.plot(v2d[t:t+2, 0], v2d[t:t+2, 1], color=color_open,   alpha=0.95)

                # start / goal
                ax2d.scatter(u2d[0,0],  u2d[0,1],  s=12, c=color_closed)
                ax2d.scatter(u2d[-1,0], u2d[-1,1], s=12, c=color_closed)
                ax2d.text(u2d[0,0],  u2d[0,1],  "S", fontsize=9, ha="center", va="center", color=color_closed)
                ax2d.text(u2d[-1,0], u2d[-1,1], "G", fontsize=9, ha="center", va="center", color=color_closed)

                ax2d.scatter(v2d[0,0],  v2d[0,1],  s=12, c=color_open)
                ax2d.scatter(v2d[-1,0], v2d[-1,1], s=12, c=color_open)
                ax2d.text(v2d[0,0],  v2d[0,1],  "S", fontsize=9, ha="center", va="center", color=color_open)
                ax2d.text(v2d[-1,0], v2d[-1,1], "G", fontsize=9, ha="center", va="center", color=color_open)

                ax2d.set_xlim(-lim, lim)
                ax2d.set_ylim(-lim, lim)
                ax2d.set_aspect("equal", adjustable="box")
                ax2d.set_xlabel("PC1")
                ax2d.set_ylabel("PC2")
                ax2d.set_title(
                    f"{name_prefix} | idx={i} | PCA(closed-anchor) top-{k_eff}: "
                    + ", ".join(f"{v:.2f}" for v in expl[:k_eff])
                )
                ax2d.legend(handles=[
                    Line2D([0],[0], color=color_closed, lw=2, label="Closed (anchor)"),
                    Line2D([0],[0], color=color_open,   lw=2, label="Open (projected)"),
                ], loc="best", frameon=True)

                # if not notebook:
                #     Logger.run().log_figure(
                #         fig2d,
                #         f"{name_prefix}-pca_closedanchor-2d-i{i}",
                #         dir_name="pca_closed_anchor/pca2d_pertraj",
                #     )
                #     plt.close(fig2d)
                # else:
                #     plt.show()
                #     plt.close(fig2d)

        # ===== 3D =====
        if k_eff >= 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa

            for i in idxs:
                fig3d = plt.figure(figsize=(8, 8), dpi=140)
                ax3d = fig3d.add_subplot(111, projection="3d")

                u3d = U_bt[i, :, :3]  # closed
                v3d = V_bt[i, :, :3]  # open

                for t in range(T - 1):
                    ax3d.plot(u3d[t:t+2, 0], u3d[t:t+2, 1], u3d[t:t+2, 2], color=color_closed, alpha=0.95)
                    ax3d.plot(v3d[t:t+2, 0], v3d[t:t+2, 1], v3d[t:t+2, 2], color=color_open,   alpha=0.95)

                s_size = 24
                off = 0.0

                ax3d.scatter(u3d[0,0],  u3d[0,1],  u3d[0,2],  s=s_size, c=color_closed, depthshade=False)
                ax3d.scatter(u3d[-1,0], u3d[-1,1], u3d[-1,2], s=s_size, c=color_closed, depthshade=False)
                ax3d.text(u3d[0,0]+off,  u3d[0,1]+off,  u3d[0,2]+off,  "S", color=color_closed, fontsize=9)
                ax3d.text(u3d[-1,0]+off, u3d[-1,1]+off, u3d[-1,2]+off, "G", color=color_closed, fontsize=9)

                ax3d.scatter(v3d[0,0],  v3d[0,1],  v3d[0,2],  s=s_size, c=color_open, depthshade=False)
                ax3d.scatter(v3d[-1,0], v3d[-1,1], v3d[-1,2], s=s_size, c=color_open, depthshade=False)
                ax3d.text(v3d[0,0]+off,  v3d[0,1]+off,  v3d[0,2]+off,  "S", color=color_open, fontsize=9)
                ax3d.text(v3d[-1,0]+off, v3d[-1,1]+off, v3d[-1,2]+off, "G", color=color_open, fontsize=9)

                ax3d.set_xlabel("PC1")
                ax3d.set_ylabel("PC2")
                ax3d.set_zlabel("PC3")
                ax3d.set_title(
                    f"PCA(3D, closed-anchor) — {name_prefix} | var exp: "
                    + ", ".join(f"{v:.2f}" for v in expl[:3])
                )
                ax3d.legend(handles=[
                    Line2D([0],[0], color=color_closed, lw=2, label="Closed (anchor)"),
                    Line2D([0],[0], color=color_open,   lw=2, label="Open (projected)"),
                ], loc="upper left", frameon=True)

                if not notebook:
                    Logger.run().log_figure(
                        fig3d,
                        f"{name_prefix}-pca_closedanchor-3d-i{i}",
                        dir_name="pca_closed_anchor/pca3d_pertraj",
                    )
                    plt.close(fig3d)
                else:
                    plt.show()
                    plt.close(fig3d)

        # ===== 時刻方向 cosine（closed vs open in closed-PCA space） =====
        try:
            U_t = torch.from_numpy(U_bt)  # [B,T,k]
            V_t = torch.from_numpy(V_bt)

            def _cos_mean(a, b, eps=1e-8):
                a = a / (a.norm(dim=-1, keepdim=True) + eps)
                b = b / (b.norm(dim=-1, keepdim=True) + eps)
                return (a * b).sum(-1).mean().item()

            cos_over_time = []
            for t in range(T):
                cos_over_time.append(_cos_mean(U_t[:, t, :k_eff], V_t[:, t, :k_eff]))

            figc, axc = plt.subplots(1, 1, figsize=(7, 3), dpi=140)
            axc.plot(range(T), cos_over_time, marker="o", linewidth=1.5)
            axc.set_xlabel("t (horizon)")
            axc.set_ylabel("cosine in PCA(closed) space")
            axc.set_title(
                f"Timewise Cosine (closed vs open) — mean={np.mean(cos_over_time):.3f} | var exp (top-{k_eff}): "
                + ", ".join(f"{v:.2f}" for v in expl[:k_eff])
            )
            figc.tight_layout()

            if not notebook:
                Logger.run().log_figure(
                    figc,
                    f"{name_prefix}-pca_closedanchor-timewise-cosine",
                    dir_name="pca_closed_anchor/pca_cos",
                )
                plt.close(figc)
            else:
                plt.show()
                plt.close(figc)
        except Exception:
            pass

        # ===== cleanup =====
        plt.close("all")
        try:
            del U_bt, V_bt, U_closed, V_open
            del Z_closed, Z_open, Z_open_in_closed
            del closed_cpu, open_cpu
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


    @torch.no_grad()
    def plot_pca_long_horizon(
        self,
        loader,                 # ★ DataLoader / NormalizedDataLoader
        jepa: "JEPA",
        name_prefix: str = "",
        notebook: bool = False,
        pool: str = "gap",      # "gap"/"flat"
        k: int = 3,
        take_per_batch: int = 1,        # ★ 各batchから何本取るか（おすすめ1）
        pick_mode: str = "first",       # "first" or "random"
        max_batches: int = None,        # ★ Noneなら全部
        align_closed: bool = True,
        max_num_plot: int = 10000000000, #プロットするエピソードの上限
    ):
        """
        loader から複数 batch を順に取り、
        各batchから take_per_batch 本のtrajを抜いて時間方向に連結し、
        その“疑似ロング軌道”を PCA 空間で可視化する。
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from sklearn.decomposition import PCA

        device = self.device

        # ---------- helper: [T,B,...] or [B,T,...] を [B,T,D] に ----------
        def _to_BTD(x, T_ref, B_ref, pool="flat"):
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
                    x = x.mean(dim=(-2, -1)).contiguous()   # -> [B,T,C]
                else:
                    C, H, W = x.shape[2:]
                    x = x.reshape(B, T, C * H * W).contiguous()
                return x

            # general
            if x.shape[0] == T_ref and x.shape[1] == B_ref:
                x = x.transpose(0, 1).contiguous()
            elif not (x.shape[0] == B_ref and x.shape[1] == T_ref):
                x = x.transpose(0, 1).contiguous()

            B, T = x.shape[:2]
            D = int(np.prod(x.shape[2:]))
            return x.reshape(B, T, D).contiguous()

        # ---------- 収集：疑似ロングにするため、CPU上に [sumT, D] を作る ----------
        enc_chunks = []     # list of [T,De] on CPU
        clo_chunks = []     # list of [T,Dc] on CPU
        seg_ends = []       # 境界可視化用（連結点のインデックス）
        total_T = 0

        itr = iter(loader)
        n_batch = 0
        n_traj = 0

        while True:
            if n_traj >= max_num_plot:
                break
            if max_batches is not None and n_batch >= max_batches:
                break
            try:
                batch = next(itr)
            except StopIteration:
                break

            # ---- states/actions: [T,B,...] へ ----
            states  = batch.states.to(device).transpose(0, 1)   # [T,B,...]
            actions = batch.actions.to(device).transpose(0, 1)
            optional_fields = get_optional_fields(batch, device=states.device)
            T_ref, B_ref = states.shape[0], states.shape[1]

            # ★ forward_posterior を1回だけ呼ぶ（closed と encoder を同時に取る）
            res = jepa.forward_posterior(states, actions, **optional_fields)
            pred_output = res.pred_output
            enc_output  = res.backbone_output


            print("[pldm/probing/evaluator.py] states.mean:", states.mean())
            print("[pldm/probing/evaluator.py] states.min:", states.min())
            print("[pldm/probing/evaluator.py] states.max:", states.max())
            print("[pldm/probing/evaluator.py] states.std:", states.std())
            
            
            

            if getattr(pred_output, "obs_component", None) is not None:
                closed_lat_seq = pred_output.obs_component
            else:
                closed_lat_seq = pred_output.predictions

            encoder_lat_seq = enc_output.obs_component

            print("[pldm/probing/evaluator.py] closed_lat_seq.mean:", closed_lat_seq.mean())
            print("[pldm/probing/evaluator.py] closed_lat_seq.min:", closed_lat_seq.min())
            print("[pldm/probing/evaluator.py] closed_lat_seq.max:", closed_lat_seq.max())
            print("[pldm/probing/evaluator.py] closed_lat_seq.std:", closed_lat_seq.std())


            print("[pldm/probing/evaluator.py] encoder_lat_seq.mean:", encoder_lat_seq.mean())
            print("[pldm/probing/evaluator.py] encoder_lat_seq.min:", encoder_lat_seq.min())
            print("[pldm/probing/evaluator.py] encoder_lat_seq.max:", encoder_lat_seq.max())
            print("[pldm/probing/evaluator.py] encoder_lat_seq.std:", encoder_lat_seq.std())

            # ---- [B,T,D] に整形 ----
            closed_bt = _to_BTD(closed_lat_seq,  T_ref, B_ref, pool=pool)   # [B,T,Dc]
            enc_bt    = _to_BTD(encoder_lat_seq, T_ref, B_ref, pool=pool)   # [B,T,De]

            B, T, Dc = closed_bt.shape
            _, _, De = enc_bt.shape

            # ---- このbatchから取るtraj indexを決める ----
            if take_per_batch >= B:
                pick_idxs = list(range(B))
            else:
                if pick_mode == "random":
                    pick_idxs = np.random.choice(B, size=take_per_batch, replace=False).tolist()
                else:
                    pick_idxs = list(range(take_per_batch))  # first

            # ---- 連結（時間方向に append）----
            for bi in pick_idxs:
                if n_traj >= max_num_plot:
                    break
                enc_i = enc_bt[bi].detach().float().cpu()     # [T,De]
                clo_i = closed_bt[bi].detach().float().cpu()  # [T,Dc]
                enc_chunks.append(enc_i)
                clo_chunks.append(clo_i)
                total_T += T
                seg_ends.append(total_T)  # ここが境界（次が別traj）
                
                n_traj += 1

            # ---- 掃除 ----
            del res, pred_output, enc_output, closed_lat_seq, encoder_lat_seq
            del closed_bt, enc_bt, states, actions, optional_fields
            torch.cuda.empty_cache()

            n_batch += 1

        if len(enc_chunks) == 0:
            print("[plot_pca_pseudo_long] No data collected from loader.")
            return

        # ---------- CPUで連結 ----------
        enc_long = torch.cat(enc_chunks, dim=0)   # [sumT, De]
        clo_long = torch.cat(clo_chunks, dim=0)   # [sumT, Dc]

        # ---------- PCAは encoder で学習 ----------
        Z_enc = enc_long.numpy()
        pca = PCA(n_components=min(k, Z_enc.shape[1]))
        U_enc = pca.fit_transform(Z_enc)          # [sumT, k_eff]
        expl = pca.explained_variance_ratio_
        k_eff = U_enc.shape[1]

        # ---------- closed を同一基底へ射影 ----------
        Z_clo = clo_long.numpy()
        if Z_clo.shape[1] == Z_enc.shape[1]:
            Z_clo_in_enc_dim = Z_clo
        else:
            if not align_closed:
                raise ValueError(f"Dc({Z_clo.shape[1]}) != De({Z_enc.shape[1]}). align_closed=Falseなら次元一致が必要。")
            A, *_ = np.linalg.lstsq(Z_clo, Z_enc, rcond=None)
            Z_clo_in_enc_dim = Z_clo @ A

        V_clo = pca.transform(Z_clo_in_enc_dim)   # [sumT, k_eff]

        # ---------- 可視化：疑似ロング1本として描く ----------
        color_u = "black"       # Encoder
        color_v = "firebrick"  # Closed

        # 2D
        if k_eff >= 2:
            fig2d, ax2d = plt.subplots(1, 1, figsize=(7, 7), dpi=140)
            u2 = U_enc[:, :2]
            v2 = V_clo[:, :2]
            lim = float(max(np.abs(u2).max(), np.abs(v2).max()))

            # 線で描く
            ax2d.plot(u2[:,0], u2[:,1], color=color_u, alpha=0.95, linewidth=1.2)
            ax2d.plot(v2[:,0], v2[:,1], color=color_v, alpha=0.95, linewidth=1.2)

            # start / end
            ax2d.scatter(u2[0,0], u2[0,1], s=18, c=color_u); ax2d.text(u2[0,0], u2[0,1], "S", color=color_u)
            ax2d.scatter(u2[-1,0], u2[-1,1], s=18, c=color_u); ax2d.text(u2[-1,0], u2[-1,1], "G", color=color_u)

            ax2d.scatter(v2[0,0], v2[0,1], s=18, c=color_v); ax2d.text(v2[0,0], v2[0,1], "S", color=color_v)
            ax2d.scatter(v2[-1,0], v2[-1,1], s=18, c=color_v); ax2d.text(v2[-1,0], v2[-1,1], "G", color=color_v)

            # 境界（traj切替点）を薄い点でマーキング（見た目で「連結」を意識できる）
            for s in seg_ends[:-1]:
                ax2d.scatter(u2[s-1,0], u2[s-1,1], s=10, c=color_u, alpha=0.35)
                ax2d.scatter(v2[s-1,0], v2[s-1,1], s=10, c=color_v, alpha=0.35)

            ax2d.set_xlim(-lim, lim); ax2d.set_ylim(-lim, lim)
            ax2d.set_aspect("equal", adjustable="box")
            ax2d.set_xlabel("PC1"); ax2d.set_ylabel("PC2")
            ax2d.set_title(
                f"{name_prefix} | pseudo-long PCA (top-{k_eff}) var exp: " +
                ", ".join(f"{v:.2f}" for v in expl[:k_eff])
            )
            handles = [
                Line2D([0],[0], color=color_u, lw=2, label="Encoder"),
                Line2D([0],[0], color=color_v, lw=2, label="Closed"),
            ]
            ax2d.legend(handles=handles, loc="best", frameon=True)

            if not notebook:
                Logger.run().log_figure(fig2d, f"{name_prefix}-pca-2d-pseudo-long", dir_name=f"pca_pseudolong/{name_prefix}/2d")
                plt.close(fig2d)
            else:
                plt.show()
                plt.close(fig2d)

        # 3D
        if k_eff >= 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            fig3d = plt.figure(figsize=(8, 8), dpi=140)
            ax3d = fig3d.add_subplot(111, projection="3d")
            u3 = U_enc[:, :3]
            v3 = V_clo[:, :3]

            ax3d.plot(u3[:,0], u3[:,1], u3[:,2], color=color_u, alpha=0.95, linewidth=1.1)
            ax3d.plot(v3[:,0], v3[:,1], v3[:,2], color=color_v, alpha=0.95, linewidth=1.1)

            ax3d.set_xlabel("PC1"); ax3d.set_ylabel("PC2"); ax3d.set_zlabel("PC3")
            ax3d.set_title(
                f"{name_prefix} | pseudo-long PCA (3D) var exp: " +
                ", ".join(f"{v:.2f}" for v in expl[:3])
            )
            handles = [
                Line2D([0],[0], color=color_u, lw=2, label="Encoder"),
                Line2D([0],[0], color=color_v, lw=2, label="Closed"),
            ]
            ax3d.legend(handles=handles, loc="upper left", frameon=True)

            if not notebook:
                Logger.run().log_figure(fig3d, f"{name_prefix}-pca-3d-pseudo-long", dir_name=f"pca_pseudolong/{name_prefix}/3d")
                plt.close(fig3d)
            else:
                plt.show()
                plt.close(fig3d)

        plt.close("all")
        del enc_long, clo_long, U_enc, V_clo
        gc.collect()
        torch.cuda.empty_cache()




    @torch.no_grad()
    def pca_visual_encoder_bluebox(
        self,
        batch,
        jepa: "JEPA",
        name_prefix: str = "",
        idxs: Optional[List[int]] = None,
        notebook: bool = False,
        pool: str = "flat",              # "gap"/"flat"
        k: int = 3,
        align_closed: bool = True,
        is_train: bool = False,
        n_each_side: int = 2,
        dx: float = 0.04,
        dy: float = 0.04,
        x_half_range: float = 0.10,
        y_half_range: float = 0.10,
        n_x_points: int = 101,
        n_y_points: int = 101,
    ):

        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import Normalize

        env_generator = FrankaEnvsGenerator(
            model_path=self.config.model_path,
            n_envs=1,
            max_dq=self.config.max_dq,
            camera_name=self.config.camera_name
        )

        env = env_generator()[0]
        device = self.device

        print("[pldm/probing/evaluator.py] batch.states.shape:", batch.states.shape)

        center = np.array([0.515, 0.0, 0.05], dtype=np.float32)
        fixed_goal = np.array([0.58, 0.00, 0.05], dtype=np.float32)

        positions = []

        x_values = np.linspace(
            center[0] - x_half_range,
            center[0] + x_half_range,
            n_x_points,
            dtype=np.float32,
        )

        y_values = np.linspace(
            center[1] - y_half_range,
            center[1] + y_half_range,
            n_y_points,
            dtype=np.float32,
        )

        x_center_idx = n_x_points // 2
        for idx, x in enumerate(x_values):
            p = center.copy()
            p[0] = x
            positions.append(("x", idx - x_center_idx, p.copy()))

        y_center_idx = n_y_points // 2
        for idx, y in enumerate(y_values):
            if np.isclose(y, center[1]):
                continue
            p = center.copy()
            p[1] = y
            positions.append(("y", idx - y_center_idx, p.copy()))

        imgs = []
        labels = []
        coords = []

        for axis, step_idx, start_pos in positions:
            obs = env.reset(
                start_pos=start_pos,
                goal_pos=fixed_goal,
                robot_only=False,
            )

            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs)

            imgs.append(obs.float().cpu())
            labels.append((axis, step_idx))
            coords.append(start_pos[:2].copy())

        imgs = torch.stack(imgs, dim=0)   # [N,C,H,W]

        states = imgs.unsqueeze(0).to(device)   # [1,N,C,H,W]

        enc_output = jepa.backbone.forward_multiple(states)
        z = enc_output.obs_component

        if z.dim() == 5:
            z = z[0]   # [N,C,H,W]
            if pool == "gap":
                z = z.mean(dim=(-2, -1))   # [N,C]
            else:
                z = z.flatten(start_dim=1) # [N,C*H*W]
        elif z.dim() == 3:
            z = z[0]   # [N,D]
        else:
            raise ValueError(f"Unexpected encoder output shape: {z.shape}")

        z_np = z.detach().cpu().numpy()

        pca = PCA(n_components=2)
        z2 = pca.fit_transform(z_np)

        coords_np = np.asarray(coords)   # [N, 2]

        # =========================================================
        # helper: 薄い -> 濃い のグラデーション色を作る
        # =========================================================
        def make_grad_colors(n, cmap_name, start=0.25, end=0.95):
            cmap = cm.get_cmap(cmap_name)
            vals = np.linspace(start, end, n)   # 薄い -> 濃い
            return [cmap(v) for v in vals]

        def draw_gradient_series(ax, pts, idx_list, labels, axis_name, cmap_name, marker):
            idx_list = sorted(idx_list, key=lambda i: labels[i][1])
            n = len(idx_list)
            colors = make_grad_colors(n, cmap_name)

            for seg_i in range(n - 1):
                i0 = idx_list[seg_i]
                i1 = idx_list[seg_i + 1]
                ax.plot(
                    [pts[i0, 0], pts[i1, 0]],
                    [pts[i0, 1], pts[i1, 1]],
                    color=colors[seg_i + 1],
                    linewidth=2.0,
                    alpha=0.95,
                )


            for local_i, global_i in enumerate(idx_list):
                ax.scatter(
                    pts[global_i, 0],
                    pts[global_i, 1],
                    color=colors[local_i],
                    s=42,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=3,
                )
                _, step_idx = labels[global_i]
                ax.text(
                    pts[global_i, 0],
                    pts[global_i, 1],
                    f"{axis_name}{step_idx}",
                    fontsize=8,
                    alpha=0.9,
                )


            ax.plot([], [], color=colors[-1], marker=marker, label=f"{axis_name}-line")

        # -----------------------------
        # 7) PCA 可視化
        # -----------------------------
        fig, ax = plt.subplots(figsize=(7, 7), dpi=140)

        xs_idx = [i for i, (axis, step_idx) in enumerate(labels) if axis == "x"]
        ys_idx = [i for i, (axis, step_idx) in enumerate(labels) if axis == "y"]

        # x方向: 青系で薄い -> 濃い
        draw_gradient_series(ax, z2, xs_idx, labels, axis_name="x", cmap_name="Blues", marker="o")

        # y方向: 赤系で薄い -> 濃い
        draw_gradient_series(ax, z2, ys_idx, labels, axis_name="y", cmap_name="Reds", marker="s")

        center_idx = [i for i, (axis, step_idx) in enumerate(labels) if step_idx == 0]


        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            f"{name_prefix} | PCA of encoder features\n"
            f"explained variance = {pca.explained_variance_ratio_[0]:.3f}, "
            f"{pca.explained_variance_ratio_[1]:.3f}"
        )
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        Logger.run().log_figure(fig, f"{name_prefix}_pca_cross", dir_name="pca_encoder_cross")
        plt.close(fig)

        # -----------------------------
        # 8) 実空間(xy平面)での配置も可視化
        # -----------------------------
        fig_xy, ax_xy = plt.subplots(figsize=(7, 7), dpi=140)

        draw_gradient_series(ax_xy, coords_np, xs_idx, labels, axis_name="x", cmap_name="Blues", marker="o")
        draw_gradient_series(ax_xy, coords_np, ys_idx, labels, axis_name="y", cmap_name="Reds", marker="s")

        # for t, i in enumerate(center_idx):
        #     ax_xy.scatter(
        #         coords_np[i, 0],
        #         coords_np[i, 1],
        #         s=140,
        #         marker="*",
        #         color="gold",
        #         edgecolors="black",
        #         linewidths=0.5,
        #         label="center" if t == 0 else None,
        #         zorder=5,
        #     )

        ax_xy.scatter(
            fixed_goal[0], fixed_goal[1],
            s=140,
            marker="X",
            color="green",
            edgecolors="black",
            linewidths=0.5,
            label="goal",
            zorder=5,
        )

        ax_xy.set_xlabel("x")
        ax_xy.set_ylabel("y")
        ax_xy.set_title(f"{name_prefix} | bluebox positions on xy-plane")
        ax_xy.legend()
        ax_xy.grid(True)
        ax_xy.set_aspect("equal", adjustable="box")
        fig_xy.tight_layout()

        Logger.run().log_figure(
            fig_xy,
            f"{name_prefix}_xy_cross",
            dir_name="pca_encoder_cross"
        )
        plt.close(fig_xy)

        print("finished making pca visual encoder")
        pass




    @torch.no_grad()
    def pca_visual_encoder_franka(
        self,
        batch,
        jepa: "JEPA",
        name_prefix: str = "",
        idxs: Optional[List[int]] = None,
        notebook: bool = False,
        pool: str = "flat",              # "gap"/"flat"
        k: int = 3,
        align_closed: bool = True,
        is_train: bool = False,
        x_half_range: float = 0.10,
        y_half_range: float = 0.10,
        n_x_points: int = 101,
        n_y_points: int = 101,
        ee_z: float = 0.10,
        fixed_box_pos: tuple = (0.515, 0.0, 0.05),
        fixed_goal_pos: tuple = (0.58, 0.00, 0.05),
        robot_only: bool = True,
        settle_steps: int = 300,
    ):

        def make_grad_colors(n, cmap_name, start=0.25, end=0.95):
            cmap = cm.get_cmap(cmap_name)
            vals = np.linspace(start, end, n)   # 薄い -> 濃い
            return [cmap(v) for v in vals]

        def draw_gradient_series(ax, pts, idx_list, labels, axis_name, cmap_name, marker):
            idx_list = sorted(idx_list, key=lambda i: labels[i][1])
            n = len(idx_list)
            colors = make_grad_colors(n, cmap_name)

            # 線（グラデーション）
            for seg_i in range(n - 1):
                i0 = idx_list[seg_i]
                i1 = idx_list[seg_i + 1]
                ax.plot(
                    [pts[i0, 0], pts[i1, 0]],
                    [pts[i0, 1], pts[i1, 1]],
                    color=colors[seg_i + 1],
                    linewidth=2.0,
                    alpha=0.95,
                )

            # 点
            for local_i, global_i in enumerate(idx_list):
                ax.scatter(
                    pts[global_i, 0],
                    pts[global_i, 1],
                    color=colors[local_i],
                    s=42,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=3,
                )
                _, step_idx = labels[global_i]
                ax.text(
                    pts[global_i, 0],
                    pts[global_i, 1],
                    f"{axis_name}{step_idx}",
                    fontsize=8,
                )

            # 凡例
            ax.plot([], [], color=colors[-1], marker=marker, label=f"{axis_name}-line")





        env_generator = FrankaEnvsGenerator(
            model_path=self.config.model_path,
            n_envs=1,
            max_dq=self.config.max_dq,
            camera_name=self.config.camera_name,
        )

        env = env_generator()[0]
        device = self.device


        # -----------------------------
        # 1) EE の十字配置を作る
        # -----------------------------
        # reset() の初期IKターゲットと揃える
        ee_center = np.array([0.515, 0.0, ee_z], dtype=np.float32)
        fixed_box_pos = np.array(fixed_box_pos, dtype=np.float32)
        fixed_goal_pos = np.array(fixed_goal_pos, dtype=np.float32)

        positions = []

        x_values = np.linspace(
            ee_center[0] - x_half_range,
            ee_center[0] + x_half_range,
            n_x_points,
            dtype=np.float32,
        )

        y_values = np.linspace(
            ee_center[1] - y_half_range,
            ee_center[1] + y_half_range,
            n_y_points,
            dtype=np.float32,
        )

        x_center_idx = n_x_points // 2
        for idx, x in enumerate(x_values):
            p = ee_center.copy()
            p[0] = x
            positions.append(("x", idx - x_center_idx, p.copy()))

        y_center_idx = n_y_points // 2
        for idx, y in enumerate(y_values):
            # if np.isclose(y, ee_center[1]):
            #     continue
            p = ee_center.copy()
            p[1] = y
            positions.append(("y", idx - y_center_idx, p.copy()))

        # -----------------------------
        # 2) 各EE位置で画像を取得
        # -----------------------------
        imgs = []
        labels = []
        coords = []

        for axis, step_idx, ee_target in positions:
            # 毎回環境を初期状態へ戻す
            obs = env.reset(
                start_pos=fixed_box_pos,
                goal_pos=fixed_goal_pos,
                robot_only=robot_only,
            )

            # EE を target 位置へ移動
            try:
                _, ee_actual = env.set_xyz(
                    target_pos=ee_target,
                    target_rotmat=None,
                    rot_weight=0.1,
                    settle_steps=settle_steps,
                    sync_ctrl=True,
                )
            except Exception as e:
                print(f"[WARN] IK failed at {axis}{step_idx}: target={ee_target}, err={e}")
                continue

            obs = env.get_obs()

            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs)

            imgs.append(obs.float().cpu())
            labels.append((axis, step_idx))
            coords.append(ee_actual[:2].copy())   # 実際のEE位置を使う

        if len(imgs) == 0:
            raise RuntimeError("No valid samples were collected in pca_visual_encoder_franka().")

        imgs = torch.stack(imgs, dim=0)   # [N,C,H,W]

        # -----------------------------
        # 3) encoder に通すため T=1 にする
        # -----------------------------
        states = imgs.unsqueeze(0).to(device)   # [1,N,C,H,W]

        enc_output = jepa.backbone.forward_multiple(states)
        z = enc_output.obs_component


        # -----------------------------
        # 4) [N,D] に整形
        # -----------------------------
        if z.dim() == 5:
            z = z[0]   # [N,C,H,W]
            if pool == "gap":
                z = z.mean(dim=(-2, -1))   # [N,C]
            else:
                z = z.flatten(start_dim=1) # [N,C*H*W]
        elif z.dim() == 3:
            z = z[0]   # [N,D]
        else:
            raise ValueError(f"Unexpected encoder output shape: {z.shape}")

        z_np = z.detach().cpu().numpy()

        # -----------------------------
        # 5) PCA
        # -----------------------------
        pca = PCA(n_components=2)
        z2 = pca.fit_transform(z_np)

        # -----------------------------
        # 6) 潜在空間の可視化
        # -----------------------------
        fig, ax = plt.subplots(figsize=(7, 7), dpi=140)

        xs_idx = [i for i, (axis, step_idx) in enumerate(labels) if axis == "x"]
        xs_idx = sorted(xs_idx, key=lambda i: labels[i][1])
        # ax.plot(z2[xs_idx, 0], z2[xs_idx, 1], marker="o", label="x-line")
        draw_gradient_series(ax, z2, xs_idx, labels, "x", "Blues", "o")
        for i in xs_idx:
            _, step_idx = labels[i]
            ax.text(z2[i, 0], z2[i, 1], f"x{step_idx}", fontsize=9)

        ys_idx = [i for i, (axis, step_idx) in enumerate(labels) if axis == "y"]
        ys_idx = sorted(ys_idx, key=lambda i: labels[i][1])
        # ax.plot(z2[ys_idx, 0], z2[ys_idx, 1], marker="s", label="y-line")
        draw_gradient_series(ax, z2, ys_idx, labels, "y", "Reds", "s")
        for i in ys_idx:
            _, step_idx = labels[i]
            ax.text(z2[i, 0], z2[i, 1], f"y{step_idx}", fontsize=9)


        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            f"{name_prefix} | PCA of encoder features (Franka EE motion)\n"
            f"explained variance = {pca.explained_variance_ratio_[0]:.3f}, "
            f"{pca.explained_variance_ratio_[1]:.3f}"
        )
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        Logger.run().log_figure(
            fig,
            f"{name_prefix}_pca_franka_cross",
            dir_name="pca_encoder_cross"
        )
        plt.close(fig)

        # -----------------------------
        # 7) 実空間(xy平面)のEE位置も可視化
        # -----------------------------
        coords_np = np.asarray(coords)   # [N,2]

        fig_xy, ax_xy = plt.subplots(figsize=(7, 7), dpi=140)

        xs_idx = [i for i, (axis, step_idx) in enumerate(labels) if axis == "x"]
        xs_idx = sorted(xs_idx, key=lambda i: labels[i][1])
        # ax_xy.plot(
        #     coords_np[xs_idx, 0],
        #     coords_np[xs_idx, 1],
        #     marker="o",
        #     label="x-line"
        # )
        draw_gradient_series(ax_xy, coords_np, xs_idx, labels, "x", "Blues", "o")
        for i in xs_idx:
            _, step_idx = labels[i]
            ax_xy.text(coords_np[i, 0], coords_np[i, 1], f"x{step_idx}", fontsize=9)

        ys_idx = [i for i, (axis, step_idx) in enumerate(labels) if axis == "y"]
        ys_idx = sorted(ys_idx, key=lambda i: labels[i][1])
        # ax_xy.plot(
        #     coords_np[ys_idx, 0],
        #     coords_np[ys_idx, 1],
        #     marker="s",
        #     label="y-line"
        # )
        draw_gradient_series(ax_xy, coords_np, ys_idx, labels, "y", "Reds", "s")
        for i in ys_idx:
            _, step_idx = labels[i]
            ax_xy.text(coords_np[i, 0], coords_np[i, 1], f"y{step_idx}", fontsize=9)

        # center_idx = [i for i, (axis, step_idx) in enumerate(labels) if step_idx == 0]
        # for t, i in enumerate(center_idx):
        #     ax_xy.scatter(
        #         coords_np[i, 0], coords_np[i, 1],
        #         s=120, marker="*",
        #         label="center" if t == 0 else None
        #     )


        if not robot_only:
            ax_xy.scatter(
                fixed_box_pos[0], fixed_box_pos[1],
                s=120, marker="X", label="box"
            )

        ax_xy.set_xlabel("x")
        ax_xy.set_ylabel("y")
        ax_xy.set_title(f"{name_prefix} | Franka EE positions on xy-plane")
        ax_xy.legend()
        ax_xy.grid(True)
        ax_xy.set_aspect("equal", adjustable="box")
        fig_xy.tight_layout()

        Logger.run().log_figure(
            fig_xy,
            f"{name_prefix}_xy_franka_cross",
            dir_name="pca_encoder_cross"
        )
        plt.close(fig_xy)

        print("finished making pca visual encoder for franka")


    @torch.no_grad()
    def pca_visual_encoder_bluebox_3d(
        self,
        batch,
        jepa: "JEPA",
        name_prefix: str = "",
        idxs: Optional[List[int]] = None,
        notebook: bool = False,
        pool: str = "flat",              # "gap"/"flat"
        k: int = 3,
        align_closed: bool = True,
        is_train: bool = False,
        n_each_side: int = 2,
        dx: float = 0.04,
        dy: float = 0.04,
        x_half_range: float = 0.10,
        y_half_range: float = 0.10,
        n_x_points: int = 101,
        n_y_points: int = 101,
    ):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        env_generator = FrankaEnvsGenerator(
            model_path=self.config.model_path,
            n_envs=1,
            max_dq=self.config.max_dq,
            camera_name=self.config.camera_name
        )

        env = env_generator()[0]
        device = self.device

        print("[pldm/probing/evaluator.py] batch.states.shape:", batch.states.shape)

        center = np.array([0.515, 0.0, 0.05], dtype=np.float32)
        fixed_goal = np.array([0.58, 0.00, 0.05], dtype=np.float32)

        positions = []

        x_values = np.linspace(
            center[0] - x_half_range,
            center[0] + x_half_range,
            n_x_points,
            dtype=np.float32,
        )

        y_values = np.linspace(
            center[1] - y_half_range,
            center[1] + y_half_range,
            n_y_points,
            dtype=np.float32,
        )

        x_center_idx = n_x_points // 2
        for idx, x in enumerate(x_values):
            p = center.copy()
            p[0] = x
            positions.append(("x", idx - x_center_idx, p.copy()))

        y_center_idx = n_y_points // 2
        for idx, y in enumerate(y_values):
            if np.isclose(y, center[1]):
                continue
            p = center.copy()
            p[1] = y
            positions.append(("y", idx - y_center_idx, p.copy()))

        imgs = []
        labels = []
        coords = []

        for axis, step_idx, start_pos in positions:
            obs = env.reset(
                start_pos=start_pos,
                goal_pos=fixed_goal,
                robot_only=False,
            )

            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs)

            imgs.append(obs.float().cpu())
            labels.append((axis, step_idx))
            coords.append(start_pos[:2].copy())

        imgs = torch.stack(imgs, dim=0)   # [N,C,H,W]
        states = imgs.unsqueeze(0).to(device)   # [1,N,C,H,W]

        enc_output = jepa.backbone.forward_multiple(states)
        z = enc_output.obs_component

        if z.dim() == 5:
            z = z[0]   # [N,C,H,W]
            if pool == "gap":
                z = z.mean(dim=(-2, -1))   # [N,C]
            else:
                z = z.flatten(start_dim=1) # [N,C*H*W]
        elif z.dim() == 3:
            z = z[0]   # [N,D]
        else:
            raise ValueError(f"Unexpected encoder output shape: {z.shape}")

        z_np = z.detach().cpu().numpy()

        # 3次元PCA
        pca = PCA(n_components=3)
        z3 = pca.fit_transform(z_np)

        coords_np = np.asarray(coords)   # [N, 2]

        def make_grad_colors(n, cmap_name, start=0.25, end=0.95):
            cmap = cm.get_cmap(cmap_name)
            vals = np.linspace(start, end, n)   # 薄い -> 濃い
            return [cmap(v) for v in vals]

        def draw_gradient_series_3d(ax, pts, idx_list, labels, axis_name, cmap_name, marker):
            idx_list = sorted(idx_list, key=lambda i: labels[i][1])
            n = len(idx_list)
            colors = make_grad_colors(n, cmap_name)

            # 線分
            for seg_i in range(n - 1):
                i0 = idx_list[seg_i]
                i1 = idx_list[seg_i + 1]
                ax.plot(
                    [pts[i0, 0], pts[i1, 0]],
                    [pts[i0, 1], pts[i1, 1]],
                    [pts[i0, 2], pts[i1, 2]],
                    color=colors[seg_i + 1],
                    linewidth=2.0,
                    alpha=0.95,
                )

            # 点
            for local_i, global_i in enumerate(idx_list):
                ax.scatter(
                    pts[global_i, 0],
                    pts[global_i, 1],
                    pts[global_i, 2],
                    color=colors[local_i],
                    s=42,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.3,
                    depthshade=True,
                )
                _, step_idx = labels[global_i]
                ax.text(
                    pts[global_i, 0],
                    pts[global_i, 1],
                    pts[global_i, 2],
                    f"{axis_name}{step_idx}",
                    fontsize=8,
                    alpha=0.9,
                )

            # 凡例用ダミー
            ax.plot([], [], [], color=colors[-1], marker=marker, label=f"{axis_name}-line")

        # -----------------------------
        # 3D PCA 可視化
        # -----------------------------
        fig = plt.figure(figsize=(8, 8), dpi=140)
        ax = fig.add_subplot(111, projection="3d")

        xs_idx = [i for i, (axis, step_idx) in enumerate(labels) if axis == "x"]
        ys_idx = [i for i, (axis, step_idx) in enumerate(labels) if axis == "y"]

        draw_gradient_series_3d(ax, z3, xs_idx, labels, axis_name="x", cmap_name="Blues", marker="o")
        draw_gradient_series_3d(ax, z3, ys_idx, labels, axis_name="y", cmap_name="Reds", marker="s")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(
            f"{name_prefix} | PCA of encoder features (3D)\n"
            f"explained variance = "
            f"{pca.explained_variance_ratio_[0]:.3f}, "
            f"{pca.explained_variance_ratio_[1]:.3f}, "
            f"{pca.explained_variance_ratio_[2]:.3f}"
        )
        ax.legend()
        fig.tight_layout()

        Logger.run().log_figure(
            fig,
            f"{name_prefix}_pca_cross_3d",
            dir_name="pca_encoder_cross_3d"
        )
        plt.close(fig)

        # -----------------------------
        # 実空間(xy)も一応そのまま保存
        # -----------------------------
        fig_xy, ax_xy = plt.subplots(figsize=(7, 7), dpi=140)

        def draw_gradient_series_2d(ax, pts, idx_list, labels, axis_name, cmap_name, marker):
            idx_list = sorted(idx_list, key=lambda i: labels[i][1])
            n = len(idx_list)
            colors = make_grad_colors(n, cmap_name)

            for seg_i in range(n - 1):
                i0 = idx_list[seg_i]
                i1 = idx_list[seg_i + 1]
                ax.plot(
                    [pts[i0, 0], pts[i1, 0]],
                    [pts[i0, 1], pts[i1, 1]],
                    color=colors[seg_i + 1],
                    linewidth=2.0,
                    alpha=0.95,
                )

            for local_i, global_i in enumerate(idx_list):
                ax.scatter(
                    pts[global_i, 0],
                    pts[global_i, 1],
                    color=colors[local_i],
                    s=42,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=3,
                )
                _, step_idx = labels[global_i]
                ax.text(
                    pts[global_i, 0],
                    pts[global_i, 1],
                    f"{axis_name}{step_idx}",
                    fontsize=8,
                    alpha=0.9,
                )

            ax.plot([], [], color=colors[-1], marker=marker, label=f"{axis_name}-line")

        draw_gradient_series_2d(ax_xy, coords_np, xs_idx, labels, axis_name="x", cmap_name="Blues", marker="o")
        draw_gradient_series_2d(ax_xy, coords_np, ys_idx, labels, axis_name="y", cmap_name="Reds", marker="s")

        ax_xy.scatter(
            fixed_goal[0], fixed_goal[1],
            s=140,
            marker="X",
            color="green",
            edgecolors="black",
            linewidths=0.5,
            label="goal",
            zorder=5,
        )

        ax_xy.set_xlabel("x")
        ax_xy.set_ylabel("y")
        ax_xy.set_title(f"{name_prefix} | bluebox positions on xy-plane")
        ax_xy.legend()
        ax_xy.grid(True)
        ax_xy.set_aspect("equal", adjustable="box")
        fig_xy.tight_layout()

        Logger.run().log_figure(
            fig_xy,
            f"{name_prefix}_xy_cross_3dsrc",
            dir_name="pca_encoder_cross_3d"
        )
        plt.close(fig_xy)

        print("finished making 3D pca visual encoder")


    @torch.no_grad()
    def pca_visual_encoder_franka_3d(
        self,
        batch,
        jepa: "JEPA",
        name_prefix: str = "",
        idxs: Optional[List[int]] = None,
        notebook: bool = False,
        pool: str = "flat",              # "gap"/"flat"
        k: int = 3,
        align_closed: bool = True,
        is_train: bool = False,
        x_half_range: float = 0.10,
        y_half_range: float = 0.10,
        n_x_points: int = 101,
        n_y_points: int = 101,
        ee_z: float = 0.10,
        fixed_box_pos: tuple = (0.515, 0.0, 0.05),
        fixed_goal_pos: tuple = (0.58, 0.00, 0.05),
        robot_only: bool = True,
        settle_steps: int = 300,
    ):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        def make_grad_colors(n, cmap_name, start=0.25, end=0.95):
            cmap = cm.get_cmap(cmap_name)
            vals = np.linspace(start, end, n)   # 薄い -> 濃い
            return [cmap(v) for v in vals]

        def draw_gradient_series_3d(ax, pts, idx_list, labels, axis_name, cmap_name, marker):
            idx_list = sorted(idx_list, key=lambda i: labels[i][1])
            n = len(idx_list)
            colors = make_grad_colors(n, cmap_name)

            # 線
            for seg_i in range(n - 1):
                i0 = idx_list[seg_i]
                i1 = idx_list[seg_i + 1]
                ax.plot(
                    [pts[i0, 0], pts[i1, 0]],
                    [pts[i0, 1], pts[i1, 1]],
                    [pts[i0, 2], pts[i1, 2]],
                    color=colors[seg_i + 1],
                    linewidth=2.0,
                    alpha=0.95,
                )

            # 点
            for local_i, global_i in enumerate(idx_list):
                ax.scatter(
                    pts[global_i, 0],
                    pts[global_i, 1],
                    pts[global_i, 2],
                    color=colors[local_i],
                    s=42,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.3,
                    depthshade=True,
                )
                _, step_idx = labels[global_i]
                ax.text(
                    pts[global_i, 0],
                    pts[global_i, 1],
                    pts[global_i, 2],
                    f"{axis_name}{step_idx}",
                    fontsize=8,
                )

            # 凡例用ダミー
            ax.plot([], [], [], color=colors[-1], marker=marker, label=f"{axis_name}-line")

        def draw_gradient_series_2d(ax, pts, idx_list, labels, axis_name, cmap_name, marker):
            idx_list = sorted(idx_list, key=lambda i: labels[i][1])
            n = len(idx_list)
            colors = make_grad_colors(n, cmap_name)

            for seg_i in range(n - 1):
                i0 = idx_list[seg_i]
                i1 = idx_list[seg_i + 1]
                ax.plot(
                    [pts[i0, 0], pts[i1, 0]],
                    [pts[i0, 1], pts[i1, 1]],
                    color=colors[seg_i + 1],
                    linewidth=2.0,
                    alpha=0.95,
                )

            for local_i, global_i in enumerate(idx_list):
                ax.scatter(
                    pts[global_i, 0],
                    pts[global_i, 1],
                    color=colors[local_i],
                    s=42,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=3,
                )
                _, step_idx = labels[global_i]
                ax.text(
                    pts[global_i, 0],
                    pts[global_i, 1],
                    f"{axis_name}{step_idx}",
                    fontsize=8,
                )

            ax.plot([], [], color=colors[-1], marker=marker, label=f"{axis_name}-line")

        env_generator = FrankaEnvsGenerator(
            model_path=self.config.model_path,
            n_envs=1,
            max_dq=self.config.max_dq,
            camera_name=self.config.camera_name,
        )

        env = env_generator()[0]
        device = self.device

        print("[pldm/probing/evaluator.py] batch.states.shape:", batch.states.shape)

        # -----------------------------
        # 1) EE の十字配置を作る
        # -----------------------------
        ee_center = np.array([0.515, 0.0, ee_z], dtype=np.float32)
        fixed_box_pos = np.array(fixed_box_pos, dtype=np.float32)
        fixed_goal_pos = np.array(fixed_goal_pos, dtype=np.float32)

        positions = []

        x_values = np.linspace(
            ee_center[0] - x_half_range,
            ee_center[0] + x_half_range,
            n_x_points,
            dtype=np.float32,
        )

        y_values = np.linspace(
            ee_center[1] - y_half_range,
            ee_center[1] + y_half_range,
            n_y_points,
            dtype=np.float32,
        )

        x_center_idx = n_x_points // 2
        for idx, x in enumerate(x_values):
            p = ee_center.copy()
            p[0] = x
            positions.append(("x", idx - x_center_idx, p.copy()))

        y_center_idx = n_y_points // 2
        for idx, y in enumerate(y_values):
            p = ee_center.copy()
            p[1] = y
            positions.append(("y", idx - y_center_idx, p.copy()))

        # -----------------------------
        # 2) 各EE位置で画像を取得
        # -----------------------------
        imgs = []
        labels = []
        coords = []

        for axis, step_idx, ee_target in positions:
            obs = env.reset(
                start_pos=fixed_box_pos,
                goal_pos=fixed_goal_pos,
                robot_only=robot_only,
            )

            try:
                _, ee_actual = env.set_xyz(
                    target_pos=ee_target,
                    target_rotmat=None,
                    rot_weight=0.1,
                    settle_steps=settle_steps,
                    sync_ctrl=True,
                )
            except Exception as e:
                print(f"[WARN] IK failed at {axis}{step_idx}: target={ee_target}, err={e}")
                continue

            obs = env.get_obs()

            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs)

            imgs.append(obs.float().cpu())
            labels.append((axis, step_idx))
            coords.append(ee_actual[:2].copy())   # xyのみ保存

        if len(imgs) == 0:
            raise RuntimeError("No valid samples were collected in pca_visual_encoder_franka_3d().")

        imgs = torch.stack(imgs, dim=0)   # [N,C,H,W]

        # -----------------------------
        # 3) encoder に通すため T=1 にする
        # -----------------------------
        states = imgs.unsqueeze(0).to(device)   # [1,N,C,H,W]

        enc_output = jepa.backbone.forward_multiple(states)
        z = enc_output.obs_component

        # -----------------------------
        # 4) [N,D] に整形
        # -----------------------------
        if z.dim() == 5:
            z = z[0]   # [N,C,H,W]
            if pool == "gap":
                z = z.mean(dim=(-2, -1))   # [N,C]
            else:
                z = z.flatten(start_dim=1) # [N,C*H*W]
        elif z.dim() == 3:
            z = z[0]   # [N,D]
        else:
            raise ValueError(f"Unexpected encoder output shape: {z.shape}")

        z_np = z.detach().cpu().numpy()

        # -----------------------------
        # 5) PCA(3D)
        # -----------------------------
        pca = PCA(n_components=3)
        z3 = pca.fit_transform(z_np)

        # -----------------------------
        # 6) 潜在空間の3D可視化
        # -----------------------------
        fig = plt.figure(figsize=(8, 8), dpi=140)
        ax = fig.add_subplot(111, projection="3d")

        xs_idx = [i for i, (axis, step_idx) in enumerate(labels) if axis == "x"]
        ys_idx = [i for i, (axis, step_idx) in enumerate(labels) if axis == "y"]

        draw_gradient_series_3d(ax, z3, xs_idx, labels, "x", "Blues", "o")
        draw_gradient_series_3d(ax, z3, ys_idx, labels, "y", "Reds", "s")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(
            f"{name_prefix} | PCA of encoder features (Franka EE motion, 3D)\n"
            f"explained variance = "
            f"{pca.explained_variance_ratio_[0]:.3f}, "
            f"{pca.explained_variance_ratio_[1]:.3f}, "
            f"{pca.explained_variance_ratio_[2]:.3f}"
        )
        ax.legend()
        fig.tight_layout()

        Logger.run().log_figure(
            fig,
            f"{name_prefix}_pca_franka_cross_3d",
            dir_name="pca_encoder_cross_3d"
        )
        plt.close(fig)

        # -----------------------------
        # 7) 実空間(xy平面)のEE位置も可視化
        # -----------------------------
        coords_np = np.asarray(coords)   # [N,2]

        fig_xy, ax_xy = plt.subplots(figsize=(7, 7), dpi=140)

        draw_gradient_series_2d(ax_xy, coords_np, xs_idx, labels, "x", "Blues", "o")
        draw_gradient_series_2d(ax_xy, coords_np, ys_idx, labels, "y", "Reds", "s")

        if not robot_only:
            ax_xy.scatter(
                fixed_box_pos[0], fixed_box_pos[1],
                s=120,
                marker="X",
                label="box"
            )

        ax_xy.set_xlabel("x")
        ax_xy.set_ylabel("y")
        ax_xy.set_title(f"{name_prefix} | Franka EE positions on xy-plane")
        ax_xy.legend()
        ax_xy.grid(True)
        ax_xy.set_aspect("equal", adjustable="box")
        fig_xy.tight_layout()

        Logger.run().log_figure(
            fig_xy,
            f"{name_prefix}_xy_franka_cross_3dsrc",
            dir_name="pca_encoder_cross_3d"
        )
        plt.close(fig_xy)

        print("finished making 3D pca visual encoder for franka")





    def _make_cartesian_action_directions(
        self,
        env,
        step_size=0.04,  # 1cm
        directions=("px", "nx", "py", "ny", "zero"),
        rot_weight=0.1,
        settle_steps=0,
    ):
        """
        初期姿勢 q0, ee0 を基準に、
        直交座標方向の微小移動に対応する action ベクトル(7,) を返す。
        action の定義は `batch.actions` と同じ空間に合わせて調整すること。
        """
        q0 = env.physics.data.qpos[:7].copy()
        ee0 = env.get_ee_position().copy()

        action_vecs = []
        labels = []

        delta_map = {
            "px": np.array([+step_size, 0.0, 0.0], dtype=np.float32),
            "nx": np.array([-step_size, 0.0, 0.0], dtype=np.float32),
            "py": np.array([0.0, +step_size, 0.0], dtype=np.float32),
            "ny": np.array([0.0, -step_size, 0.0], dtype=np.float32),
            "zero": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }

        for key in directions:
            if key == "zero":
                q_des = q0.astype(np.float32)
                action_vecs.append(q_des)
                labels.append("zero")
                continue

            target_pos = ee0 + delta_map[key]

            # IK だけ解きたいので set_xyz ではなく calc_inverse_kinematic を直接使う方が軽い
            result = env.calc_inverse_kinematic(
                target_pos,
                target_rotmat=None,
                rot_weight=rot_weight,
            )
            if not result.success:
                print(f"[WARN] IK failed for {key}: target={target_pos}")
                continue

            q_des = result.qpos[:7].copy().astype(np.float32)


            action_vecs.append(q_des)
            labels.append(key)

        return action_vecs, labels


    @torch.no_grad()
    def plot_pca_cartesian_action_rollout(
        self,
        batch,
        jepa: "JEPA",
        name_prefix: str = "",
        idx: int = 0,
        horizon: int = 5,
        pool: str = "gap",
        k: int = 3,
        notebook: bool = False,
        is_train: bool = False,
        step_size: float = 0.04,
    ):
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from sklearn.decomposition import PCA

        device = self.device

        # -----------------------------
        # 1) batch から1サンプルの初期状態を取る
        # -----------------------------
        states = batch.states.to(device).transpose(0, 1)   # [T,B,...]
        actions = batch.actions.to(device).transpose(0, 1) # [T,B,A]
        optional_fields = get_optional_fields(batch, device=states.device)

        obs0 = states[0:1, idx:idx+1]  # [1,1,C,H,W] 想定

        # propio があるなら初期時刻だけ抜く
        opt0 = {}
        for key, val in optional_fields.items():
            if torch.is_tensor(val) and val.shape[0] == states.shape[0] and val.shape[1] == states.shape[1]:
                opt0[key] = val[0:1, idx:idx+1]
            else:
                opt0[key] = val

        # -----------------------------
        # 2) 初期潜在 z0 を作る
        # -----------------------------
        enc0 = jepa.forward_posterior(
            obs0,
            actions=None,
            encode_only=True,
            **opt0,
        ).backbone_output

        # z0 = enc0.obs_component if getattr(enc0, "obs_component", None) is not None else enc0.encodings
        z0 = enc0.encodings

        # -----------------------------
        # 3) 環境を1つ作って EE方向 action を作る
        # -----------------------------
        env_generator = FrankaEnvsGenerator(
            model_path=self.config.model_path,
            n_envs=1,
            max_dq=self.config.max_dq,
            camera_name=self.config.camera_name,
        )
        env = env_generator()[0]

        # 初期姿勢を reset で揃える
        _ = env.reset(robot_only=True)

        action_vecs, labels = self._make_cartesian_action_directions(
            env,
            step_size=step_size,
            directions=("px", "nx", "py", "ny", "zero"),
        )
        
        
        print("labels:", labels)
        for label, a in zip(labels, action_vecs):
            print(label, "norm =", np.linalg.norm(a), "a =", a)

        action_seqs = []
        for a_abs in action_vecs:
            a_abs_t = torch.tensor(a_abs, dtype=torch.float32, device=device)   # (7,)
            a_norm = self.normalizer.normalize_action(a_abs_t)                  # normalized absolute angle
            seq = a_norm.unsqueeze(0).repeat(horizon, 1)                        # [H,7]
            action_seqs.append(seq)

        action_seqs = torch.stack(action_seqs, dim=0)                           # [N,H,7]
        

        # -----------------------------
        # 5) predictor rollout
        #    predictor.forward_multiple(state_encs, actions, T, ...)
        #    を直接使う
        # -----------------------------
        # z0 を predictor が期待する T x B x ... へ揃える
        if z0.dim() == 5:
            # [1,1,C,H,W]
            state_encs = z0
        elif z0.dim() == 3:
            # [1,1,D]
            state_encs = z0
        else:
            raise ValueError(f"Unexpected z0 shape: {z0.shape}")

        rollout_latents = []

        for n in range(action_seqs.shape[0]):
            a_seq = action_seqs[n:n+1].transpose(0, 1).contiguous()  # [H,1,7]

            pred_output = jepa.predictor.forward_multiple(
                state_encs=state_encs,
                actions=a_seq,
                T=horizon,
                compute_posterior=False,
                alpha=0.0,
            )
            

            # lat = pred_output.obs_component if getattr(pred_output, "obs_component", None) is not None else pred_output.predictions
            lat = pred_output.predictions

            rollout_latents.append(lat)

        # -----------------------------
        # 6) [N,H,D] に整形
        # -----------------------------
        def _to_TBD(x, pool="gap"):
            if x.dim() == 5:  # [T,B,C,H,W]
                if pool == "gap":
                    x = x.mean(dim=(-2, -1))  # [T,B,C]
                else:
                    T, B, C, H, W = x.shape
                    x = x.reshape(T, B, C * H * W)
            elif x.dim() > 3:
                T, B = x.shape[:2]
                x = x.reshape(T, B, -1)
            return x

        all_rolls = []
        for lat in rollout_latents:
            lat = _to_TBD(lat, pool=pool)[:, 0, :]   # [H,D]
            all_rolls.append(lat)
        all_rolls = torch.stack(all_rolls, dim=0)    # [N,H,D]

        # z0 も潰す
        z0_vec = _to_TBD(z0, pool=pool)[0, 0, :].unsqueeze(0)  # [1,D]

        # -----------------------------
        # 7) PCA
        # -----------------------------
        Z = torch.cat([
            z0_vec.repeat(all_rolls.shape[0], 1),         # [N,D]
            all_rolls.reshape(-1, all_rolls.shape[-1]),   # [N*H,D]
        ], dim=0).detach().cpu().numpy()

        pca = PCA(n_components=min(k, Z.shape[1]))
        U = pca.fit_transform(Z)
        expl = pca.explained_variance_ratio_
        k_eff = U.shape[1]

        N = all_rolls.shape[0]
        rollout_len = all_rolls.shape[1]
        
        U0 = U[:N]
        Ur = U[N:].reshape(N, rollout_len, k_eff)

        # -----------------------------
        # 8) 2D可視化
        # -----------------------------
        if k_eff >= 2:
            fig, ax = plt.subplots(figsize=(7, 7), dpi=140)

            colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]

            # 始点は1個だけ黒で表示
            ax.scatter(U0[0, 0], U0[0, 1], s=40, color="black", marker="o", label="start")

            for n, label in enumerate(labels):
                traj = Ur[n, :, :2]

                # 軌道
                ax.plot(
                    traj[:, 0], traj[:, 1],
                    marker="o",
                    alpha=0.9,
                    color=colors[n],
                    label=label,
                )

                # 終点を強調
                ax.scatter(
                    traj[-1, 0], traj[-1, 1],
                    s=60,
                    color=colors[n],
                    marker="X"
                )

                # 終点ラベル
                ax.text(
                    traj[-1, 0], traj[-1, 1],
                    label,
                    fontsize=9,
                    color=colors[n],
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[n], alpha=0.8)
                )

            ax.legend()

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(
                f"{name_prefix} | Cartesian-action rollout PCA | "
                + ", ".join(f"{v:.3f}" for v in expl[:k_eff])
            )
            ax.grid(True)
            fig.tight_layout()

            if not notebook:
                if not is_train:
                    Logger.run().log_figure(fig, f"{name_prefix}-cartesian-action-rollout-pca2d", dir_name="pca_ac_cond_val/action_rollout")
                else:
                    Logger.run().log_figure(fig, f"{name_prefix}-cartesian-action-rollout-pca2d", dir_name="pca_ac_cond_train/action_rollout")
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)

        # =========================
        # EE直交座標の可視化
        # =========================
        ee_positions = []
        ee_labels = []

        # 初期状態
        q0 = env.physics.data.qpos[:7].copy()
        ee0 = env.get_ee_position().copy()

        ee_positions.append(ee0)
        ee_labels.append("origin")

        for label, a_abs in zip(labels, action_vecs):
            if label == "zero":
                q_target = q0.copy()
            else:
                q_target = a_abs.copy()

            # 一度関節角をセットしてEE位置を確認
            env.physics.data.qpos[:7] = q_target
            env.physics.forward()

            ee_pos = env.get_ee_position().copy()
            ee_positions.append(ee_pos)
            ee_labels.append(label)

        # 元に戻す
        env.physics.data.qpos[:7] = q0
        env.physics.forward()

        ee_positions = np.array(ee_positions)

        # -------------------------
        # 2Dプロット（xy平面）
        # -------------------------
        fig, ax = plt.subplots(figsize=(6,6), dpi=140)

        origin = ee_positions[0]

        for i in range(1, len(ee_positions)):
            p = ee_positions[i]
            dx = p[0] - origin[0]
            dy = p[1] - origin[1]

            ax.arrow(
                origin[0], origin[1],
                dx, dy,
                head_width=0.005,
                length_includes_head=True,
                alpha=0.8
            )
            ax.text(p[0], p[1], ee_labels[i], fontsize=9)

        # origin
        ax.scatter(origin[0], origin[1], s=50)
        ax.text(origin[0], origin[1], "origin", fontsize=10)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("EE Cartesian Movement (from IK actions)")
        ax.grid(True)
        ax.axis("equal")

        if notebook:
            plt.show()
        else:
            Logger.run().log_figure(fig, f"{name_prefix}-ee_cartesian_debug", dir_name="pca_ac_cond_val/action_log")
            plt.close(fig)


    @torch.no_grad()
    def plot_pca_rgb(
        self,
        btc,
        model,
        name_prefix: str = "",
        idxs: Optional[List[int]] = None,
        notebook: bool = False,
        is_train: bool = False,
        upsample: Optional[Tuple[int, int]] = None,   # 例: (224, 224)
        max_cols: int = 8,
    ):
        """
        上段: 実画像
        下段: encoder の潜在特徴マップを PCA で 3 次元に落として RGB 化した画像
        """
        import gc
        import math
        import itertools
        import numpy as np
        import matplotlib.pyplot as plt
        import torch
        import torch.nn.functional as F
        from sklearn.decomposition import PCA

        device = self.device

        def _minmax01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
            x = x - x.min()
            x = x / (x.max() + eps)
            return x

        def _resize_rgb(rgb_hw3: np.ndarray, out_hw):
            if out_hw is None:
                return rgb_hw3
            out_h, out_w = out_hw
            rgb_t = torch.from_numpy(rgb_hw3).permute(2, 0, 1).unsqueeze(0).float()
            rgb_t = F.interpolate(rgb_t, size=(out_h, out_w), mode="bilinear", align_corners=False)
            rgb = rgb_t[0].permute(1, 2, 0).cpu().numpy()
            return np.clip(rgb, 0.0, 1.0)

        def _best_perm(rgb_hw3: np.ndarray) -> np.ndarray:
            perms = list(itertools.permutations([0, 1, 2]))
            best_img = None
            best_score = -1e18
            for p in perms:
                cand = rgb_hw3[:, :, p]
                score = cand.std(axis=(0, 1)).sum()
                if score > best_score:
                    best_score = score
                    best_img = cand
            return best_img

        def _to_BTCHW(x, T_ref, B_ref):
            if x.shape[0] == T_ref and x.shape[1] == B_ref:
                # [T,B,...] -> [B,T,...]
                x = x.permute(1, 0, 2, 3, 4)
            elif x.shape[0] == B_ref and x.shape[1] == T_ref:
                # [B,T,...]
                pass
            else:
                raise ValueError(
                    f"unexpected shape {x.shape}, expected [T,B,...]=[{T_ref},{B_ref},...] "
                    f"or [B,T,...]=[{B_ref},{T_ref},...]"
                )
            return x.contiguous()

        # --------------------------------------------------
        # 1. states を取得
        # --------------------------------------------------
        states = btc.states.to(device)

        if states.dim() != 5:
            raise ValueError(f"btc.states must be 5D, got shape={tuple(states.shape)}")

        # btc.states は [B,T,C,H,W] 前提
        states_TB = states.transpose(0, 1).contiguous()   # [T,B,C,H,W]
        T_ref, B_ref = states_TB.shape[0], states_TB.shape[1]

        # 可視化用に元画像も [B,T,C,H,W] に揃える
        states_BT = states.contiguous()  # [B,T,C,H,W]

        # --------------------------------------------------
        # 2. encoder 出力を取得
        # --------------------------------------------------
        if hasattr(model, "backbone") and hasattr(model.backbone, "forward_multiple"):
            enc_output = model.backbone.forward_multiple(states_TB)
        elif hasattr(model, "forward_multiple"):
            enc_output = model.forward_multiple(states_TB)
        else:
            raise AttributeError(
                "model.backbone.forward_multiple(...) か model.forward_multiple(...) が必要です。"
            )

        if getattr(enc_output, "obs_component", None) is not None:
            z = enc_output.obs_component
        else:
            raise AttributeError("encoder output に obs_component がありません。")

        print("[plot_pca_rgb] states_TB.shape:", states_TB.shape)
        print("[plot_pca_rgb] z.shape:", z.shape)

        # --------------------------------------------------
        # 3. [B,T,C,H,W] に正規化
        # --------------------------------------------------
        z = _to_BTCHW(z, T_ref, B_ref)
        B, T, C, H, W = z.shape

        if idxs is None:
            idxs = list(range(B))

        # --------------------------------------------------
        # 4. 各 trajectory ごとに PCA を fit して RGB 化
        # --------------------------------------------------
        rgb_maps = {}

        for i in idxs:
            feat_i = z[i]              # [T,C,H,W]
            img_i  = states_BT[i]      # [T,C,H,W]

            # trajectory 全体で共通 PCA
            Z = (
                feat_i.permute(0, 2, 3, 1)   # [T,H,W,C]
                .reshape(T * H * W, C)
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            pca = PCA(n_components=min(3, C))
            Y = pca.fit_transform(Z)   # [T*H*W, 3]
            expl = pca.explained_variance_ratio_

            # 成分ごとに [0,1] 正規化
            Y_norm = np.zeros_like(Y)
            for c_idx in range(Y.shape[1]):
                Y_norm[:, c_idx] = _minmax01(Y[:, c_idx])

            Y_norm = Y_norm.reshape(T, H, W, Y.shape[1])

            # 3次元未満なら 0 埋め
            if Y_norm.shape[-1] < 3:
                pad = np.zeros((T, H, W, 3 - Y_norm.shape[-1]), dtype=Y_norm.dtype)
                Y_norm = np.concatenate([Y_norm, pad], axis=-1)

            frames_rgb = []
            frames_img = []

            for t in range(T):
                # PCA-RGB
                rgb = Y_norm[t]
                rgb = _best_perm(rgb)
                rgb = _resize_rgb(rgb, upsample)
                frames_rgb.append(rgb)

                # 実画像
                img = img_i[t].detach().float().cpu()   # [C,H,W]
                img = img.permute(1, 2, 0).numpy()      # [H,W,C]

                # 画像が正規化済みなら可視化用に 0-1 に戻す
                # いったん min-max で表示
                img_vis = img.copy()
                for ch in range(img_vis.shape[2]):
                    img_vis[:, :, ch] = _minmax01(img_vis[:, :, ch])

                if upsample is not None:
                    img_vis = _resize_rgb(img_vis, upsample)

                frames_img.append(np.clip(img_vis, 0.0, 1.0))

            rgb_maps[i] = np.stack(frames_rgb, axis=0)   # [T,H,W,3]

            # --------------------------------------------------
            # 5. 可視化
            # --------------------------------------------------
            # 2行 x T列
            ncols = T
            fig, axes = plt.subplots(
                2, ncols,
                figsize=(2.2 * ncols, 4.6),
                dpi=140
            )

            if T == 1:
                axes = np.array(axes).reshape(2, 1)

            for t in range(T):
                # 上段: 実画像
                axes[0, t].imshow(frames_img[t])
                axes[0, t].set_title(f"t={t}", fontsize=8)
                axes[0, t].axis("off")

                # 下段: PCA-RGB
                axes[1, t].imshow(frames_rgb[t])
                axes[1, t].axis("off")

            axes[0, 0].set_ylabel("Image", fontsize=10)
            axes[1, 0].set_ylabel("PCA-RGB", fontsize=10)

            fig.suptitle(
                f"{name_prefix} | idx={i} | PCA-RGB | "
                + ", ".join(f"{v:.3f}" for v in expl[:min(3, len(expl))]),
                fontsize=11
            )
            fig.tight_layout()

            if not notebook:
                if not is_train:
                    Logger.run().log_figure(
                        fig,
                        f"{name_prefix}-pca-rgb-i{i}",
                        dir_name="pca_rgb_val"
                    )
                else:
                    Logger.run().log_figure(
                        fig,
                        f"{name_prefix}-pca-rgb-i{i}",
                        dir_name="pca_rgb_train"
                    )
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)

        plt.close("all")
        try:
            del z, enc_output, states, states_TB, states_BT
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        return rgb_maps