import multiprocessing
import warnings

warnings.filterwarnings("ignore", message="Ill-formed record")

from typing import Optional
import os
import datetime
import shutil
from dataclasses import dataclass, field
import dataclasses
import random
import time
from omegaconf import MISSING

import torch
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import matplotlib
from pldm.logger import Logger, MetricTracker

try:
    multiprocessing.set_start_method("fork")  # noqa
except:
    pass

from pldm.configs import ConfigBase
from pldm.data.enums import DataConfig
from pldm.data.dataset_factory import DatasetFactory
from pldm.data.utils import get_optional_fields
from pldm.optimizers.schedulers import Scheduler, LRSchedule
from pldm.optimizers.optimizer_factory import OptimizerFactory, OptimizerType
from pldm.evaluation.evaluator import EvalConfig, Evaluator

# if "AMD" not in torch.cuda.get_device_name(0):

from pldm.models.hjepa import HJEPA, HJEPAConfig

from pldm.objectives import ObjectivesConfig
import pldm.utils as utils

from pldm.objectives.idm import IDMObjective

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class TrainConfig(ConfigBase):
    device: int = 0
    env_name: str = MISSING
    n_steps: int = 17
    val_n_steps: int = 17
    l1_n_steps: int = 17
    wandb: bool = True
    run_name: Optional[str] = None
    run_group: Optional[str] = None
    run_project: str = "PLDM"
    output_root: Optional[str] = None
    output_dir: Optional[str] = None
    eval_mpcs: int = 20
    quick_debug: bool = False
    seed: int = 42
    load_checkpoint_path: Optional[str] = None
    load_l1_only: bool = False
    eval_only: bool = False
    train_only: bool = False
    epochs: int = 100
    base_lr: float = 0.2
    disable_l2: bool = True
    optimizer_type: OptimizerType = OptimizerType.LARS
    optimizer_schedule: LRSchedule = LRSchedule.Cosine

    data: DataConfig = field(default_factory=DataConfig)

    objectives_l1: ObjectivesConfig = field(default_factory=ObjectivesConfig)

    eval_at_beginning: bool = False
    eval_during_training: bool = False

    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 20

    hjepa: HJEPAConfig = field(default_factory=HJEPAConfig)

    resume_if_possible: bool = True
    compile_model: bool = True

    eval_cfg: EvalConfig = field(default_factory=EvalConfig)
    
    use_closed_loss_func: bool = False

    def __post_init__(self):
        if self.quick_debug:
            self.data.quick_debug = True

            # Wall stuff
            self.data.dot_config.size = self.data.dot_config.batch_size
            self.data.wall_config.size = self.data.wall_config.batch_size
            self.eval_cfg.wall_planning.n_envs = 7
            self.eval_cfg.wall_planning.n_steps = 4
            self.eval_cfg.wall_planning.level1.sgd.n_iters = 2
            self.data.offline_wall_config.lazy_load = True

            # D4RL stuff
            self.data.d4rl_config.quick_debug = True
            self.data.d4rl_config.num_workers = 1
            self.eval_cfg.d4rl_planning.n_envs = 5
            self.eval_cfg.d4rl_planning.n_envs_batch_size = 2
            self.eval_cfg.d4rl_planning.replan_every = 1
            self.eval_cfg.d4rl_planning.n_steps = 6
            self.eval_cfg.d4rl_planning.plot_every = 1

        # Wall stuff
        self.eval_cfg.wall_planning.fix_wall = self.data.wall_config.fix_wall
        self.data.dot_config.n_steps = self.n_steps
        self.data.wall_config.n_steps = self.n_steps
        self.eval_cfg.wall_planning.padding = self.data.wall_config.border_wall_loc

        # D4RL stuff
        if self.hjepa.level1.backbone.arch in ["resnet18", "menet5"]:
            self.eval_cfg.d4rl_planning.image_obs = True
        # assert (
        #     self.eval_cfg.d4rl_planning.plot_every
        #     % self.eval_cfg.d4rl_planning.replan_every
        #     == 0
        # )
        self.eval_cfg.d4rl_planning.stack_states = self.data.d4rl_config.stack_states
        self.eval_cfg.d4rl_planning.img_size = self.data.d4rl_config.img_size

        # general
        self.val_n_steps = self.n_steps
        self.eval_cfg.eval_l2 = not self.hjepa.disable_l2
        
        
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.output_path = os.path.join(
            self.output_root.rstrip("/"), f'{self.output_dir.lstrip("/")}_{self.timestamp}'
        )
        self.run_group = self.output_dir

        if self.train_only:
            self.eval_cfg.eval_l1 = False
            self.eval_cfg.probe_preds = False
            self.eval_cfg.probe_encoder = False
            self.eval_cfg.disable_planning = True

        if "test" in self.output_dir:
            test_dir = os.path.join(self.output_root.rstrip("/"), "test")
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

        self.objectives_l1.idm.action_dim = self.hjepa.level1.action_dim


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        print('use_closed_loss:',self.config.use_closed_loss_func)

        print(f"Logger output path: {self.config.output_path}")
        Logger.run().initialize(
            output_path=self.config.output_path,
            wandb_enabled=self.config.wandb,
            project=config.run_project,
            name=f"{config.run_name}_{config.timestamp}",
            group=config.run_group,
            config=dataclasses.asdict(config),
        )

        seed_everything(config.seed)

        self.sample_step = 0
        self.epoch = 0
        self.step = 0

        # create data
        datasets = DatasetFactory(
            config.data,
            probing_cfg=config.eval_cfg.probing,
            disable_l2=config.hjepa.disable_l2,
        ).create_datasets()

        self.datasets = datasets

        self.ds = datasets.ds
        print('self.ds.normalizer:', self.ds.normalizer)
        print('self.ds.normalizer.state_dict():', self.ds.normalizer.state_dict())
        self.val_ds = datasets.val_ds

        # infer obs shape
        sample_data = next(iter(self.ds))
        
        # # propio_pos/vel の正規化状態を確認
        # print("=== propio_pos sample ===")
        # print(sample_data.propio_pos[0])  # shape: [T, 7]
        # print("mean:", sample_data.propio_pos[0].mean().item())
        # print("std :", sample_data.propio_pos[0].std().item())

        # print("=== propio_vel sample ===")
        # print(sample_data.propio_vel[0])  # shape: [T, 7]
        # print("mean:", sample_data.propio_vel[0].mean().item())
        # print("std :", sample_data.propio_vel[0].std().item())
        input_dim = sample_data.states.shape[2:]

        if len(input_dim) == 1:
            input_dim = input_dim[0]

        # check if proprioceptive states are used
        use_propio_pos = (
            hasattr(sample_data, "propio_pos")
            and sample_data.propio_pos is not None
            and bool(sample_data.propio_pos.shape[-1])
        )
        use_propio_vel = (
            hasattr(sample_data, "propio_vel")
            and sample_data.propio_vel is not None
            and bool(sample_data.propio_vel.shape[-1])
        )
        print("use_propio_pos:", use_propio_pos)
        print("use_propio_vel:", use_propio_vel)


        self.device = torch.device(f'cuda:{self.config.device_id}' if torch.cuda.is_available() else 'cpu')
        
        # create model
        self.model = HJEPA(
            config.hjepa,
            input_dim=input_dim,
            normalizer=self.ds.normalizer,
            use_propio_pos=use_propio_pos,
            use_propio_vel=use_propio_vel,
        ).to(self.device)

        self.model = self.model.to(self.device)

        # create open objectives
        self.open_objectives_l1 = self.config.objectives_l1.build_open_objectives_list(
            name_prefix="l1", repr_dim=self.model.level1.spatial_repr_dim
        )
        # create closed objectives
        self.closed_objectives_l1 = None
        if self.config.use_closed_loss_func:
            self.closed_objectives_l1 = self.config.objectives_l1.build_closed_objectives_list(
                name_prefix="l1", repr_dim=self.model.level1.spatial_repr_dim
            )
        # other stuff...

        load_result = self.maybe_load_model()

        if (
            config.eval_only
            and not config.eval_cfg.probing.full_finetune
            and not load_result
        ):
            print("WARN: probing a random network. Is that intentional?")

        assert not (self.config.hjepa.train_l1 and self.config.hjepa.freeze_l1)

        if self.config.hjepa.freeze_l1:
            print("freezing first level weights")
            for m in self.model.level1.modules():
                for p in m.parameters():
                    p.requires_grad = False

        print(self.model)
        self.n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("number of params:", self.n_parameters)

        l1_predictor_n_parameters = sum(
            p.numel()
            for p in self.model.level1.predictor.parameters()
            if p.requires_grad
        )
        print("number of l1 predictor params:", l1_predictor_n_parameters)

        l1_backbone_n_parameters = sum(
            p.numel()
            for p in self.model.level1.backbone.parameters()
            if p.requires_grad
        )
        print("number of l1 backbone params:", l1_backbone_n_parameters)

        Logger.run().log_summary(
            {
                "n_params": self.n_parameters,
            }
        )

        self.metric_tracker = MetricTracker(window_size=100)

        if self.config.compile_model:
            print("compiling model")
            c_time = time.time()
            self.model = torch.compile(self.model)
            print(f"compilation finished after {time.time() - c_time:.3f}s")

    def maybe_resume(self):
        if not os.path.exists(self.config.output_path):
            return False
        latest_checkpoint = utils.pick_latest_model(self.config.output_path)
        if latest_checkpoint is None:
            return False
        print("resuming from", latest_checkpoint)
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.sample_step = checkpoint["sample_step"]
        print("resumed from epoch", self.epoch, "step", self.step)

    def maybe_load_model(self):
        if self.config.load_checkpoint_path is not None:
            checkpoint = torch.load(self.config.load_checkpoint_path, map_location=self.device, )
            state_dict = checkpoint["model_state_dict"]
            # remove "_orig_mod." prefix from the keys
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            # remove all posterior parameters (incompatible because we don't
            # use it in l1).

            if (
                "backbone.layer1.0.weight" in state_dict
            ):  # this is jepa only model (legacy)
                res = self.model.level1.load_state_dict(state_dict)
            else:
                if self.config.load_l1_only:
                    for k in list(state_dict.keys()):
                        # 1. remove all posterior parameters
                        # (incompatible because we don't use it in l1).
                        # 2. remove everything belonging to l2
                        if "decoder" in k:  # this is for loading RSSM
                            del state_dict[k]
                res = self.model.load_state_dict(state_dict, strict=False)
            assert (
                len(res.unexpected_keys) == 0
            ), f"Unexpected keys when loading weights: {res.unexpected_keys}"
            print(f"loaded model from {self.config.load_checkpoint_path}")

            # IDM のパラメータを読み込む
            if "idm_open_state_dicts" in checkpoint:
                for obj in self.open_objectives_l1:
                    if isinstance(obj, IDMObjective):
                        state = checkpoint["idm_open_state_dicts"].get(obj.name_prefix, None)
                        if state is not None:
                            obj.action_predictor.load_state_dict(state)
                            print(f"✅ open-IDM loaded for {obj.name_prefix}")
                        else:
                            print(f"⚠️ No open-IDM weights found for {obj.name_prefix}")

            if "idm_closed_state_dicts" in checkpoint:
                if self.closed_objectives_l1 is not None:
                    for obj in self.closed_objectives_l1:
                        if isinstance(obj, IDMObjective):
                            state = checkpoint["idm_closed_state_dicts"].get(obj.name_prefix, None)
                            if state is not None:
                                obj.action_predictor.load_state_dict(state)
                                print(f"✅ closed-IDM loaded for {obj.name_prefix}")
                            else:
                                print(f"⚠️ No closed-IDM weights found for {obj.name_prefix}")
            if "normalizer" in checkpoint:
                self.ds.normalizer.load_state_dict(checkpoint["normalizer"])
                print("✅ Normalizer loaded!")
            else:
                print("⚠️ No normalizer found in checkpoint.")

            return True
        return False

    def train(self):
        self.optimizer = OptimizerFactory(
            model=self.model,
            optimizer_type=self.config.optimizer_type,
            base_lr=self.config.base_lr,
        ).create_optimizer()

        if self.config.resume_if_possible:
            if self.maybe_resume():
                print("resuming training")

        scheduler = Scheduler(
            schedule=self.config.optimizer_schedule,
            base_lr=self.config.base_lr,
            data_loader=self.ds,
            epochs=self.config.epochs,
            optimizer=self.optimizer,
        )

        first_step = None

        if self.config.eval_at_beginning and not self.config.quick_debug:
            self.validate()

        for epoch in tqdm(range(self.epoch, self.config.epochs + 1), desc="Epoch"):
            self.epoch = epoch
            end_time = time.time()
            for step, batch in (
                pbar := tqdm(
                    enumerate(self.ds, start=epoch * len(self.ds)),
                    desc="Batch",
                    total=len(self.ds),
                    maxinterval=10,
                )
            ):
                if first_step is None:
                    first_step = step
                start_time = time.time()
                if end_time is not None:
                    # data time is the time it took to load the data
                    # (which is the time between the end of the previous
                    # batch and the start of this batch)
                    data_time = start_time - end_time
                else:
                    data_time = None

                # move to cuda and swap batch and time
                s = batch.states.to(self.device).transpose(0, 1)
                a = batch.actions.to(self.device).transpose(0, 1)
                print('====state====')
                print('s.type:', type(s))
                print('s.shape:', s.shape)
                print('s.mean:', s.mean())
                print('s.std:', s.std())
                print('s.min:', s.min())
                print('s.max:', s.max())
                print('====action====')
                print('a.type:', type(a))
                print('a.shape:', a.shape)
                print('a.mean:', a.mean())
                print('a.std:', a.std())
                print('a.min:', a.min())
                print('a.max:', a.max())

                lr = scheduler.adjust_learning_rate(step)

                self.sample_step += s.shape[1]
                self.step = step

                self.optimizer.zero_grad()

                optional_fields = get_optional_fields(batch, device=s.device)
                # print('confirm optional_fields:',optional_fields["propio_pos"].shape, optional_fields["propio_vel"].shape)
                
                
                with torch.no_grad():
                    # closed_output = self.model.level1.forward_closed(
                    #     input_states=s,  # [T+1, B, C, H, W]
                    #     actions=a,       # [T, B, A]
                    #     propio_pos=optional_fields.get("propio_pos", None),
                    #     propio_vel=optional_fields.get("propio_vel", None),
                    # )

                    # if self.closed_objectives_l1 is not None:
                    #     closed_loss_infos = [
                    #         objective(batch, [closed_output])
                    #         for objective in self.closed_objectives_l1
                    #     ]
                    # else:
                    #     closed_loss_infos = [
                    #         objective(batch, [closed_output])
                    #         for objective in self.open_objectives_l1
                    #     ]
                    # closed_total_loss = sum(info.total_loss for info in closed_loss_infos)

                    # Logger.run().log(
                    #     {
                    #     "closed_loop_loss": closed_total_loss.item(),
                    #     "custom_step": step,
                    #     "epoch": epoch,
                    
                    #     }, 
                    #     commit=False, 
                    #     )
                    # Logger.run().commit()
                    
                    
                    #実際のopen_forward (のはず)
                    open_output = self.model.forward_open(
                        input_states=s,  # [T, B, C, H, W] = [15, 64, 3, 64, 64]
                        actions=a,       # [T - 1, B, D] = [14, 64, 2]
                        propio_pos=optional_fields.get("propio_pos", None), #[T, B, D]
                        propio_vel=optional_fields.get("propio_vel", None), #[T, B, D]
                        
                    )
                    
                    open_loss_infos = [
                        objective(batch, [open_output.level1])
                        for objective in self.open_objectives_l1
                    ]
                    open_total_loss = sum(info.total_loss for info in open_loss_infos)
                    
                    
                    Logger.run().log(
                        {
                        "open_loop_loss": open_total_loss.item(),
                        "custom_step": self.step,
                        "epoch": epoch,
                        },
                                     )
                    Logger.run().commit()
                
                
                forward_result = self.model.forward_posterior(s.to(self.device), a.to(self.device), **optional_fields)
                loss_infos = []
                if self.config.hjepa.train_l1:
                    loss_infos += [
                        objective(batch, [forward_result.level1])
                        for objective in self.open_objectives_l1
                    ]
                for i, loss in enumerate(loss_infos):
                    if loss is None:
                        print(f"[WARNING] loss_infos[{i}] is None → skipping this objective")

                print("loss_infos:", loss_infos)
                total_loss = sum([loss_info.total_loss for loss_info in loss_infos if loss_info is not None])
                if total_loss.isnan():
                    raise RuntimeError("NaN loss")
                total_loss.backward()
                self.optimizer.step()
                self.model.update_ema()  # if ema is enabled, update ema encoder
                
                self.validate_loss_val_ds() 

                train_time = time.time() - start_time
                log_start_time = time.time()

                self.metric_tracker.update("train_time", train_time)
                self.metric_tracker.update("data_time", data_time)

                if self.config.quick_debug or (step % 100 == 0) or True:
                    metric_log = self.metric_tracker.build_log_dict()
                    pbar.set_description(
                        f"Loss: {total_loss.item():.4f}, "
                        f"train: {metric_log['train_time/mean']:.3f}s, "
                        f"data: {metric_log['data_time/mean']:.3f}s, "
                        f"log: {metric_log['log_time/mean'] if 'log_time/mean' in metric_log else 0:.3f}s"  # noqa
                    )
                    log_dict = {}

                    for loss_info in loss_infos:
                        if hasattr(loss_info, "build_log_dict"):
                            log_dict.update(loss_info.build_log_dict())

                    if data_time is not None:
                        log_dict["data_time"] = data_time

                    Logger.run().log(
                        {
                            "sample_step": self.sample_step,
                            "loss": total_loss.item(),
                            "learning_rate": lr,
                            "custom_step": step,
                            "epoch": epoch,
                            **log_dict,
                            **metric_log,
                        },
                        commit=False,
                    )
                    Logger.run().commit()

                    if step - first_step == 5 and False:
                        return

                self.metric_tracker.update("log_time", time.time() - log_start_time)
                end_time = time.time()

            if (
                self.epoch % self.config.save_every_n_epochs == 0 and self.epoch > 0
            ) or self.epoch >= self.config.epochs:
                self.save_model()

            if (
                self.epoch % self.config.eval_every_n_epochs == 0
                and self.config.eval_during_training
            ) or self.epoch >= self.config.epochs:
                self.validate()

    @torch.no_grad()
    def eval_on_objectives(self):
        if self.val_ds is None:
            return

        losses = {}

        for step, batch in tqdm(enumerate(self.val_ds)):
            # move to cuda and swap batch and time
            s = batch.states.to(self.device).transpose(0, 1)
            a = batch.actions.to(self.device).transpose(0, 1)

            optional_fields = get_optional_fields(batch, device=s.device)

            forward_result = self.model.forward_posterior(s.to(self.device), a.to(self.device), **optional_fields)

            loss_infos = []

            if self.config.hjepa.train_l1:
                loss_infos += [
                    objective(batch, [forward_result.level1])
                    for objective in self.open_objectives_l1
                ]

            for loss_info in loss_infos:
                for attr in loss_info._fields:
                    val = getattr(loss_info, attr)
                    if isinstance(val, str):
                        continue

                    assert isinstance(val, torch.Tensor)
                    assert len(val.shape) == 0

                    key = f"val_epoch_{self.epoch}_{loss_info.name_prefix}/{loss_info.loss_name}_{attr}"

                    if key in losses:
                        losses[key].append(val.item())
                    else:
                        losses[key] = [val.item()]

            if step > 2:
                break

        # take mean over batches
        for key, val in losses.items():
            losses[key] = sum(val) / len(val)

        Logger.run().log(losses)
        Logger.run().commit()

    def validate(self):
        training = self.model.training
        self.model.eval()

        # evals on the same objectives used for training
        self.eval_on_objectives()

        # create evaluator (for both probing and planning)
        self.evaluator = Evaluator(
            config=self.config.eval_cfg,
            model=self.model,
            quick_debug=self.config.quick_debug,
            normalizer=self.ds.normalizer,
            epoch=self.epoch,
            probing_datasets=self.datasets.probing_datasets,
            l2_probing_datasets=self.datasets.l2_probing_datasets,
            objectives_l1=self.config.objectives_l1,
            load_checkpoint_path=self.config.load_checkpoint_path,
            output_path=self.config.output_path,
            data_config=self.config.data.wall_config,  # TODO: refactor name to data_config
        )

        log_dict = self.evaluator.evaluate()
        log_dict["custom_step"] = self.step

        Logger.run().log(log_dict)
        Logger.run().log_summary(log_dict)
        Logger.run().save_summary(
            f"summary_epoch={self.epoch}_sample_step={self.sample_step}.json"
        )

        for v in log_dict.values():
            if isinstance(v, matplotlib.figure.Figure):
                plt.close(v)

        if training:
            # if model is previously in training
            self.model.train()

        return

    def save_model(self):
        if self.config.output_path is not None:
            os.makedirs(self.config.output_path, exist_ok=True)
            
            # IDM の action_predictor を含むすべての open objectives を対象に保存
            idm_open_state_dicts = {}
            for obj in self.open_objectives_l1:
                if isinstance(obj, IDMObjective):
                    idm_open_state_dicts[obj.name_prefix] = obj.action_predictor.state_dict()
            
            idm_closed_state_dicts = {}
            if self.closed_objectives_l1 is not None:
                for obj in self.closed_objectives_l1:
                    if isinstance(obj, IDMObjective):
                        idm_closed_state_dicts[obj.name_prefix] = obj.action_predictor.state_dict()
                    
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                    "step": self.step,
                    "sample_step": self.sample_step,
                    "idm_open_state_dicts": idm_open_state_dicts,
                    "idm_closed_state_dicts": idm_closed_state_dicts,
                    "normalizer": self.ds.normalizer.state_dict(), 
                },
                os.path.join(
                    self.config.output_path,
                    f"epoch={self.epoch}_sample_step={self.sample_step}.ckpt",
                ),
            )
            
    #val-loss の可視化
    @torch.no_grad()
    def validate_loss_val_ds(self):
        if self.val_ds is None:
            return
        self.model.eval()
        open_losses = []
        closed_losses = []

        for step, batch in enumerate(self.val_ds):
            s = batch.states.to(self.device).transpose(0, 1)
            a = batch.actions.to(self.device).transpose(0, 1)
            optional_fields = get_optional_fields(batch, device=s.device)

            # official - loop
            open_result = self.model.forward_posterior(s, a, **optional_fields)
            open_loss_infos = [
                obj(batch, [open_result.level1]) for obj in self.open_objectives_l1
            ]
            open_total_loss = sum(info.total_loss.item() for info in open_loss_infos)
            open_losses.append(open_total_loss)

            # Closed-loop
            # closed_result = self.model.level1.forward_closed(
            #     input_states=s,
            #     actions=a,
            #     propio_pos=optional_fields.get("propio_pos", None),
            #     propio_vel=optional_fields.get("propio_vel", None),
            # )
            # if self.closed_objectives_l1 is not None:
            #     closed_loss_infos = [
            #         obj(batch, [closed_result]) for obj in self.closed_objectives_l1
            #     ]
            # else:
            #     closed_loss_infos = [
            #         obj(batch, [closed_result]) for obj in self.open_objectives_l1
            #     ]
            # closed_total_loss = sum(info.total_loss.item() for info in closed_loss_infos)
            # closed_losses.append(closed_total_loss)

            if self.config.quick_debug or step >= 2:
                break

        Logger.run().log(
            {
            "val_loop_loss": np.mean(open_losses),
            # "val_closed_loop_loss": np.mean(closed_losses),
            "val_epoch": self.epoch,
            "val_sample_step": self.sample_step,
            "val_step": self.step, 
            },
            )
        Logger.run().commit()


def main(config: TrainConfig):
    torch.set_num_threads(1)
    trainer = Trainer(config)

    if config.eval_only and not config.quick_debug:
        trainer.validate()
    else:
        trainer.train()

    if config.quick_debug:
        trainer.validate()


if __name__ == "__main__":
    try:
        cfg = TrainConfig.parse_from_command_line()
        main(cfg)
    except Exception as e:
        import traceback
        print("🔥 TRAINING CRASHED")
        traceback.print_exc()
