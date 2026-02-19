from typing import Optional, Callable, List

import torch
from torch import nn

from .mppi_torch import MPPI
from pldm.models.utils import flatten_conv_output
from .planner import PlanningResult


class LearnedDynamics:
    def __init__(self, model, state_dim=None):
        self.model = model
        self.dump_dict = None
        self.state_dim = state_dim
        self.max_batch_size = 500

    def __call__(self, state, action, only_return_last=True, flatten_output=True):
        """
        state: [K x nx]
        action: [K x nx]
        """
        
        # print("[DBG][pldm/planning/planners/cem_planner.py][LearnedDynamics] state.shape:", tuple(state.shape), "action.shape:", tuple(action.shape))
        # print("[DBG][pldm/planning/planners/cem_planner.py][LearnedDynamics] state_dim:", self.state_dim)

        # make sure state is in correct format
        og_shape = state.shape
        n_samples = og_shape[0]

        if isinstance(self.state_dim, int):
            self.state_dim = (self.state_dim,)

        new_shape = (n_samples, *self.state_dim)
        state = state.view(new_shape)

        # introduce time dimension to action if needed
        if len(action.shape) < 3:
            action = action.unsqueeze(0)

        T = action.shape[0]

        if self.model.config.action_dim:
            pred_output = self.model.predictor.forward_multiple(
                state.unsqueeze(0),
                action.float(),
                T,
            )
        else:
            pred_output = self.model.predictor.forward_multiple(
                state.unsqueeze(0),
                actions=None,
                T=T,
                latents=action.float(),
            )
        
        preds = pred_output.predictions
        pred_obs = pred_output.obs_component
        pred_propio = pred_output.propio_component

        if flatten_output:
            preds = flatten_conv_output(preds)  # required for 3rd party MPPI code...
            pred_obs = flatten_conv_output(pred_obs)
            pred_propio = flatten_conv_output(pred_propio)

        if only_return_last:
            preds = preds[-1]
            pred_obs = pred_obs[-1]
            pred_propio = pred_propio[-1]
        

        # we need to return both. preds is used to propagate the state forward. pred_obs is used to take cost
        return preds, pred_obs, pred_propio

    def before_planning_callback(self):
        self.orig_training_state = self.model.training
        self.model.train(False)

    def after_planning_callback(self):
        self.model.train(self.orig_training_state)


class RunningCost:
    def __init__(
        self, 
        objective, 
        idx=None, 
        obs_projector=None, 
        propio_projector=None,
        obs_coeff = 1.0,
        propio_coeff = 0.0,
        ):
        
        self.objective = objective
        self.idx = idx
        self.obs_projector = nn.Identity() if obs_projector is None else obs_projector
        self.propio_projector = nn.Identity() if propio_projector is None else propio_projector
        self.obs_coeff = obs_coeff
        self.propio_coeff = propio_coeff
        print("[DBG][pldm/planning/planners/cem_planner.py] self.obs_coeff:", self.obs_coeff)
        print("[DBG][pldm/planning/planners/cem_planner.py] self.propio_coeff:", self.propio_coeff)
        
        

    def __call__(self, state_obs, state_propio=None, action=None):
        objective = self.objective
        target_obs = objective.target_enc[self.idx]

        state_obs = flatten_conv_output(self.obs_projector(state_obs))
        target_obs = flatten_conv_output(self.obs_projector(target_obs))

        obs_diff = (state_obs - target_obs).pow(2).mean(dim=1)

        if state_propio is not None:
            target_propio = objective.target_propio_enc[self.idx]
            state_propio = flatten_conv_output(self.propio_projector(state_propio))
            target_propio = flatten_conv_output(self.propio_projector(target_propio))
            propio_diff = (state_propio - target_propio).pow(2).mean(dim=1)
        else:
            propio_diff = torch.zeros_like(obs_diff)

        return self.obs_coeff * obs_diff + self.propio_coeff * propio_diff
