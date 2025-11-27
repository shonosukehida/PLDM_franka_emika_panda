from dataclasses import dataclass, field
import enum
from typing import List

from pldm.objectives.vicreg import VICRegObjective, VICRegObjectiveConfig  # noqa
from pldm.objectives.idm import IDMObjective, IDMObjectiveConfig  # noqa
from pldm.objectives.prediction import PredictionObjective, PredictionObjectiveConfig
from pldm.objectives.sigreg import SIGRegObjective, SIGRegObjectiveConfig


class ObjectiveType(enum.Enum):
    VICReg = enum.auto()
    VICRegObs = enum.auto()
    VICRegPropio = enum.auto()
    IDM = enum.auto()
    Prediction = enum.auto()
    PredictionObs = enum.auto()
    PredictionPropio = enum.auto()
    SIGReg = enum.auto()


@dataclass
class ObjectivesConfig:
    clsd_objectives: List[ObjectiveType] = field(default_factory=lambda: [])
    opn_objectives: List[ObjectiveType] = field(default_factory=lambda: [])
    vicreg: VICRegObjectiveConfig = field(default_factory=VICRegObjectiveConfig)
    vicreg_obs: VICRegObjectiveConfig = field(default_factory=VICRegObjectiveConfig)
    vicreg_propio: VICRegObjectiveConfig = field(default_factory=VICRegObjectiveConfig)
    idm: IDMObjectiveConfig = field(default_factory=IDMObjectiveConfig)
    prediction: PredictionObjectiveConfig = field(default_factory=PredictionObjectiveConfig)
    prediction_obs: PredictionObjectiveConfig = field(default_factory=PredictionObjectiveConfig)
    prediction_propio: PredictionObjectiveConfig = field(default_factory=PredictionObjectiveConfig)
    sigreg: SIGRegObjectiveConfig = field(default_factory=SIGRegObjectiveConfig)

    def build_clsd_objectives_list(
        self,
        repr_dim: int,
        name_prefix: str = "",
    ):
        objectives = []
        for objective_type in self.clsd_objectives:
            if objective_type == ObjectiveType.VICReg:
                objectives.append(
                    VICRegObjective(
                        self.vicreg, name_prefix=name_prefix, repr_dim=repr_dim
                    )
                )
            elif objective_type == ObjectiveType.VICRegObs:
                objectives.append(
                    VICRegObjective(
                        self.vicreg_obs,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="obs",
                    )
                )
            elif objective_type == ObjectiveType.VICRegPropio:
                objectives.append(
                    VICRegObjective(
                        self.vicreg_propio,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="propio",
                    )
                )
            elif objective_type == ObjectiveType.IDM:
                objectives.append(
                    IDMObjective(self.idm, name_prefix=name_prefix, repr_dim=repr_dim)
                )
            elif objective_type == ObjectiveType.PredictionObs:
                objectives.append(
                    PredictionObjective(
                        self.prediction_obs,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="obs",
                    )
                )
            elif objective_type == ObjectiveType.PredictionPropio:
                objectives.append(
                    PredictionObjective(
                        self.prediction_propio,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="propio",
                    )
                )
                
            elif objective_type == ObjectiveType.SIGReg:
                objectives.append(
                    SIGRegObjective(
                        self.sigreg,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                    )
                )

            else:
                raise NotImplementedError()
        return objectives
    
    def build_opn_objectives_list(self, repr_dim: int, name_prefix: str = ""):
        objectives = []
        for objective_type in self.closed_objectives:
            if objective_type == ObjectiveType.VICReg:
                objectives.append(
                    VICRegObjective(
                        self.vicreg, name_prefix=name_prefix, repr_dim=repr_dim
                    )
                )
            elif objective_type == ObjectiveType.VICRegObs:
                objectives.append(
                    VICRegObjective(
                        self.vicreg_obs,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="obs",
                    )
                )
            elif objective_type == ObjectiveType.VICRegPropio:
                objectives.append(
                    VICRegObjective(
                        self.vicreg_propio,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="propio",
                    )
                )
            elif objective_type == ObjectiveType.IDM:
                objectives.append(
                    IDMObjective(self.idm, name_prefix=name_prefix, repr_dim=repr_dim)
                )
            elif objective_type == ObjectiveType.PredictionObs:
                objectives.append(
                    PredictionObjective(
                        self.prediction_obs,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="obs",
                    )
                )
            elif objective_type == ObjectiveType.PredictionPropio:
                objectives.append(
                    PredictionObjective(
                        self.prediction_propio,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="propio",
                    )
                )
            elif objective_type == ObjectiveType.Prediction:
                objectives.append(
                    PredictionObjective(
                        self.prediction,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="state",  
                    )
                )
                
            elif objective_type == ObjectiveType.SIGReg:
                objectives.append(
                    SIGRegObjective(
                        self.sigreg,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                    )
                )

            else:
                raise NotImplementedError()
        return objectives
