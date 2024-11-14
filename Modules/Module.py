# --------------------------------------------------------------------------------------------------
# ------------------------------- Base Module (Neural-Network) Class -------------------------------
# --------------------------------------------------------------------------------------------------
from safetensors.torch import save_file
from safetensors.torch import load_file

import torch.optim.lr_scheduler as schedulers
import torch.optim as optimizers
import torch.nn as networks
import torch.nn.init as initializers

import pathlib
import torch
from torch.nn import BatchNorm1d


# --------------------------------------------------------------------------------------------------
# -------------------------- CLASS :: Base Module (Neural-Network) Class ---------------------------
# --------------------------------------------------------------------------------------------------
class Module(networks.Module):
    """The (abstract) base class from which other modules in the project inherit. Serves to define
    the save and load functions as well as weight initialization. Pay potentially house data
    visualization methods in the future."""


    # ------------------------------------------------------------------------------------------
    # ------------------------------- CONSTRUCTOR :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        super(Module, self).__init__()


    # ----------------------------------------------------------------------------------------------
    # ---------------------------------- METHOD :: Load the Model ----------------------------------
    # ----------------------------------------------------------------------------------------------
    def load(self, path: str | pathlib.Path) -> None:
        self.load_state_dict(load_file(path))


    # ----------------------------------------------------------------------------------------------
    # ---------------------------------- METHOD :: Save the Model ----------------------------------
    # ----------------------------------------------------------------------------------------------
    def save(self, path: str | pathlib.Path) -> None:
        save_file(self.state_dict(), path)


    # ----------------------------------------------------------------------------------------------
    # -------------------------- METHOD :: Initialize the Model's Weights --------------------------
    # ----------------------------------------------------------------------------------------------
    def init(self) -> None:

        for module in self.modules():

            match type(module):

                case networks.Conv2d():

                    initializers.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        initializers.zeros_(module.bias)

                case networks.Linear():

                    init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        initializers.zeros_(module.bias)

                case networks.BatchNorm2d() | networks.BatchNorm1d():

                    initializers.ones_(module.weight)
                    initializers.zeros_(module.bias)