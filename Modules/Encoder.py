# --------------------------------------------------------------------------------------------------
# -------------------------------------- Style Vector Encoder --------------------------------------
# --------------------------------------------------------------------------------------------------
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torch.optim as optimizers
import torch.nn.init as initializers
import torch.nn.functional as functions
import torch.nn as networks

import typing
import torch

from . Module import Module


# --------------------------------------------------------------------------------------------------
# -------------------------- CLASS :: Base Module (Neural-Network) Class ---------------------------
# --------------------------------------------------------------------------------------------------
class Encoder(Module):
    """The Style-Encoder class. Batches of pairs of images are provided to allow for both positive
    and negative Euclidean distance loss. Vectors are also rewarded for their unit-vector length,
    and KL-Divergence is used to promote a smoother embedding space.

    Batches of images as passed through four blocks of two convolutional layers each with
    bilinear down-sampling between them. Resulting feature maps are global-average-pooled to
    produce vectors identifying each image's style."""

    # ------------------------------------------------------------------------------------------
    # ------------------------------- CONSTRUCTOR :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self, max_lr: float, min_lr: float = 1e-1, T_0: int = 256) -> None:
        super(Encoder, self).__init__()

        # downscaling function, activation function, and dropout layer
        self.dsize = networks.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.funct = networks.ReLU()
        self.noise = networks.Dropout(0.1)


        self.conv1 = networks.Conv2d(3,    128, kernel_size=3, stride=1, padding=1)
        self.norm1 = networks.BatchNorm2d(num_features=128)
        self.conv2 = networks.Conv2d(128,  128, kernel_size=5, stride=1, padding=1)
        self.norm2 = networks.BatchNorm2d(num_features=128)
        self.conv3 = networks.Conv2d(128,  128, kernel_size=3, stride=1, padding=1)
        self.norm3 = networks.BatchNorm2d(num_features=128)
        self.conv4 = networks.Conv2d(128,  128, kernel_size=5, stride=1, padding=1)
        self.norm4 = networks.BatchNorm2d(num_features=128)
        self.conv5 = networks.Conv2d(128,  128, kernel_size=3, stride=1, padding=1)
        self.norm5 = networks.BatchNorm2d(num_features=128)
        self.conv6 = networks.Conv2d(128,  128, kernel_size=5, stride=1, padding=1)
        self.norm6 = networks.BatchNorm2d(num_features=128)
        self.conv7 = networks.Conv2d(128,  128, kernel_size=3, stride=1, padding=1)
        self.norm7 = networks.BatchNorm2d(num_features=128)
        self.conv8 = networks.Conv2d(128,  128, kernel_size=5, stride=1, padding=1)
        self.norm8 = networks.BatchNorm2d(num_features=128)

        # converts a feature-tensor to a feature-vector
        self.final = networks.AdaptiveAvgPool2d(1)

        # mean and log-variance for reparameterization
        self.mu = networks.Linear(128, 128)
        self.lv = networks.Linear(128, 128)

        # scaling factors for the various losses
        self.beta_identical = 1e2
        self.beta_different = 1e2
        self.beta_deviation = 1e2
        self.beta_magnitude = 1e2

        self.optimizer = torch.optim.Adam(self.parameters(), lr=max_lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, eta_min=min_lr)



    # ------------------------------------------------------------------------------------------
    # --------------------------------- METHOD :: Forward Pass ---------------------------------
    # ------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Four blocks of two convolutional layers with bilinear downscaling between them, then
        output is squeezed into a batch of vectors. Downscaled from 512x512 to 64x64."""

        # 1st block: 512x512 pixels
        x = self.noise(self.funct(self.norm1(self.conv1(x)))) # 3x3: 512x512x128 -> 512x512x128
        x = self.noise(self.funct(self.norm2(self.conv2(x)))) # 5x5: 512x512x128 -> 512x512x128

        x = self.dsize(x) # 512x512 pixels -> 256x256x128

        # 2nd block: 256x256 pixels
        x = self.noise(self.funct(self.norm3(self.conv3(x)))) # 3x3: 256x256x128 -> 256x256x128
        x = self.noise(self.funct(self.norm4(self.conv4(x)))) # 5x5: 256x256x128 -> 256x256x128

        x = self.dsize(x) # 256x256 pixels -> 128x128x128

        x = self.noise(self.funct(self.norm5(self.conv5(x)))) # 3x3: 128x128x128 -> 128x128x128
        x = self.noise(self.funct(self.norm6(self.conv6(x)))) # 5x5: 128x128x128 -> 128x128x128

        x = self.dsize(x) # 128x128 pixels -> 64x64x128

        x = self.noise(self.funct(self.norm7(self.conv7(x)))) # 3x3: 64x64x128 -> 64x64x128
        x = self.noise(self.funct(self.norm8(self.conv8(x)))) # 5x5: 64x64x128 -> 64x64x128

        x = self.final(x).view(x.size(0), -1) # 64x64x128 -> 128 (vector)

        # reparameterize for KL-Divergence
        mu  = self.mu(x)
        lv  = self.lv(x)
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        z   = mu + eps * std

        return z, mu, lv


    # ------------------------------------------------------------------------------------------
    # -------------------------------- METHOD :: Compute Losses --------------------------------
    # ------------------------------------------------------------------------------------------
    def loss(self, encodings: torch.Tensor, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:

        # for cutting the whole batch in the two sub-batches
        half_size = encodings.size(0) // 2

        # split the tensors by class
        encodings_a, encodings_b = encodings[:half_size], encodings[half_size:]
        mu_a, mu_b = mu[:half_size], mu[half_size:]
        lv_a, lv_b = lv[:half_size], lv[half_size:]

        # push images in the same class together (Euclidean distance between encodings)
        identical_loss_a = torch.cdist(encodings_a, encodings_a).mean()
        identical_loss_b = torch.cdist(encodings_b, encodings_b).mean()
        identical_loss   = 0.50 * (identical_loss_a + identical_loss_b)
        identical_loss   = identical_loss * self.beta_identical

        # push images in the same class apart (Euclidean distance between encodings)
        different_max    = torch.sqrt(torch.tensor(2.0, device=encodings.device))
        different_max    = different_max * half_size * half_size
        different_loss   = torch.cdist(encodings_a, encodings_b).mean()
        different_loss   = different_max - different_loss
        different_loss   = different_loss * self.beta_different

        # measures the magnitude of the vectors against a unit vector
        magnitude_loss_a = ((encodings_a.norm(p=2, dim=1) - 1) ** 2).mean()
        magnitude_loss_b = ((encodings_b.norm(p=2, dim=1) - 1) ** 2).mean()
        magnitude_loss   = 0.50 * (magnitude_loss_a + magnitude_loss_b)
        magnitude_loss   = magnitude_loss * self.beta_magnitude

        # push images in the same class apart (Euclidean distance between encodings)
        deviation_loss_a = -0.50 * torch.sum(1 + lv_a - mu_a.pow(2) - lv_a.exp(), dim=1).mean()
        deviation_loss_b = -0.50 * torch.sum(1 + lv_b - mu_b.pow(2) - lv_b.exp(), dim=1).mean()
        deviation_loss   =  0.50 * (deviation_loss_a + deviation_loss_b)
        deviation_loss   = deviation_loss * self.beta_deviation

        # total loss
        total_loss = identical_loss + different_loss + magnitude_loss + deviation_loss

        # total loss and individual losses for debugging
        return total_loss, identical_loss, different_loss, magnitude_loss, deviation_loss
