# --------------------------------------------------------------------------------------------------
# ---------------------------------------------- MAIN ----------------------------------------------
# --------------------------------------------------------------------------------------------------
import os; os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from Dataset.Dataset import Dataset
from Modules.Encoder import Encoder

import typing
import torch


# --------------------------------------------------------------------------------------------------
# -------------------------------- Environment and Hyper-Parameters --------------------------------
# --------------------------------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')

MAX_LEARNING = 1e-5
MIN_LEARNING = 1e-9
BATCH_SIZE   = 8
EPOCHS       = 100

FINDER = Dataset(batch_size=BATCH_SIZE)
LOADER = torch.utils.data.DataLoader(dataset=FINDER, sampler=FINDER, batch_size=BATCH_SIZE)


# --------------------------------------------------------------------------------------------------
# -------------------------------- Load or Initialize Model Weights --------------------------------
# --------------------------------------------------------------------------------------------------
encoder = Encoder(max_lr=MAX_LEARNING, min_lr=MIN_LEARNING, T_0=len(LOADER))
encoder = encoder.to(DEVICE)

try:
    encoder.load('./Weights/encoder.safetensors')
except FileNotFoundError:
    encoder.init()
    encoder.save('./Weights/encoder.safetensors')


# --------------------------------------------------------------------------------------------------
# --------------------------------------- Main Training Loop ---------------------------------------
# --------------------------------------------------------------------------------------------------
for eid in range(EPOCHS):
    for bid, (images, labels) in enumerate(LOADER):

        # move batch and labels to the GPU
        labels = labels.to(DEVICE)
        images = images.to(DEVICE)

        # reset the gradients
        encoder.optimizer.zero_grad()

        # execute both the forward and backward passes
        embeddings, mu, lv = encoder(images)
        total_loss, *losses = encoder.loss(embeddings, mu, lv)

        # back-propagate the loss and step
        total_loss.backward()
        encoder.optimizer.step()
        encoder.scheduler.step()

        # debugging
        print(f"[{eid:03}:{bid:03}] Identical Loss: {losses[0].item():.4f}")
        print(f"[{eid:03}:{bid:03}] Different Loss: {losses[1].item():.4f}")
        print(f"[{eid:03}:{bid:03}] Deviation Loss: {losses[2].item():.4f}")
        print(f"[{eid:03}:{bid:03}] Magnitude Loss: {losses[3].item():.4f}")

    encoder.save('./Weights/encoder.safetensors')