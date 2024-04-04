import copy
import glob
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import sys
import torch
import torchvision
import os
from lightly.loss import NegativeCosineSimilarity, NTXentLoss
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from numpy.linalg import norm
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import ParameterGrid
from torch import nn
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm

# Select device based on availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    from cuml import UMAP
    from cuml.cluster import hdbscan
    from cuml.metrics.cluster.silhouette_score import cython_silhouette_score as silhouette_score
    accelerator = "gpu"
else:
    import hdbscan
    import umap as UMAP
    from sklearn.metrics import silhouette_score
    accelerator = "cpu"

torch.set_float32_matmul_precision("high")

# Adjusting the system path to include the directory where custom modules are located
sys.path.append(os.path.abspath('./hits_sdo_scripts'))
from augmentation import Augmentations
from augmentation_list import AugmentationList
from image_utils import read_image

# Setting a seed for reproducibility
seed = 12345
pl.seed_everything(seed, workers=True)

# Defining global variables for the training process
data_path = os.path.abspath('./AIA_171_Images')
epochs = 16
data_stride = 1
batch_size = 64
augmentation = 'double'
loss = 'contrast'  # Can be 'contrast' or 'cos'
learning_rate = 0.1
cosine_scheduler_start = .1
cosine_scheduler_end = 1.0
projection_size = 128
prediction_size = 128
name = "sdo_byol_model"

class SDOTilesDataset(Dataset):
    def __init__(self, data_path: str, augmentation: str='single',
                 data_stride:int = 1, datatype=np.float32):
        self.data_path = data_path
        self.image_files = glob.glob(data_path + "/**/*.jpg", recursive=True)
        if data_stride>1:
            self.image_files = self.image_files[::data_stride]
        self.augmentation_list = AugmentationList(instrument="euv")
        self.augmentation_list.keys.remove('brighten')
        self.datatype=datatype
        self.augmentation = augmentation
        if self.augmentation is None:
            self.augmentation = 'none'

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = read_image(image_loc = self.image_files[idx],
                           image_format="jpg")

        if self.augmentation.lower() != 'none':

            aug = Augmentations(image, self.augmentation_list.randomize())
            image2, _ = aug.perform_augmentations(fill_void='Nearest')

            if self.augmentation.lower() == 'double':
                aug = Augmentations(image, self.augmentation_list.randomize())
                image, _ = aug.perform_augmentations(fill_void='Nearest')

            image = np.moveaxis(image, [0, 1, 2], [1, 2, 0]).astype(self.datatype)           
            image2 = np.moveaxis(image2, [0, 1, 2], [1, 2, 0]).astype(self.datatype)           

            return image, image2, self.image_files[idx]

        else:

            image = np.moveaxis(image, [0, 1, 2], [1, 2, 0]).astype(self.datatype)           
            return image, self.image_files[idx]

class BYOL(pl.LightningModule):
    def __init__(self, lr=0.1, projection_size=256, prediction_size=256, cosine_scheduler_start=0.1, cosine_scheduler_end=1.0, epochs=10, loss='cos'):
        super().__init__()

        resnet = torchvision.models.resnet18() # Play w/ resnet.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BYOLProjectionHead(512, 1024, projection_size)
        self.prediction_head = BYOLPredictionHead(projection_size, 1024, prediction_size)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)


        self.loss = loss
        self.loss_cos = NegativeCosineSimilarity()
        self.loss_contrast = NTXentLoss()

        self.cosine_scheduler_start = cosine_scheduler_start
        self.cosine_scheduler_end = cosine_scheduler_end
        self.epochs = epochs
        self.lr = lr

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):

        momentum = cosine_schedule(self.current_epoch, self.epochs, self.cosine_scheduler_start, self.cosine_scheduler_end)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        (x0, x1, _) = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)

        loss_cos = 0.5 * (self.loss_cos(p0, z1) + self.loss_cos(p1, z0))
        loss_contrast = 0.5 * (self.loss_contrast(p0, z1) + self.loss_contrast(p1, z0))

        if self.loss == 'cos':
            loss = loss_cos
        else:
            loss = loss_contrast

        self.log('loss cos', loss_cos)
        self.log('loss contrast', loss_contrast)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


# Main script execution: model instantiation, data preparation, training, and saving
if __name__ == "__main__":
    model = BYOL(lr=learning_rate, projection_size=projection_size, prediction_size=prediction_size, cosine_scheduler_start=cosine_scheduler_start, cosine_scheduler_end=cosine_scheduler_end, epochs=epochs, loss=loss)

    dataset = SDOTilesDataset(data_path=data_path, augmentation=augmentation, data_stride=data_stride)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=16,
    )

    trainer = pl.Trainer(max_epochs=epochs,
                         accelerator=accelerator, devices=1, strategy="auto",
                         log_every_n_steps=4, deterministic=True)

    trainer.fit(model=model, train_dataloaders=dataloader)

    torch.save(model.state_dict(), f'{name}_.pt')
