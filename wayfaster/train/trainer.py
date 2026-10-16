from os import path

from matplotlib.pylab import rint
from matplotlib.pyplot import grid
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.utils as vutils

from .utils import path_to_map
from models.traversability_net import TravNet


class TrainingModule(pl.LightningModule):
    """
    Training module for the TravNet model using PyTorch Lightning.
    """

    def __init__(self, configs):
        """
        Initialize the TrainingModule.

        Args:
            configs (object): Configuration object containing training parameters.
        """
        super().__init__()

        # Save hyperparameters to hparams.yaml
        self.save_hyperparameters()

        # Get training params
        self.eps = 1e-6

        gamma = torch.tensor([
            configs.TRAINING.GAMMA**i
            for i in range(configs.TRAINING.HORIZON)
        ])

        self.gamma = nn.Parameter(
            gamma,
            requires_grad=False
        )

        self.learning_rate = configs.OPTIMIZER.LR
        self.weight_decay = configs.OPTIMIZER.WEIGHT_DECAY
        self.depth_weight = configs.TRAINING.DEPTH_WEIGHT

        # Depth training
        self.train_depth = configs.MODEL.TRAIN_DEPTH

        # Depth prediction
        self.predict_depth = configs.MODEL.PREDICT_DEPTH

        # Mapping Model
        self.model = TravNet(
            configs.MODEL.GRID_BOUNDS,
            configs.MODEL.INPUT_SIZE,
            downsample=configs.MODEL.DOWNSAMPLE,
            image_dim=configs.MODEL.LATENT_DIM,
            temporal_length=configs.MODEL.TIME_LENGTH,
            predict_depth=configs.MODEL.PREDICT_DEPTH,
            fuse_pcloud=configs.MODEL.FUSE_PCLOUD
        )

        # Set grid bounds
        self.grid_bounds = configs.MODEL.GRID_BOUNDS

        self.map_size = (
            int(
                (
                    self.grid_bounds['xbound'][1]
                    - self.grid_bounds['xbound'][0]
                )
                / self.grid_bounds['xbound'][2]
            ),
            int(
                (
                    self.grid_bounds['ybound'][1]
                    - self.grid_bounds['ybound'][0]
                )
                / self.grid_bounds['ybound'][2]
            )
        )

        self.map_origin = (
            int(
                self.grid_bounds['xbound'][1]
                / self.grid_bounds['xbound'][2]
            ),
            int(
                self.grid_bounds['ybound'][1]
                / self.grid_bounds['ybound'][2]
            )
        )

        self.map_resolution = (
            self.grid_bounds['xbound'][2],
            self.grid_bounds['ybound'][2]
        )

        # Visualization rate
        self.vis_interval = configs.TRAINING.VIS_INTERVAL

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        """

        # Get data
        (
            color_img,
            pcloud,
            inv_intrinsics,
            extrinsics,
            path,
            target_trav,
            trav_weights,
            depth_target,
            depth_mask
        ) = batch

        # Forward pass
        trav_map, pred_depth, debug = self.model(
            color_img,
            pcloud,
            inv_intrinsics,
            extrinsics,
            depth_target
        )

        # Project path to map
        executed_path = path_to_map(
            path.unsqueeze(1),
            torch.ones_like(
                path[..., 0, 0]
            ).unsqueeze(1),
            self.map_size,
            self.map_resolution,
            self.map_origin
        )

        # Calculate traversability loss
        trav_loss, _ = self.trav_criterion(
            path,
            trav_map,
            target_trav,
            trav_weights
        )

        # Calculate depth classification loss
        depth_target = depth_target.view(
            -1,
            *depth_target.shape[2:]
        )

        depth_mask = depth_mask.view(
            -1,
            *depth_mask.shape[2:]
        )

        depth_loss = self.depth_criterion(
            pred_depth,
            depth_target,
            depth_mask
        )

        if self.train_depth:
            loss = (
                trav_loss
                + self.depth_weight * depth_loss
            )
        else:
            loss = trav_loss

        # Visualize results
        if (batch_idx % self.vis_interval) == 0:
            self.visualize(
                color_img,
                pcloud,
                trav_map,
                pred_depth,
                depth_target,
                depth_mask,
                debug,
                executed_path,
                prefix='train'
            )

        # Logging
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "train_trav_loss",
            trav_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "train_depth_loss",
            depth_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        """

        # Get data
        (
            color_img,
            pcloud,
            inv_intrinsics,
            extrinsics,
            path,
            target_trav,
            trav_weights,
            depth_target,
            depth_mask
        ) = batch

        # Forward pass
        trav_map, pred_depth, debug = self.model(
            color_img,
            pcloud,
            inv_intrinsics,
            extrinsics,
            depth_target
        )

        # Project path to map
        executed_path = path_to_map(
            path.unsqueeze(1),
            torch.ones_like(
                path[..., 0, 0]
            ).unsqueeze(1),
            self.map_size,
            self.map_resolution,
            self.map_origin
        )

        # Calculate traversability loss
        trav_loss, trav_error = self.trav_criterion(
            path,
            trav_map,
            target_trav,
            trav_weights
        )

        # Calculate depth classification loss
        depth_target = depth_target.view(
            -1,
            *depth_target.shape[2:]
        )

        depth_mask = depth_mask.view(
            -1,
            *depth_mask.shape[2:]
        )

        depth_loss = self.depth_criterion(
            pred_depth,
            depth_target,
            depth_mask
        )

        if self.train_depth:
            loss = (
                trav_loss
                + self.depth_weight * depth_loss
            )
        else:
            loss = trav_loss

        # Visualize results
        if (batch_idx % self.vis_interval) == 0:
            self.visualize(
                color_img,
                pcloud,
                trav_map,
                pred_depth,
                depth_target,
                depth_mask,
                debug,
                executed_path,
                prefix='valid'
            )

        # Logging
        self.log(
            "valid_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "valid_trav_loss",
            trav_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "valid_trav_error",
            trav_error,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "valid_depth_loss",
            depth_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate scheduler.
        """

        model_params = self.model.parameters()

        optimizer = torch.optim.AdamW(
            model_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=1e-5
        )

        return [optimizer], [scheduler]

    def trav_criterion(
        self,
        states,
        trav_map,
        target_trav,
        target_weights
    ):
        """
        Calculate the traversability loss.
        """

        # ==========================================================
        # CALCULATE MAP COORDINATES
        # ==========================================================

        idxu = (
            2
            * (
                states[..., 0]
                - self.grid_bounds['xbound'][0]
            )
            / (
                self.grid_bounds['xbound'][1]
                - self.grid_bounds['xbound'][0]
            )
            - 1
        )

        idxv = (
            2
            * (
                states[..., 1]
                - self.grid_bounds['ybound'][0]
            )
            / (
                self.grid_bounds['ybound'][1]
                - self.grid_bounds['ybound'][0]
            )
            - 1
        )

        grid = torch.stack(
            (-idxv, -idxu),
            -1
        ).unsqueeze(1)

        # ==========================================================
        # FINITE MASK
        # ==========================================================

        finite_mask = torch.isfinite(
            states[..., 0]
        ) & torch.isfinite(
            states[..., 1]
        )

        # ==========================================================
        # INSIDE-MAP MASK
        # ==========================================================

        inside_map_mask = (
            (grid[..., 0] >= -1)
            & (grid[..., 0] <= 1)
            & (grid[..., 1] >= -1)
            & (grid[..., 1] <= 1)
        )

        # ==========================================================
        # FINAL VALID MASK
        # ==========================================================

        mask = (
            finite_mask
            & inside_map_mask.squeeze(1)
        )

        final_valid_count = int(
            mask.sum().item()
        )

        # ==========================================================
        # DEBUG IF NOTHING IS VALID
        # ==========================================================

        if final_valid_count == 0:

            print(
                "\n"
                "============================================================"
            )

            print(
                "TRAV DEBUG: NO VALID TRAJECTORY POINTS"
            )

            print(
                "============================================================"
            )

            print(
                "states shape:",
                tuple(states.shape)
            )

            print(
                "trav_map shape:",
                tuple(trav_map.shape)
            )

            print(
                "grid shape:",
                tuple(grid.shape)
            )

            print(
                "\nGRID BOUNDS:"
            )

            print(
                "xbound:",
                self.grid_bounds['xbound']
            )

            print(
                "ybound:",
                self.grid_bounds['ybound']
            )

            print(
                "\nMAP:"
            )

            print(
                "map_size:",
                self.map_size
            )

            print(
                "map_origin:",
                self.map_origin
            )

            print(
                "map_resolution:",
                self.map_resolution
            )

            # ------------------------------------------------------
            # FINITE STATE DEBUG
            # ------------------------------------------------------

            finite_count = int(
                finite_mask.sum().item()
            )

            print(
                "\nFINITE XY POINTS:",
                finite_count
            )

            if finite_count > 0:

                finite_xy = states[
                    ..., :2
                ][finite_mask]

                print(
                    "finite x range:",
                    float(finite_xy[:, 0].min().item()),
                    "to",
                    float(finite_xy[:, 0].max().item())
                )

                print(
                    "finite y range:",
                    float(finite_xy[:, 1].min().item()),
                    "to",
                    float(finite_xy[:, 1].max().item())
                )

                print(
                    "\nFIRST 20 FINITE XY:"
                )

                print(
                    finite_xy[:20]
                )

            else:

                print(
                    "NO FINITE XY VALUES EXIST"
                )

            # ------------------------------------------------------
            # GRID DEBUG
            # ------------------------------------------------------

            finite_grid = torch.isfinite(
                grid
            ).all(dim=-1)

            finite_grid_count = int(
                finite_grid.sum().item()
            )

            print(
                "\nFINITE GRID POINTS:",
                finite_grid_count
            )

            if finite_grid_count > 0:

                finite_grid_values = grid[
                    finite_grid
                ]

                print(
                    "grid value range:"
                )

                print(
                    "  first coordinate:",
                    float(
                        finite_grid_values[..., 0].min().item()
                    ),
                    "to",
                    float(
                        finite_grid_values[..., 0].max().item()
                    )
                )

                print(
                    "  second coordinate:",
                    float(
                        finite_grid_values[..., 1].min().item()
                    ),
                    "to",
                    float(
                        finite_grid_values[..., 1].max().item()
                    )
                )

            # ------------------------------------------------------
            # MASK COUNTS
            # ------------------------------------------------------

            print(
                "\nMASK COUNTS:"
            )

            print(
                "finite:",
                int(finite_mask.sum().item())
            )

            print(
                "inside map:",
                int(inside_map_mask.squeeze(1).sum().item())
            )

            print(
                "both:",
                final_valid_count
            )

            # ------------------------------------------------------
            # FIRST STATES
            # ------------------------------------------------------

            print(
                "\nFIRST 20 STATES:"
            )

            print(
                states.reshape(-1, states.shape[-1])[:20]
            )

            print(
                "\n============================================================"
            )

            raise RuntimeError(
                "TRAV DEBUG: no trajectory points are both "
                "finite and inside the map."
            )

        # ==========================================================
        # SAMPLE FROM MAP
        # ==========================================================

        traversability = F.grid_sample(
            trav_map,
            grid,
            align_corners=True
        )

        traversability = traversability.permute(
            (0, 2, 3, 1)
        ).squeeze(1)

        # ==========================================================
        # CALCULATE LOSS
        # ==========================================================

        gamma = self.gamma[
            :traversability.shape[1]
        ].view(
            1,
            -1,
            1
        ).expand(
            traversability.shape[0],
            -1,
            2
        )

        error = F.smooth_l1_loss(
            traversability,
            target_trav,
            reduction='none',
            beta=0.1
        )

        loss = (
            error
            * target_weights
            * gamma
        )

        loss = torch.mean(
            loss[mask, :]
        )

        # ==========================================================
        # CALCULATE ERROR
        # ==========================================================

        error = torch.mean(
            error[mask, :]
        )

        return loss, error

    def depth_criterion(
        self,
        prediction,
        target,
        mask
    ):
        """
        Calculate the depth classification loss.
        """

        loss = (
            F.cross_entropy(
                prediction,
                target,
                reduction='none'
            )
            * mask
        )

        return torch.mean(loss)

    def visualize(
        self,
        color_img,
        pcloud,
        trav_map,
        pred_depth,
        depth_target,
        depth_mask,
        debug,
        executed_path,
        prefix='train'
    ):
        if self.logger is None or not hasattr(self.logger, 'log_image'):
            return

        import torchvision.utils as vutils
        import wandb
        import torch

        # Ensure color_img is [B, C, H, W]
        if not torch.is_tensor(color_img):
            color_img = torch.tensor(color_img)

        # If more than 4 dims, flatten sequence into batch
        if color_img.ndim > 4:
            color_img = color_img.reshape(
                -1,
                *color_img.shape[-3:]
            )

        if color_img.ndim != 4:
            print(
                f"[visualize] Unexpected color_img shape: "
                f"{color_img.shape}"
            )
            return

        B, C, H, W = color_img.shape

        # Make grid with max 6 images per row
        nrow = min(B, 6)

        try:
            grid = vutils.make_grid(
                color_img,
                nrow=nrow,
                normalize=True
            )

        except Exception as e:
            print(
                f"[visualize] make_grid failed: {e}"
            )
            return

        grid_np = grid.permute(
            1,
            2,
            0
        ).cpu().numpy()

        # Log to W&B safely
        if self.logger is not None:
            try:
                self.logger.log_image(
                    key=f"{prefix}_color_grid",
                    images=[
                        wandb.Image(grid_np)
                    ]
                )

            except Exception as e:
                print(
                    f"[visualize] W&B logging failed: {e}"
                )