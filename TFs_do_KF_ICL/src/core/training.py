import logging
import time
import hashlib
import os
import subprocess
import sys
import numpy as np

from core import Config
import os
import glob
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import DeviceStatsMonitor


class WandbCacheCleanupCallback(pl_callbacks.Callback):
    """Periodically trim the local wandb artifact cache so long runs don't fill the root disk.

    `WandbLogger(log_model="all")` uploads every checkpoint as an artifact and caches
    it under ~/.cache/wandb/artifacts. `wandb artifact cache cleanup <target>` removes
    entries above the target size, keeping the most recently used.
    """

    def __init__(self, target_size: str = "1GB"):
        super().__init__()
        self.target_size = target_size

    def _cleanup(self, step):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "wandb", "artifact", "cache", "cleanup", self.target_size],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.warning(
                    f"wandb cache cleanup at step {step} exited {result.returncode}: "
                    f"{result.stderr.strip()[:500]}"
                )
            else:
                logger.info(f"wandb cache cleaned at step {step} (target {self.target_size})")
        except Exception as e:
            logger.warning(f"wandb cache cleanup at step {step} raised: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank != 0:
            return
        step = trainer.global_step
        if step > 0 and step % config.train_int == 0:
            self._cleanup(step)

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        self._cleanup(trainer.global_step)
# from log_scale_checkpoints import LogScaleCheckpoint

# Setup logger
logger = logging.getLogger(__name__)
config = Config()


def setup_train(model, train_mix_dist=False, train_mix_state_dim=False):
    output_dir = None
    if config.ckpt_path is not None and config.ckpt_path != '':
        output_dir = "/".join(config.ckpt_path.split("/")[:-2])
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Resuming from the checkpoint: {config.ckpt_path}')
    if output_dir is None:
        print("in output_dir is None")
        identifier = config.model_type + ("_NoPE" if not config.use_pos_emb else "") + '/'
                      
        timestamp = time.strftime('%y%m%d_%H%M%S') + '.' + hashlib.md5(config.get_full_yaml().encode('utf-8')).hexdigest()[:6]

        experiment_name = (f"back_len_{config.backstory_len}_" if config.masking and config.backstory_len != config.ny+2 else "") + ("iid_gaussian_" if config.masking and config.iid_gaussian else "")+ ("init_" if config.masking and config.mask_only_init else "") + timestamp + ("_multi_sys_trace" if config.multi_sys_trace else "") + ("_zero_cut" if config.zero_cut else "") + f"_{config.dataset_typ}_state_dim_{config.nx}{config.C_dist}" + ("_dist_mix" if train_mix_dist else "") + ("_state_dim_mix" if train_mix_state_dim else "") + "_lr_" + str(config.learning_rate) + "_num_train_sys_" + str(config.num_tasks)

        if config.mem_suppress:
            experiment_name = mem_suppress_ckpt_path(config, experiment_name, 0)

        output_dir = '../outputs/' + identifier + experiment_name

        #for BLISS server 
        ckpt_dir = f"{os.environ.get('BASE_PATH')}model_checkpoints/" + identifier + experiment_name


        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Write source code to output dir
        config.write_file_contents(output_dir)

    print("os.path.isdir(output_dir)", os.path.isdir(output_dir))
    # Log messages to file
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(output_dir + '/messages.log')
    if not root_logger.handlers:
        root_logger.addHandler(logging.StreamHandler())
    file_handler.setFormatter(root_logger.handlers[0].formatter)
    for handler in root_logger.handlers[1:]:  # all except stdout
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    # Print model details
    num_params = sum([
        np.prod(p.size())
        for p in filter(lambda p: p.requires_grad, model.parameters())
    ])
    logger.info('\nThere are %d trainable parameters.\n' % num_params)

    return output_dir, ckpt_dir, experiment_name


def get_callbacks_and_loggers_old(model, output_dir):
    lr_monitor = pl_callbacks.LearningRateMonitor(logging_interval='epoch')
    tb_logger = pl_loggers.TensorBoardLogger(output_dir)
    loggers = [tb_logger]

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="{step}",
        save_top_k=-1,
        every_n_train_steps=20000,
    )

    ckpt_path = None
    ckpt_paths = glob.glob(os.path.join(output_dir, "checkpoints", "epoch=*"))
    if len(ckpt_paths) > 0:
        ckpt_paths = [int(ckpt_path.split(os.path.sep)[-1]
                          [len("epoch="):len("epoch=") + 2]) for ckpt_path in ckpt_paths]
        max_idx = max(ckpt_paths)
        ckpt_path = os.path.join(output_dir, "checkpoints",
                                 "epoch={:02d}.ckpt".format(max_idx))
        logger.info("Resuming from checkpoint: {}".format(
            ckpt_path.split(os.path.sep)[-1]))

    callbacks = [checkpoint_callback, lr_monitor]
    return callbacks, loggers

def get_callbacks_and_loggers_new_eig(model, output_dir, emb_dim): #add emb_dim as a parameter
    lr_monitor = pl_callbacks.LearningRateMonitor(logging_interval='epoch')
    tb_logger = pl_loggers.TensorBoardLogger(output_dir)
    loggers = [tb_logger]

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="new_eig_{step}", # for embed dim experiments: "emb_dim_" + str(emb_dim) + "_{step}",
        save_top_k=-1, 
        every_n_train_steps=10000,
    )

    ckpt_path = None
    ckpt_paths = glob.glob(os.path.join(output_dir, "checkpoints", "epoch=*"))
    if len(ckpt_paths) > 0:
        ckpt_paths = [int(ckpt_path.split(os.path.sep)[-1]
                          [len("epoch="):len("epoch=") + 2]) for ckpt_path in ckpt_paths]
        max_idx = max(ckpt_paths)
        ckpt_path = os.path.join(output_dir, "checkpoints",
                                 "epoch={:02d}.ckpt".format(max_idx))
        logger.info("Resuming from checkpoint: {}".format(
            ckpt_path.split(os.path.sep)[-1]))

    callbacks = [checkpoint_callback, lr_monitor]
    return callbacks, loggers

def mem_suppress_ckpt_path(config, output_dir, experiment_ind=59):
    if "mask" in output_dir or "backstory" in output_dir in output_dir:
        if config.plateau:
            output_dir = f"{output_dir[:experiment_ind]}plateau_{output_dir[experiment_ind:]}"
        print("output_dir already has mask or backstory")
        return output_dir
    if config.masking:
        output_dir = f"{output_dir[:experiment_ind]}masked_{output_dir[experiment_ind :]}"
    else:
        output_dir = f"{output_dir[:experiment_ind]}unmasked_{output_dir[experiment_ind:]}"

    if config.backstory:
        output_dir = f"{output_dir[:experiment_ind]}backstory_{output_dir[experiment_ind:]}"
    else:
        raise ValueError("No backstory specified")
    
    if config.plateau:
        output_dir = f"{output_dir[:experiment_ind]}plateau_{output_dir[experiment_ind:]}"
    
    return output_dir

def get_callbacks_and_loggers(config, output_dir, train_int): #add emb_dim as a parameter
    lr_monitor = pl_callbacks.LearningRateMonitor(logging_interval='epoch')
    try:
        tb_logger = pl_loggers.TensorBoardLogger(output_dir)
    except ModuleNotFoundError:
        tb_logger = CSVLogger(save_dir=output_dir, name="logs")
    loggers = [tb_logger]


    experiment_ind = 71

    if config.mem_suppress: #create new dir for mem suppress checkpoints
        output_dir = mem_suppress_ckpt_path(config, output_dir, experiment_ind)
        
        os.makedirs(output_dir, exist_ok=True)

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="{step}", # for embed dim experiments: "emb_dim_" + str(emb_dim) + "_{step}",
        save_top_k=-1, 
        every_n_train_steps=train_int if not config.use_true_len else 0, #this changed from 10000 to train_step
        every_n_epochs = 10 if config.use_true_len else 0
    )

    print("\n\n\ncheckpoint_callback", checkpoint_callback.dirpath)

    ckpt_path = None
    ckpt_paths = glob.glob(os.path.join(output_dir, "checkpoints", "epoch=*"))
    if len(ckpt_paths) > 0:
        ckpt_paths = [int(ckpt_path.split(os.path.sep)[-1]
                          [len("epoch="):len("epoch=") + 2]) for ckpt_path in ckpt_paths]
        max_idx = max(ckpt_paths)
        ckpt_path = os.path.join(output_dir, "checkpoints",
                                 "epoch={:02d}.ckpt".format(max_idx))
        logger.info("Resuming from checkpoint: {}".format(
            ckpt_path.split(os.path.sep)[-1]))
        

    wandb_cache_cleanup = WandbCacheCleanupCallback()

    callbacks = [checkpoint_callback, lr_monitor, DeviceStatsMonitor(cpu_stats=True), wandb_cache_cleanup]
    return callbacks, loggers


## experiment with log scale checkpoints
    # log_scale_checkpoint = LogScaleCheckpoint(
    #     dirpath='checkpoints/',
    #     filename='{epoch}-{step}',
    #     save_top_k=-1,  # Save all checkpoints
    #     verbose=True
    # )
