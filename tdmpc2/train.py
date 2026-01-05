import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer, DemoBuffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from trainer.demo_trainer import DemoTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	# Check if demo path is provided
	demo_path = cfg.get('demo_path', None)
	use_demos = demo_path is not None and demo_path != 'null'

	# Select trainer class
	if cfg.multitask:
		trainer_cls = OfflineTrainer
	elif use_demos:
		trainer_cls = DemoTrainer
	else:
		trainer_cls = OnlineTrainer

	# Create environment first (needed for obs_shape in demo loading)
	env = make_env(cfg)

	# Load demonstration buffer if demo_path is provided (after env creation)
	demo_buffer = None
	if use_demos:
		print(colored('Loading demonstrations...', 'cyan', attrs=['bold']))
		demo_buffer = DemoBuffer(cfg, demo_path)

		# Disable BC pretraining if policy_pretraining is False
		if not cfg.get('policy_pretraining', True):
			cfg = cfg.__class__(**{**{k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')}, 'bc_pretraining_steps': 0})

	# Create trainer
	trainer_kwargs = dict(
		cfg=cfg,
		env=env,
		agent=TDMPC2(cfg),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	if demo_buffer is not None:
		trainer_kwargs['demo_buffer'] = demo_buffer

	trainer = trainer_cls(**trainer_kwargs)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
