import pickle
import torch
import h5py
import numpy as np
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class DemoBuffer:
	"""
	Demonstration buffer for TD-MPC2 training.
	Loads demonstrations from ManiSkill2 .pkl or .h5 files.
	"""

	def __init__(self, cfg, demo_path):
		self.cfg = cfg
		self._device = torch.device('cuda:0')
		self._demo_path = demo_path
		self._num_eps = 0
		self._demos = []
		self._total_steps = 0

		# Load demonstrations
		self._load_demos()

		# Initialize buffer with demos
		self._init_buffer()

	def _load_demos(self):
		"""Load demonstrations from file."""
		print(f'Loading demonstrations from {self._demo_path}...')

		if self._demo_path.endswith('.pkl'):
			self._load_pkl_demos()
		elif self._demo_path.endswith('.h5'):
			self._load_h5_demos()
		else:
			raise ValueError(f'Unsupported demo file format: {self._demo_path}')

		print(f'Loaded {self._num_eps} demonstration episodes ({self._total_steps} steps)')

	def _load_pkl_demos(self):
		"""Load demonstrations from pickle file."""
		with open(self._demo_path, 'rb') as f:
			data = pickle.load(f)

		# Handle different pickle formats
		if isinstance(data, dict):
			# Format: {'trajectories': [...]} or {'obs': ..., 'actions': ...}
			if 'trajectories' in data:
				trajectories = data['trajectories']
			elif 'episodes' in data:
				trajectories = data['episodes']
			else:
				# Assume it's a single trajectory dict
				trajectories = [data]
		elif isinstance(data, list):
			trajectories = data
		else:
			raise ValueError(f'Unknown pickle format: {type(data)}')

		for traj in trajectories:
			self._process_trajectory(traj)

	def _load_h5_demos(self):
		"""Load demonstrations from HDF5 file (ManiSkill2 format)."""
		with h5py.File(self._demo_path, 'r') as f:
			# ManiSkill2 format: traj_0, traj_1, etc.
			traj_keys = [k for k in f.keys() if k.startswith('traj')]

			for traj_key in sorted(traj_keys, key=lambda x: int(x.split('_')[1])):
				traj_group = f[traj_key]

				# ManiSkill2 v0 format uses env_states instead of obs
				if 'env_states' in traj_group:
					obs = np.array(traj_group['env_states'])
				elif 'obs' in traj_group:
					obs = np.array(traj_group['obs'])
				else:
					raise ValueError(f'No observation key found in {traj_key}: {list(traj_group.keys())}')

				actions = np.array(traj_group['actions'])

				# Handle rewards - use success as sparse reward if no rewards
				if 'rewards' in traj_group:
					rewards = np.array(traj_group['rewards'])
				elif 'success' in traj_group:
					rewards = np.array(traj_group['success']).astype(np.float32)
				else:
					rewards = np.zeros(len(actions))

				# Handle termination
				if 'terminated' in traj_group:
					terminated = np.array(traj_group['terminated'])
				elif 'dones' in traj_group:
					terminated = np.array(traj_group['dones'])
				else:
					terminated = np.zeros(len(actions))
					terminated[-1] = 1.0

				traj = {
					'obs': obs,
					'actions': actions,
					'rewards': rewards,
					'terminated': terminated,
				}
				self._process_trajectory(traj)

	def _process_trajectory(self, traj):
		"""Process a single trajectory into TensorDict format."""
		# Extract observations
		if 'obs' in traj:
			obs = traj['obs']
		elif 'observations' in traj:
			obs = traj['observations']
		elif 'state' in traj:
			obs = traj['state']
		else:
			raise ValueError(f'No observation key found in trajectory: {traj.keys()}')

		# Truncate observations to match expected dimension if needed
		# (ManiSkill2 env_states may contain more info than the observation)
		expected_obs_dim = getattr(self.cfg, 'obs_shape', None)
		if expected_obs_dim is not None and expected_obs_dim != '???':
			# obs_shape is a dict like {'state': (51,)}
			if isinstance(expected_obs_dim, dict):
				obs_key = getattr(self.cfg, 'obs', 'state')
				if obs_key in expected_obs_dim:
					shape = expected_obs_dim[obs_key]
					expected_obs_dim = shape[0] if isinstance(shape, tuple) else shape
				else:
					expected_obs_dim = None
			elif isinstance(expected_obs_dim, (list, tuple)):
				expected_obs_dim = expected_obs_dim[0]
			elif isinstance(expected_obs_dim, str):
				try:
					expected_obs_dim = int(expected_obs_dim.strip('()').split(',')[0])
				except ValueError:
					expected_obs_dim = None
			if isinstance(expected_obs_dim, int) and obs.shape[-1] > expected_obs_dim:
				obs = obs[..., :expected_obs_dim]

		# Extract actions
		if 'actions' in traj:
			actions = traj['actions']
		elif 'action' in traj:
			actions = traj['action']
		else:
			raise ValueError(f'No action key found in trajectory: {traj.keys()}')

		# Extract rewards (optional, default to zeros)
		if 'rewards' in traj:
			rewards = traj['rewards']
		elif 'reward' in traj:
			rewards = traj['reward']
		else:
			rewards = np.zeros(len(actions))

		# Extract termination (optional, default to zeros except last step)
		if 'terminated' in traj:
			terminated = traj['terminated']
		elif 'dones' in traj:
			terminated = traj['dones']
		else:
			terminated = np.zeros(len(actions))
			terminated[-1] = 1.0

		# Convert to tensors
		obs = torch.tensor(obs, dtype=torch.float32)
		actions = torch.tensor(actions, dtype=torch.float32)
		rewards = torch.tensor(rewards, dtype=torch.float32)
		terminated = torch.tensor(terminated, dtype=torch.float32)

		# Ensure shapes are correct
		if len(rewards.shape) == 1:
			rewards = rewards.unsqueeze(-1)
		if len(terminated.shape) == 1:
			terminated = terminated.unsqueeze(-1)

		# Create TensorDict for each step
		# Note: obs has T+1 steps, actions/rewards have T steps
		ep_len = len(actions)
		tds = []
		for t in range(ep_len + 1):
			if t == 0:
				# First step: no action/reward yet
				td = TensorDict({
					'obs': obs[t].unsqueeze(0),
					'action': torch.full_like(actions[0], float('nan')).unsqueeze(0),
					'reward': torch.tensor([float('nan')]),
					'terminated': torch.tensor([float('nan')]),
					'episode': torch.tensor([self._num_eps], dtype=torch.int64),
				}, batch_size=(1,))
			else:
				td = TensorDict({
					'obs': obs[t].unsqueeze(0) if t < len(obs) else obs[-1].unsqueeze(0),
					'action': actions[t-1].unsqueeze(0),
					'reward': rewards[t-1] if len(rewards) > t-1 else torch.tensor([0.0]),
					'terminated': terminated[t-1] if len(terminated) > t-1 else torch.tensor([0.0]),
					'episode': torch.tensor([self._num_eps], dtype=torch.int64),
				}, batch_size=(1,))
			tds.append(td)

		self._demos.append(torch.cat(tds))
		self._total_steps += ep_len + 1
		self._num_eps += 1

	def _init_buffer(self):
		"""Initialize the replay buffer with loaded demos."""
		if self._num_eps == 0:
			raise ValueError('No demonstrations loaded!')

		# Determine storage size
		self._capacity = self._total_steps
		print(f'Demo buffer capacity: {self._capacity:,}')

		# Initialize sampler
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=False,
		)
		self._batch_size = self.cfg.batch_size * (self.cfg.horizon + 1)

		# Determine storage device
		mem_free, _ = torch.cuda.mem_get_info()
		sample_td = self._demos[0][0]
		bytes_per_step = sum([
			(v.numel() * v.element_size() if not isinstance(v, TensorDict)
			 else sum([x.numel() * x.element_size() for x in v.values()]))
			for v in sample_td.values()
		])
		total_bytes = bytes_per_step * self._capacity
		print(f'Demo storage required: {total_bytes/1e9:.2f} GB')
		storage_device = 'cuda:0' if 2.5 * total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for demo storage.')
		self._storage_device = torch.device(storage_device)

		# Create buffer
		self._buffer = ReplayBuffer(
			storage=LazyTensorStorage(self._capacity, device=self._storage_device),
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

		# Load all demos into buffer
		for demo in self._demos:
			self._buffer.extend(demo)

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _prepare_batch(self, td):
		"""Prepare a sampled batch for training."""
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, terminated, task

	def sample(self):
		"""Sample a batch of subsequences from the demo buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon + 1).permute(1, 0)
		return self._prepare_batch(td)


class MixedBuffer:
	"""
	Buffer that mixes samples from online replay and demonstration buffers.
	"""

	def __init__(self, online_buffer, demo_buffer, demo_ratio=0.5):
		self.online_buffer = online_buffer
		self.demo_buffer = demo_buffer
		self.demo_ratio = demo_ratio
		self._device = torch.device('cuda:0')

	def sample(self):
		"""Sample a mixed batch from online and demo buffers."""
		# Calculate number of samples from each buffer
		batch_size = self.online_buffer.cfg.batch_size
		num_demo = int(batch_size * self.demo_ratio)
		num_online = batch_size - num_demo

		if num_demo > 0 and self.demo_buffer.num_eps > 0:
			# Get demo samples
			demo_obs, demo_action, demo_reward, demo_terminated, demo_task = self.demo_buffer.sample()

			if num_online > 0 and self.online_buffer.num_eps > 0:
				# Get online samples
				online_obs, online_action, online_reward, online_terminated, online_task = self.online_buffer.sample()

				# Mix the batches
				# Adjust sizes based on actual batch sizes
				demo_batch = demo_obs.shape[1]
				online_batch = online_obs.shape[1]

				# Take proportional amounts
				demo_take = min(num_demo, demo_batch)
				online_take = min(num_online, online_batch)

				obs = torch.cat([demo_obs[:, :demo_take], online_obs[:, :online_take]], dim=1)
				action = torch.cat([demo_action[:, :demo_take], online_action[:, :online_take]], dim=1)
				reward = torch.cat([demo_reward[:, :demo_take], online_reward[:, :online_take]], dim=1)
				terminated = torch.cat([demo_terminated[:, :demo_take], online_terminated[:, :online_take]], dim=1)

				if demo_task is not None and online_task is not None:
					task = torch.cat([demo_task[:demo_take], online_task[:online_take]], dim=0)
				else:
					task = None

				return obs, action, reward, terminated, task
			else:
				return demo_obs, demo_action, demo_reward, demo_terminated, demo_task
		else:
			return self.online_buffer.sample()

	@property
	def num_eps(self):
		return self.online_buffer.num_eps


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda:0')
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		num_new_eps = len(td)
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, terminated, task

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		return self._prepare_batch(td)
