# Same as download_d4rl_datasets.py, but fetches the raw D4RL hdf5 files directly
# so that neither d4rl nor mujoco-py need to be installed.
import collections
import os
import pickle
import urllib.request

import h5py
import numpy as np

URL = 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/{}_{}-v2.hdf5'

for env_name in ['halfcheetah', 'hopper', 'walker2d']:
	for dataset_type in ['medium', 'medium-replay', 'expert']:
		name = f'{env_name}-{dataset_type}-v2'
		h5_path = f'{name}.hdf5'
		if not os.path.exists(h5_path):
			urllib.request.urlretrieve(URL.format(env_name, dataset_type.replace('-', '_')), h5_path)
		with h5py.File(h5_path, 'r') as f:
			dataset = {k: f[k][:] for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals', 'timeouts']}

		N = dataset['rewards'].shape[0]
		data_ = collections.defaultdict(list)

		episode_step = 0
		paths = []
		for i in range(N):
			done_bool = bool(dataset['terminals'][i])
			final_timestep = dataset['timeouts'][i]
			for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
				data_[k].append(dataset[k][i])
			if done_bool or final_timestep:
				episode_step = 0
				episode_data = {}
				for k in data_:
					episode_data[k] = np.array(data_[k])
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
			episode_step += 1

		returns = np.array([np.sum(p['rewards']) for p in paths])
		num_samples = np.sum([p['rewards'].shape[0] for p in paths])
		print(name)
		print(f'Number of samples collected: {num_samples}')
		print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

		with open(f'{name}.pkl', 'wb') as f:
			pickle.dump(paths, f)
