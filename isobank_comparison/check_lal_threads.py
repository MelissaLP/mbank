"""
Check that lalsimulation's TaylorF2Ecc gives the same waveform at every call.

	python check_lal_threads.py                     # with the current OMP_NUM_THREADS
	OMP_NUM_THREADS=18 python check_lal_threads.py  # e.g. as many threads as a Slurm job

lalsimulation 7.0.0 (conda-forge, macOS) differs by up to 20 % between calls with more
than one thread; with OMP_NUM_THREADS=1 it is exact.
"""

import os

import lal
import lalsimulation
import numpy as np


def waveform(e):
	hp, _ = lalsimulation.SimInspiralChooseFDWaveform(1.568*lal.MSUN_SI, 1.046*lal.MSUN_SI,
		0, 0, 0.04, 0, 0, 0.027, 1e6*lal.PC_SI, 0., 0., 0., e, 0., 0.5, 15., 1024., 15.,
		lal.CreateDict(), lalsimulation.TaylorF2Ecc)
	return hp.data.data.copy()


def main():
	print('lalsimulation {}, OMP_NUM_THREADS = {}'.format(lalsimulation.__version__,
		os.environ.get('OMP_NUM_THREADS', 'not set')))
	for e in (0.01, 0.05, 0.135):
		w = [waveform(e) for _ in range(10)]
		diff = max(np.max(np.abs(w[0] - x)) for x in w[1:])/np.max(np.abs(w[0]))
		print('e = {:.3f}: max relative difference over 10 calls {:.1e} -> {}'.format(
			e, diff, 'OK' if diff == 0 else 'NOT REPRODUCIBLE'))


if __name__ == '__main__':
	main()
