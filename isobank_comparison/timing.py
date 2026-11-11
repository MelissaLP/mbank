"""
Estimate how long the three mbank steps take for an ini file, by timing a small
piece of each and scaling it up:

	python timing.py eccentric_5d.ini [--flow eccentric_5d/flow.zip] [--n-templates 3e6]

- dataset: one metric per datapoint, computed one after the other
- training: time per epoch, times n-epochs (an upper bound: early stopping can end sooner)
- placement: one metric per livepoint, then covering iterations. Each iteration draws
  3000 proposals and keeps those inside the boundaries as templates, so the number of
  iterations is n-templates / (3000 * fraction inside).

The cost per epoch and per iteration does not depend on the weights, so training is
timed with an untrained flow. The fraction of proposals inside the boundaries does:
pass a trained flow with --flow to measure it, otherwise all proposals are assumed inside.
"""

import argparse
import os
import time
import warnings

import numpy as np
import torch

import mbank.parser
from mbank import cbc_metric
from mbank.flow import STD_GW_Flow
from mbank.utils import load_PSD
import scipy.linalg

PROPOSALS_PER_ITER = 3000	# as in mbank.placement.place_random_flow


def read_ini(ini_file):
	"Arguments of the ini file, read as the mbank scripts do"
	parser = argparse.ArgumentParser()
	mbank.parser.add_general_options(parser)
	mbank.parser.add_metric_options(parser)
	mbank.parser.add_flow_options(parser)
	mbank.parser.add_range_options(parser)
	mbank.parser.add_template_placement_options(parser)
	parser.add_argument("--n-datapoints", default = 10000, type = int)
	args, _ = parser.parse_known_args([])
	return mbank.parser.updates_args_from_ini(ini_file, args, parser)


def human(seconds):
	for unit, size in (('d', 86400), ('h', 3600), ('min', 60)):
		if seconds >= size:
			return '{:.1f} {}'.format(seconds/size, unit)
	return '{:.1f} s'.format(seconds)


def time_metric(args, n):
	"Seconds per metric evaluation, at n points drawn inside the boundaries"
	m = cbc_metric(args.variable_format,
		PSD = load_PSD(args.psd, args.asd, args.ifo, df = args.df),
		approx = args.approximant,
		f_min = args.f_min, f_max = args.f_max)
	points = mbank.parser.boundary_keeper(args).sample(n + 1, args.variable_format)
	m.get_metric(points[0], metric_type = args.metric_type)	# warm-up
	times, failed = [], 0
	for p in points[1:]:
		t0 = time.perf_counter()
		try:
			m.get_metric(p, metric_type = args.metric_type)
		except Exception:
			failed += 1
		times.append(time.perf_counter() - t0)
	return np.median(times), np.percentile(times, 90), failed


def time_epoch(args, D, n_epochs):
	"Seconds per training epoch, on random data of the dataset's size"
	n_train = int(args.n_datapoints*args.train_fraction)
	data = np.random.uniform(0, 1, (args.n_datapoints, D))
	ll = np.random.normal(0, 1, args.n_datapoints)
	flow = STD_GW_Flow(D, n_layers = args.n_layers, hidden_features = args.hidden_features, has_constant = True)
	optimizer = torch.optim.Adam(flow.parameters(), lr = args.learning_rate)
	t0 = time.perf_counter()
	flow.train_flow(args.loss_function, N_epochs = n_epochs,
		train_data = data[:n_train], train_weights = ll[:n_train],
		validation_data = data[n_train:], validation_weights = ll[n_train:],
		optimizer = optimizer, batch_size = args.batch_size, validation_step = 100,
		verbose = False)
	return (time.perf_counter() - t0)/n_epochs, flow


def time_iteration(flow, n_livepoints, D, n_repeats = 3):
	"Seconds per covering iteration with all livepoints alive (an upper bound)"
	cholesky = np.tile(np.eye(D, dtype = np.float32), (n_livepoints, 1, 1))
	livepoints = np.random.uniform(0, 1, (n_livepoints, D)).astype(np.float32)
	t0 = time.perf_counter()
	for _ in range(n_repeats):
		with torch.no_grad():
			proposals = flow.sample(PROPOSALS_PER_ITER).numpy()
		for l, L in zip(livepoints, cholesky):
			diff = scipy.linalg.blas.sgemm(1, proposals - l, L)
			np.any(np.sum(np.square(diff), axis = 1) < 0.03)
	return (time.perf_counter() - t0)/n_repeats


def main():
	parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
	parser.add_argument("ini")
	parser.add_argument("--n-metric", default = 20, type = int, help = "metrics to time")
	parser.add_argument("--n-epochs", default = 100, type = int, help = "epochs to time")
	parser.add_argument("--n-templates", default = 3e6, type = float, help = "expected bank size")
	parser.add_argument("--flow", default = None, help = "trained flow, to measure the fraction of proposals inside")
	cli = parser.parse_args()

	if os.environ.get('OMP_NUM_THREADS') != '1':
		warnings.warn("OMP_NUM_THREADS is not 1: lalsimulation 7.0.0 TaylorF2Ecc is then not reproducible (see README)")

	args = read_ini(cli.ini)
	D = mbank.variable_handler().D(args.variable_format)
	print('{}: {}, {}, D = {}, torch threads = {}'.format(cli.ini, args.variable_format,
		args.approximant, D, torch.get_num_threads()))

	t_metric, t_metric90, failed = time_metric(args, cli.n_metric)
	print('metric: {:.2f} s median, {:.2f} s 90th pct ({} of {} failed)'.format(
		t_metric, t_metric90, failed, cli.n_metric))

	t_epoch, flow = time_epoch(args, D, cli.n_epochs)
	print('training: {:.1f} ms per epoch'.format(1e3*t_epoch))

	inside = 1.
	if cli.flow:
		flow = STD_GW_Flow.load_flow(cli.flow)
		with torch.no_grad():
			samples = flow.sample(20000).numpy()
		inside = mbank.parser.boundary_keeper(args)(samples, args.variable_format).mean()
	t_iter = time_iteration(flow, args.n_livepoints, D)
	n_iter = cli.n_templates/(PROPOSALS_PER_ITER*inside)
	print('placement: {:.2f} s per iteration with {} livepoints, {:.1%} of proposals inside{}'.format(
		t_iter, args.n_livepoints, inside, '' if cli.flow else ' (assumed)'))

	rows = [
		('dataset', '{} metrics'.format(args.n_datapoints), args.n_datapoints*t_metric),
		('training', '{} epochs (at most)'.format(args.n_epochs), args.n_epochs*t_epoch),
		('placement: metrics', '{} livepoints'.format(args.n_livepoints), args.n_livepoints*t_metric),
		('placement: covering', '{:.0f} iterations (at most)'.format(n_iter), n_iter*t_iter),
	]
	print('\n{:<22}{:<28}{:>10}'.format('step', 'work', 'time'))
	for step, work, seconds in rows:
		print('{:<22}{:<28}{:>10}'.format(step, work, human(seconds)))
	print('{:<22}{:<28}{:>10}'.format('total', '', human(sum(r[2] for r in rows))))


if __name__ == '__main__':
	main()
