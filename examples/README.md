# Miniature research sample

`mini_events.npz` contains eight simulated events taken from the research
sample. It contains raw detector features, parton targets, labels, decay
assignments and high-level observables. It is stored as plain NumPy arrays
and loads with `allow_pickle=False`.

The original standardized values were converted back to physical units with
their original statistics. Absent kinematic slots and invalid high-level
targets are represented by NaN. The original normalization statistics are
excluded. No events were generated or duplicated.

`python scripts/make_example_cache.py --output data/example.pkl` uses the
first six events for training and the last two for testing, and calculates
normalization from the training split. Dimensions with no training observations
use mean zero and standard deviation one. Dimensions with a single observation
use that observation as the mean and standard deviation one.

This sample is sufficient to exercise the interfaces. Its size and fixed split
make it unsuitable for evaluating reconstruction performance or reproducing
the manuscript's results.
