import random
import cmath
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class ObservationPredictionRoots(text_problems.Text2TextProblem):
  """Transduction from observation to prediction for roots of polynomial ax**2 + bx + c = 0"""

  @property
  def approx_vocab_size(self):
    return 2**14  # ~16k

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 7,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 3,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    del dataset_split

    for n in range(1000000):
      # coefficient a
      a = random.randint(1, 32)
      # coefficient b
      b = random.randint(-15, 16)
      # coefficient c
      c = random.randint(-15, 16)
      # roots
      r1 = (-b-cmath.sqrt(b**2-4*a*c))/(2*a)
      r2 = (-b+cmath.sqrt(b**2-4*a*c))/(2*a)
      # feed the protocol buffer
      yield {
        "inputs": "A_POLYNOMIAL with coefficients " + repr(a) + " " + repr(b) + " " + repr(c),
        "targets": "A_ROOT real " + r1.real.__format__('.2f') + " imag " + r1.imag.__format__('.2f') + \
          " A_ROOT real " + r2.real.__format__('.2f') + " imag " + r2.imag.__format__('.2f')
      }

if __name__ == '__main__':
  gen = ObservationPredictionRoots.generate_samples(None, None, None, None)
  for i in gen:
    print(i)
