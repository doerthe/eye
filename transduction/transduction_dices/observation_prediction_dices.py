import random
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class ObservationPredictionDices(text_problems.Text2TextProblem):
  """Transduction from observation to prediction for dices."""

  @property
  def approx_vocab_size(self):
    return 512

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

    for n in range(10000):
      outcome = random.randint(1, 6)
      yield {
        "inputs": "A_THROW " + repr(n),
        "targets": repr(outcome)
      }

if __name__ == '__main__':
  gen = ObservationPredictionDices.generate_samples(None, None, None, None)
  for i in gen:
    print(i)
