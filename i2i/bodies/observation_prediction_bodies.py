#!/usr/bin/env python3

import random
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class ObservationPredictionBodies(text_problems.Text2TextProblem):
  """Transduction from observation to prediction for bodies."""

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

    for n in range(100000):
      # wind turbine size factor
      size_factor = random.randint(1, 10)
      # wind speed
      wind_speed = max(0, int(random.gauss(25, 15)))
      # wind turbine power
      turbine_power = int(0.01*size_factor*wind_speed**3)
      yield {
        "inputs": "_:TURBINE <i2i#size_factor> " + repr(size_factor) + "; <i2i#windspeed_km_h> " + repr(wind_speed) + ".",
        "targets": "_:TURBINE <i2i#producing_kW> " + repr(turbine_power) + "."
      }

      # weight
      weight = max(35, int(random.gauss(62, 20)))
      # height
      height = max(110, int(random.gauss(170, 15)))
      # body mass index
      bmi = int(weight/(height*0.01)**2)
      if bmi < 18:
        bmi_class = "U"
      elif bmi >= 18 and bmi < 25:
        bmi_class = "N"
      elif bmi >= 25 and bmi < 30:
        bmi_class = "O"
      elif bmi >= 30 and bmi < 35:
        bmi_class = "O1"
      elif bmi >= 35 and bmi < 40:
        bmi_class = "O2"
      elif bmi >= 40:
        bmi_class = "O3"
      yield {
        "inputs": "_:PERSON <i2i#weight_kg> " + repr(weight) + "; <i2i#height_cm> " + repr(height) + ".",
        "targets": "_:PERSON <i2i#bmi_class> " + repr(bmi_class) + "."
      }

if __name__ == '__main__':
  gen = ObservationPredictionBodies.generate_samples(None, None, None, None)
  for i in gen:
    print(i)
