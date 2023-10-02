import os
import unittest

from fsscore.models.nn_utils import get_new_model_and_trainer


class TestGetNewModelAndTrainer(unittest.TestCase):
    def test_returns_tuple(self):
        result = get_new_model_and_trainer(save_dir="test_dir")
        self.assertIsInstance(result, tuple)

    def test_model_and_trainer_not_none(self):
        model, trainer = get_new_model_and_trainer(save_dir="test_dir")
        self.assertIsNotNone(model)
        self.assertIsNotNone(trainer)

    def test_save_dir_created(self):
        save_dir = "test_dir"
        model, trainer = get_new_model_and_trainer(save_dir=save_dir)
        self.assertTrue(os.path.exists(save_dir))


if __name__ == "__main__":
    unittest.main()
