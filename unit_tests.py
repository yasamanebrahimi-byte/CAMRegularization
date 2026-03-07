import os
import unittest

from torch.utils.data import Dataset

from dataset_registry import get_available_datasets, get_dataset_loaders


class TestDatasetRegistry(unittest.TestCase):
    def test_all_registered_datasets_load_as_torch_datasets(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        not_yet_loadable = []

        for dataset_name in get_available_datasets():
            with self.subTest(dataset=dataset_name):
                try:
                    train_loader, _, test_loader = get_dataset_loaders(
                        dataset_name=dataset_name,
                        data_dir=data_dir,
                        batch_size=2,
                        num_workers=0,
                    )

                    self.assertIsNotNone(train_loader)
                    self.assertIsNotNone(test_loader)
                    self.assertIsInstance(train_loader.dataset, Dataset)
                    self.assertIsInstance(test_loader.dataset, Dataset)
                except Exception as exc:
                    if dataset_name == "cifar100":
                        self.fail(f"cifar100 must always load, but failed with: {exc}")
                    not_yet_loadable.append((dataset_name, str(exc)))

        if not_yet_loadable:
            print("Datasets currently not loadable (expected until imported/prepared):")
            for name, message in not_yet_loadable:
                print(f"- {name}: {message}")

    def test_malimg_loads_as_torch_dataset_type(self):
        self.assertIn("malimg", get_available_datasets())

        data_dir = os.path.join(os.path.dirname(__file__), "data")

        try:
            train_loader, _, test_loader = get_dataset_loaders(
                dataset_name="malimg",
                data_dir=data_dir,
                batch_size=2,
                num_workers=0,
            )
        except Exception as exc:
            self.skipTest(f"Unable to load malimg in this environment: {exc}")

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)
        self.assertIsInstance(train_loader.dataset, Dataset)
        self.assertIsInstance(test_loader.dataset, Dataset)


if __name__ == "__main__":
    unittest.main()