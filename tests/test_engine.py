import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sleeptcn.engine import (
    epoch_forward,
    fit_model,
    load_checkpoint,
    load_model_checkpoint,
    run_loader,
    seed_everything,
)
from sleeptcn.training import masked_cross_entropy


HASH_A = "a" * 64
HASH_B = "b" * 64


def loaders(generator: torch.Generator) -> tuple[DataLoader, DataLoader]:
    features = torch.tensor(
        [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    dataset = TensorDataset(features, targets)
    return (
        DataLoader(dataset, batch_size=2, shuffle=True, generator=generator),
        DataLoader(dataset, batch_size=2, shuffle=False),
    )


class LoaderTests(unittest.TestCase):
    def test_train_and_evaluate_are_finite(self) -> None:
        seed_everything(42)
        model = nn.Linear(2, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        generator = torch.Generator().manual_seed(42)
        train_loader, validation_loader = loaders(generator)
        trained = run_loader(
            model,
            train_loader,
            "cpu",
            epoch_forward,
            masked_cross_entropy,
            optimizer=optimizer,
        )
        evaluated = run_loader(
            model,
            validation_loader,
            "cpu",
            epoch_forward,
            masked_cross_entropy,
        )
        self.assertEqual(trained.valid_epochs, 4)
        self.assertEqual(evaluated.metrics["n_valid_epochs"], 4)
        self.assertGreater(trained.loss, 0.0)


class CheckpointTests(unittest.TestCase):
    def fit(
        self,
        directory: Path,
        max_epochs: int,
        *,
        resume_from: Path | None = None,
    ):
        seed_everything(7)
        model = nn.Linear(2, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        generator = torch.Generator().manual_seed(7)
        train_loader, validation_loader = loaders(generator)
        result = fit_model(
            model,
            train_loader,
            validation_loader,
            optimizer,
            masked_cross_entropy,
            epoch_forward,
            device="cpu",
            max_epochs=max_epochs,
            patience=10,
            checkpoint_dir=directory,
            experiment_id="TEST",
            stage="linear",
            outer_fold=0,
            seed=7,
            config_sha256=HASH_A,
            split_sha256=HASH_B,
            data_variant="paper_raw_v1",
            loader_generator=generator,
            resume_from=resume_from,
        )
        return model, optimizer, generator, result

    def test_atomic_checkpoints_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, _, _, first = self.fit(directory, 2)
            self.assertEqual(first.progress.completed_epochs, 2)
            self.assertTrue(first.best_checkpoint.is_file())
            self.assertTrue(first.latest_checkpoint.is_file())
            _, _, _, resumed = self.fit(
                directory, 3, resume_from=first.latest_checkpoint
            )
            self.assertEqual(resumed.progress.completed_epochs, 3)
            self.assertEqual(resumed.progress.validation_events, 3)
            self.assertEqual(len(resumed.history), 3)
            self.assertEqual(list(directory.glob("*.tmp")), [])

    def test_resumed_weights_equal_uninterrupted_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            continuous_model, _, _, _ = self.fit(root / "continuous", 3)
            _, _, _, partial = self.fit(root / "resumed", 2)
            resumed_model, _, _, _ = self.fit(
                root / "resumed", 3, resume_from=partial.latest_checkpoint
            )
            for name, expected in continuous_model.state_dict().items():
                torch.testing.assert_close(
                    resumed_model.state_dict()[name], expected, rtol=0.0, atol=0.0
                )

    def test_early_stopping_counts_validation_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            seed_everything(11)
            model = nn.Linear(2, 5)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            generator = torch.Generator().manual_seed(11)
            train_loader, validation_loader = loaders(generator)
            result = fit_model(
                model,
                train_loader,
                validation_loader,
                optimizer,
                masked_cross_entropy,
                epoch_forward,
                device="cpu",
                max_epochs=10,
                patience=2,
                checkpoint_dir=Path(temporary),
                experiment_id="TEST",
                stage="early_stop",
                outer_fold=0,
                seed=11,
                config_sha256=HASH_A,
                split_sha256=HASH_B,
                data_variant="paper_raw_v1",
                loader_generator=generator,
            )
            self.assertTrue(result.stopped_early)
            self.assertEqual(result.progress.completed_epochs, 3)
            self.assertEqual(result.progress.validation_events, 3)
            self.assertEqual(result.progress.bad_validations, 2)

    def test_rejects_wrong_split_hash_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, _, _, result = self.fit(directory, 1)
            model = nn.Linear(2, 5)
            optimizer = torch.optim.Adam(model.parameters())
            before = {
                name: parameter.detach().clone()
                for name, parameter in model.state_dict().items()
            }
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                load_checkpoint(
                    result.latest_checkpoint,
                    model,
                    optimizer,
                    expected_metadata={"split_sha256": "c" * 64},
                    device="cpu",
                    restore_rng=False,
                )
            for name, expected in before.items():
                torch.testing.assert_close(model.state_dict()[name], expected)

    def test_validation_loss_can_select_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            seed_everything(13)
            model = nn.Linear(2, 5)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            generator = torch.Generator().manual_seed(13)
            train_loader, validation_loader = loaders(generator)
            result = fit_model(
                model,
                train_loader,
                validation_loader,
                optimizer,
                masked_cross_entropy,
                epoch_forward,
                device="cpu",
                max_epochs=2,
                patience=1,
                checkpoint_dir=Path(temporary),
                experiment_id="TEST",
                stage="weighted_cnn",
                outer_fold=0,
                seed=13,
                config_sha256=HASH_A,
                split_sha256=HASH_B,
                data_variant="paper_raw_v1",
                selection_metric="validation_loss",
                loader_generator=generator,
            )
            self.assertEqual(result.progress.best_epoch, 0)
            restored = nn.Linear(2, 5)
            metadata = load_model_checkpoint(
                result.best_checkpoint,
                restored,
                expected_metadata={"selection_metric": "validation_loss"},
                device="cpu",
            )
            self.assertEqual(metadata["stage"], "weighted_cnn")


if __name__ == "__main__":
    unittest.main()
