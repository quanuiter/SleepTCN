"""Explicit opt-ins: a source checkout never requires private payloads or training."""
import pytest

# These existing regression tests intentionally exercise actual optimizer updates.
# Retain them, but do not execute them during the default no-training review/CI.
OPTIMIZER_TESTS = {
    'test_atomic_checkpoints_and_resume',
    'test_rejects_wrong_split_hash_before_loading',
    'test_train_and_evaluate_are_finite',
    'test_resumed_weights_equal_uninterrupted_training',
    'test_early_stopping_counts_validation_events',
    'test_validation_loss_can_select_checkpoint',
    'test_restart_matches_uninterrupted_training',
    'test_training_recipe_matches_original_fold_zero',
    'test_budget_stop_retains_checkpoint_and_resumes_exactly',
    'test_matched_initialization_and_selected_checkpoint_replay',
    'test_matched_initialization_training_resume_and_no_target_source_only',
    'test_epoch_boundary_resume_replays_uninterrupted_training',
    'test_cuda_adapter_update_matches_original_on_cpu',
    'test_resume_pair_budget_and_source_only_no_target_forward',
    'test_instrumented_update_exactly_matches_historical_adapter',
    'test_tiny_train_saves_epoch_metrics_best_final_rng_and_never_overwrites',
    'test_reference_exactly_matches_frozen_update',
    'test_no_alignment_never_forwards_or_steps_discriminator_and_loss_reconstructs',
    'test_tiny_ablations_save_best_final_and_effective_coefficients',
    'test_tiny_three_arms_save_selection_final_rng_and_losses',
}

PRIVATE_TESTS = {
    'test_inspect_both_variants',
    'test_load_record_preserves_ignored_epoch',
    'test_rejects_variant_mismatch',
    'test_paths_for_role_match_manifest',
    'test_fold_zero_is_exact_and_disjoint',
    'test_matched_initialization_training_resume_and_no_target_source_only',
    'test_epoch_boundary_resume_replays_uninterrupted_training',
    'test_cuda_adapter_update_matches_original_on_cpu',
    'test_resume_pair_budget_and_source_only_no_target_forward',
    'test_instrumented_update_exactly_matches_historical_adapter',
    'test_tiny_train_saves_epoch_metrics_best_final_rng_and_never_overwrites',
    'test_main_cuda_preflight_reaches_training_boundary_and_exports_safe_stop',
    'test_reference_exactly_matches_frozen_update',
    'test_no_alignment_never_forwards_or_steps_discriminator_and_loss_reconstructs',
    'test_tiny_ablations_save_best_final_and_effective_coefficients',
    'test_tiny_three_arms_save_selection_final_rng_and_losses',
    'test_observed_colab_initial_bytes_reconstructed',
    'test_scientific_execution_and_input_unchanged',
    'test_backup_only_complete_immutable_pairs_and_hashes',
}


def pytest_addoption(parser):
    parser.addoption('--run-private', action='store_true', help='Require local private historical payloads')
    parser.addoption('--run-optimizer-tests', action='store_true', help='Allow synthetic CPU optimizer regression tests')


def pytest_configure(config):
    config.addinivalue_line('markers', 'private_artifacts: requires local historical payloads, never downloaded by tests')
    config.addinivalue_line('markers', 'optimizer_step: runs synthetic optimizer updates; opt-in only')


@pytest.fixture(autouse=True)
def forbid_unrequested_optimizer_steps(request, monkeypatch):
    if request.config.getoption('--run-optimizer-tests'):
        return
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError('Optimizer update blocked: default source tests must not train')
    for name in dir(torch.optim):
        optimizer = getattr(torch.optim, name)
        if isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
            monkeypatch.setattr(optimizer, 'step', forbidden)


def pytest_collection_modifyitems(config, items):
    for item in items:
        name = item.name.split('[')[0]
        if name in PRIVATE_TESTS:
            item.add_marker(pytest.mark.private_artifacts)
        if name in OPTIMIZER_TESTS:
            item.add_marker(pytest.mark.optimizer_step)
        if item.get_closest_marker('private_artifacts') and not config.getoption('--run-private'):
            item.add_marker(pytest.mark.skip(reason='Private historical artifacts: opt in with --run-private'))
        if item.get_closest_marker('optimizer_step') and not config.getoption('--run-optimizer-tests'):
            item.add_marker(pytest.mark.skip(reason='No-training default: opt in with --run-optimizer-tests'))
