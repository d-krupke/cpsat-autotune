import click
from .model_loading import import_model
from .tune import (
    tune_for_gap_within_timelimit,
    tune_time_to_optimal,
    tune_for_quality_within_timelimit,
)

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """CLI for CP-SAT hyperparameter tuning."""
    pass


def _estimate_time(max_time, n_trials, n_samples):
    expected_time = max_time * n_samples * n_trials
    # convert to hours and minutes
    hours = int(expected_time // 3600)
    minutes = int((expected_time % 3600) // 60)
    if hours > 0:
        logging.info(
            f"The expected time for the tuning process is {hours} hours and {minutes} minutes."
        )
    else:
        logging.info(f"The expected time for the tuning process is {minutes} minutes.")
    logging.info(
        "The tuning algorithm will try to take shortcuts whenever possible, potentially reducing the time drastically."
    )
    logging.info(
        "To reduce the expected time, you can try to reduce the number of trials or samples per trial, as well as the maximum time allowed for each solve operation. However, this may affect the reliability of the tuning process."
    )


@click.command(
    help="""
    Tune CP-SAT hyperparameters to minimize the time required to find an optimal solution.

    This command tunes the hyperparameters of a CP-SAT model to minimize the time required to find an optimal solution.
    You need to provide the path to the model file and specify the maximum time allowed for each solve operation,
    the relative optimality gap, and the number of trials and samples for the tuning process.
    """
)
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--max-time",
    type=float,
    required=True,
    help="Maximum time allowed for each solve operation in seconds.",
)
@click.option(
    "--relative-gap",
    type=float,
    default=0.0,
    help="Relative optimality gap for considering a solution as optimal.",
)
@click.option(
    "--n-trials",
    type=int,
    default=100,
    help="Number of trials to execute in the tuning process.",
)
@click.option(
    "--n-samples-trial",
    type=int,
    default=10,
    help="Number of samples to take in each trial.",
)
@click.option(
    "--n-samples-verification",
    type=int,
    default=30,
    help="Number of samples for verifying parameters.",
)
def time(
    model_path,
    max_time,
    relative_gap,
    n_trials,
    n_samples_trial,
    n_samples_verification,
):
    """Tune CP-SAT hyperparameters to minimize the time required to find an optimal solution."""
    _estimate_time(max_time, n_trials, n_samples_trial)
    model = import_model(model_path)
    best_params = tune_time_to_optimal(
        model=model,
        max_time_in_seconds=max_time,
        relative_gap_limit=relative_gap,
        n_samples_for_trial=n_samples_trial,
        n_samples_for_verification=n_samples_verification,
        n_trials=n_trials,
    )
    click.echo(f"Best parameters: {best_params}")


@click.command(
    help="""
    Tune CP-SAT hyperparameters to maximize or minimize solution quality within a given time limit.

    This command tunes the hyperparameters of a CP-SAT model to optimize the solution quality within a specified time limit.
    You need to provide the path to the model file, the maximum time allowed for each solve operation, the objective value
    to return if the solver times out, and the direction to optimize the objective value. Additionally, you can specify
    the number of trials and samples for the tuning process.
    """
)
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--max-time",
    type=float,
    required=True,
    help="Time limit for each solve operation in seconds.",
)
@click.option(
    "--obj-for-timeout",
    type=int,
    required=True,
    help="Objective value to return if the solver times out.",
)
@click.option(
    "--direction",
    type=click.Choice(["maximize", "minimize"]),
    required=True,
    help="Direction to optimize the objective value.",
)
@click.option(
    "--n-trials",
    type=int,
    default=100,
    help="Number of trials to execute in the tuning process.",
)
@click.option(
    "--n-samples-trial",
    type=int,
    default=10,
    help="Number of samples to take in each trial.",
)
@click.option(
    "--n-samples-verification",
    type=int,
    default=30,
    help="Number of samples for verifying parameters.",
)
def quality(
    model_path,
    max_time,
    obj_for_timeout,
    direction,
    n_trials,
    n_samples_trial,
    n_samples_verification,
):
    """Tune CP-SAT hyperparameters to maximize or minimize solution quality within a given time limit."""
    _estimate_time(max_time, n_trials, n_samples_trial)
    model = import_model(model_path)
    best_params = tune_for_quality_within_timelimit(
        model=model,
        max_time_in_seconds=max_time,
        obj_for_timeout=obj_for_timeout,
        direction=direction,
        n_samples_for_trial=n_samples_trial,
        n_samples_for_verification=n_samples_verification,
        n_trials=n_trials,
    )
    click.echo(f"Best parameters: {best_params}")


@click.command(
    help="""
    Tune CP-SAT hyperparameters to minimize the gap within a given time limit.

    This command tunes the hyperparameters of a CP-SAT model to minimize the gap within a specified time limit.
    It is useful for complex models where finding the optimal solution within the time limit is not feasible,
    but you still want some guarantee on the quality of the solution.
    """
)
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--max-time",
    type=float,
    required=True,
    help="The time limit for each solve operation in seconds.",
)
@click.option(
    "--n-samples-trial",
    type=int,
    default=10,
    help="The number of samples to take in each trial. Defaults to 10.",
)
@click.option(
    "--n-samples-verification",
    type=int,
    default=30,
    help="The number of samples for verifying parameters. Defaults to 30.",
)
@click.option(
    "--n-trials",
    type=int,
    default=100,
    help="The number of trials to execute in the tuning process. Defaults to 100.",
)
@click.option(
    "--limit", type=int, default=10, help="The limit for the gap. Defaults to 10."
)
def gap(model_path, max_time, n_samples_trial, n_samples_verification, n_trials, limit):
    """Tune CP-SAT hyperparameters to minimize the gap within a given time limit."""
    _estimate_time(max_time, n_trials, n_samples_trial)
    model = import_model(model_path)
    best_params = tune_for_gap_within_timelimit(
        model=model,
        max_time_in_seconds=max_time,
        n_samples_for_trial=n_samples_trial,
        n_samples_for_verification=n_samples_verification,
        n_trials=n_trials,
        limit=limit,
    )
    click.echo(f"Best parameters: {best_params}")


cli.add_command(time)
cli.add_command(quality)
cli.add_command(gap)

if __name__ == "__main__":
    cli()
