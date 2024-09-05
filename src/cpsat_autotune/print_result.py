from collections.abc import Callable
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.rule import Rule

from .metrics import Metric
from .caching_solver import MultiResult

from .cpsat_parameters import get_parameter_by_name


console = Console()


def print_results(
    result, default_score: MultiResult, metric: Metric, fn: Callable = console.print
) -> None:
    """
    Prints the evaluation results in a professional format.
    """
    fn(Rule("OPTIMIZED PARAMETERS", align="center"))

    if not result.optimized_params:
        fn(Panel("No significant parameter changes were identified.", style="bold red"))
    else:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Parameter", style="bold")
        table.add_column("Value", justify="center")
        table.add_column("Contribution", justify="center")
        table.add_column("Default Value", justify="center")

        # Accumulate descriptions separately
        descriptions = []

        for i, (key, value) in enumerate(result.optimized_params.items(), start=1):
            default_value = get_parameter_by_name(key).get_cpsat_default()
            description = get_parameter_by_name(key).description.strip()
            contribution_value = (
                f"{result.contribution.get(key, '<NA>'):.2%}"
                if key in result.contribution
                else "<NA>"
            )

            table.add_row(
                str(i),
                key,
                str(value),
                contribution_value,
                str(default_value),
            )

            # Append the description to the list with proper formatting
            descriptions.append(f"**{i}. {key}**\n{description}\n")

        fn(table)
        fn(Rule("Descriptions", align="center"))
        fn(Markdown("\n".join(descriptions)))  # Use Markdown for proper rendering

    fn(Rule())

    metrics_table = Table(show_header=True, header_style="bold green")
    metrics_table.add_column("Metric", style="bold green")
    metrics_table.add_column("Mean", justify="right")
    metrics_table.add_column("Min", justify="right")
    metrics_table.add_column("Max", justify="right")
    metrics_table.add_column("#Samples", justify="right")

    metrics_table.add_row(
        f"{metric.objective_name()} with Default Parameters",
        str(round(default_score.mean(), 2)),
        str(round(default_score.min(), 2)),
        str(round(default_score.max(), 2)),
        str(len(default_score)),
    )
    metrics_table.add_row(
        f"{metric.objective_name()} with Optimized Parameters",
        str(round(result.optimized_score.mean(), 2)),
        str(round(result.optimized_score.min(), 2)),
        str(round(result.optimized_score.max(), 2)),
        str(len(result.optimized_score)),
    )

    fn(metrics_table)

    fn(Rule())

    warning_message = """
        The optimized parameters listed above were obtained based on a sampling approach
        and may not fully capture the complexities of the entire problem space.
        While statistical reasoning has been applied, these results should be considered
        as a suggestion for further evaluation rather than definitive settings.

        It is strongly recommended to validate these parameters in larger, more comprehensive
        experiments before adopting them in critical applications.
        """
    fn(Panel(Markdown(warning_message), title="WARNING", style="bold yellow"))
