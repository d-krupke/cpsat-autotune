from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Union

from cpsat_autotune.caching_solver import CachingScorer

from .cpsat_parameters import get_parameter_by_name

from .metrics import Comparison, Metric

def log(message: str) -> None:
    """
    Logs a message to the console.
    """
    print(message)

@dataclass
class EvaluationResult:
    """
    Data class that stores the results of the parameter evaluation.
    """
    optimized_params: Dict[str, Union[int, bool, float, list, tuple]]
    contribution: Dict[str, float]
    optimized_score: float

    def print_results(self, default_score) -> None:
        """
        Prints the evaluation results in a professional format.
        """
        print("============================================================")
        print("                 OPTIMIZED PARAMETERS")
        print("============================================================")
        if not self.optimized_params:
            print("No significant parameter changes were identified.")
        for i, (key, value) in enumerate(self.optimized_params.items(), start=1):
            default_value = get_parameter_by_name(key).get_cpsat_default()
            description = get_parameter_by_name(key).description.strip().replace('\n', '\n\t\t')
            contribution_value = f"{self.contribution[key]:.2%}" if key in self.contribution else "<NA>"
            print(f"\n{i}. {key}: {value}")
            print(f"\tContribution: {contribution_value}")
            print(f"\tDefault Value: {default_value}")
            print(f"\tDescription:\n\t\t{description}")

        print("------------------------------------------------------------")
        print(f"Default Metric Value: {default_score}")
        print(f"Optimized Metric Value: {self.optimized_score}")
        print("------------------------------------------------------------")
        
        print("\n============================================================")
        print("*** WARNING ***")
        print("============================================================")
        print(
            "The optimized parameters listed above were obtained based on a sampling approach "
            "and may not fully capture the complexities of the entire problem space. "
            "While statistical reasoning has been applied, these results should be considered "
            "as a suggestion for further evaluation rather than definitive settings.\n"
            "It is strongly recommended to validate these parameters in larger, more comprehensive "
            "experiments before adopting them in critical applications."
        )
        print("============================================================")


class ParameterEvaluator:
    """
    Evaluates the impact of parameter changes on the model's performance.
    Identifies essential parameters and determines whether suggested parameters
    offer significant improvements over the defaults.
    """

    def __init__(
        self,
        params: Dict[str, Union[int, bool, float, list, tuple]],
        scorer: CachingScorer,
        metric: Metric,
        min_baseline_size: int = 10,
    ) -> None:
        self.params = params
        self.scorer = scorer
        self.metric = metric
        self.results = defaultdict(list)
        self.min_baseline_size = min_baseline_size

    def _generate_variants(
        self, params: Dict[str, Union[int, bool, float, list, tuple]]
    ) -> Iterable[tuple[str, dict[str, Union[int, bool, float, list, tuple]]]]:
        """
        Generates solver variants by excluding one parameter at a time.
        """
        for key in params:
            reduced_params = {k: v for k, v in params.items() if k != key}
            yield key, reduced_params

    def evaluate(self) -> EvaluationResult:
        """
        Evaluates the impact of excluding each parameter individually, identifies
        the optimized set of parameters, and returns an EvaluationResult object
        containing the results.
        """
        log("Checking which parameter changes obtained by the hyperparameter optimization are essential...")
        optuna_baseline = self.scorer.evaluate(self.params, num_runs=self.min_baseline_size)
        print("optuna_baseline", optuna_baseline)
        optimized_params = {}
        diffs = {}

        for key, params in self._generate_variants(self.params):
            log(f"Evaluating resetting parameter '{key}' to default..." )
            score = self.scorer.evaluate(params, num_runs=1, knockout_score=self.metric.worst(optuna_baseline))
            print("score", score)
            if self.metric.comp(self.metric.best(score), self.metric.worst(optuna_baseline)) in (Comparison.EQUAL, Comparison.BETTER):
                # The score did not degrade. Note: We compare best to worst to be conservative regarding deviating from the default.
                log(f"Seems like we can drop parameter '{key}' from the optimized parameters.")
                continue
            else:
                log(f"The parameter '{key}' seems to be essential for the performance.")
                optimized_params[key] = self.params[key]
                diffs[key] = abs(optuna_baseline.mean() - score.mean())

        # Calculate parameter significance
        total_diff = sum(diffs.values())
        significance = {key: diff / total_diff for key, diff in diffs.items()}

        # Final evaluation with optimized parameters
        optimized_score = self.scorer.evaluate(optimized_params, num_runs=self.min_baseline_size)
        print("optimized_score", optimized_score)

        if self.metric.comp(self.metric.worst(optuna_baseline),self.metric.best(optimized_score)) == Comparison.BETTER:
            # revert to initial parameters as the seems to be some difficult correlations
            log("The final evaluation indicates that dropping all of the seemingly uninfluential parameters did worsen the performance. Reverting to the initial parameters.")
            optimized_params = self.params
            optimized_score = optuna_baseline
            significance = {}


        # Convert the metrics before storing them in the result
        return EvaluationResult(
            optimized_params=optimized_params,
            contribution=significance,
            optimized_score=optimized_score.mean(),
        )
