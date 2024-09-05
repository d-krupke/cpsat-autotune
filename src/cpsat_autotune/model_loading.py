from pathlib import Path
from ortools.sat.python import cp_model
from google.protobuf import text_format


def import_model(filepath: Path | str) -> cp_model.CpModel:
    """
    Imports a CP-SAT model from a protobuffer file.

    Args:
        filepath (Path | str): Path to the file containing the model.

    Returns:
        cp_model.CpModel: The imported CP model.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")

    model = cp_model.CpModel()
    with filepath.open("r") as file:
        text_format.Parse(file.read(), model.Proto())

    return model


def export_model(model: cp_model.CpModel, filename: str):
    with open(filename, "w") as file:
        file.write(text_format.MessageToString(model.Proto()))
