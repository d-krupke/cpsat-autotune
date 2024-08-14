from pathlib import Path
from ortools.sat.python import cp_model
from google.protobuf import text_format


def import_model(filepath: Path | str) -> cp_model.CpModel:
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")
    model = cp_model.CpModel()
    with filepath.open("r") as file:
        text_format.Parse(file.read(), model.Proto())
    return model
