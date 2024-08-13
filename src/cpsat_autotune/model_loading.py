from pathlib import Path
from ortools.sat.python import cp_model
from google.protobuf import text_format

def import_model(filepath: Path) -> cp_model.CpModel:
    model = cp_model.CpModel()
    with filepath.open("r") as file:
        text_format.Parse(file.read(), model.Proto())
    return model