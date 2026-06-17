import numpy as np
from HoloNew.evaluation.export.csv_writer import write_csv


def test_writes_header_and_rows_roundtrip(tmp_path):
    header = ["time", "dynamics/com/x"]
    table = np.array([[0.0, 1.0], [0.1, 2.0], [0.2, 3.0]])
    path = tmp_path / "nested" / "run_signals.csv"
    write_csv(path, header, table)

    text = path.read_text().splitlines()
    assert text[0] == "time,dynamics/com/x"
    assert len(text) == 4  # header + 3 rows

    read = np.loadtxt(path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(read, table)
