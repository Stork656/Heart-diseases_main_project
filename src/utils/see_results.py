from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parents[2] / "results"


def show_results(path : Path) -> None:
    """
    Quick view of the results
    Parameters:
        path : Path
            Path to the folder containing the results
    """
    files = path.glob("*.csv")
    for file in files:
        print(f"\n{file.name}:")
        df = pd.read_csv(file)
        print(df.to_string(index=False))


if __name__ == "__main__":
    show_results(path)
