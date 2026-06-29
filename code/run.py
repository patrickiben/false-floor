"""One command: check data, train the ensemble, run the kill-test."""
import os
import evaluate

if not os.path.exists("data/mushroom.csv"):
    raise SystemExit(
        "Missing data/mushroom.csv -- download it:\n"
        "  mkdir -p data && curl -fsSL -o data/mushroom.csv "
        "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    )
evaluate.main()
