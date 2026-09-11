"""Download and validate the official PubMedQA PQA-L dataset."""

from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path

from utils import LABELS, read_json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_DIR = PROJECT_ROOT / "pubmedqa_official"
DATA_DIR = REPOSITORY_DIR / "data"
PUBMEDQA_REPOSITORY = "https://github.com/pubmedqa/pubmedqa.git"


def download_dataset() -> None:
    """Clone PubMedQA only when the local dataset is absent."""
    required_files = (DATA_DIR / "ori_pqal.json", DATA_DIR / "test_set.json")
    if all(path.is_file() for path in required_files):
        return
    if REPOSITORY_DIR.exists():
        raise FileNotFoundError(
            f"{REPOSITORY_DIR} exists but does not contain the required data files."
        )
    subprocess.run(
        ["git", "clone", "--depth", "1", PUBMEDQA_REPOSITORY, str(REPOSITORY_DIR)],
        check=True,
    )


def label_counts(data: dict) -> Counter:
    return Counter(str(row["final_decision"]).lower() for row in data.values())


def main() -> None:
    download_dataset()
    original = read_json(DATA_DIR / "ori_pqal.json")
    test = read_json(DATA_DIR / "test_set.json")
    if len(original) != 1000 or len(test) != 500:
        raise ValueError("Expected 1,000 PQA-L rows and 500 official-test rows.")

    test_pmids = set(test)
    cv_pmids: set[str] = set()
    for fold_index in range(10):
        fold_dir = DATA_DIR / f"pqal_fold{fold_index}"
        train = read_json(fold_dir / "train_set.json")
        validation = read_json(fold_dir / "dev_set.json")
        if len(train) != 450 or len(validation) != 50:
            raise ValueError(f"Fold {fold_index} is not a 450/50 split.")
        if set(train) & set(validation):
            raise ValueError(f"Fold {fold_index} contains train/dev overlap.")
        cv_pmids.update(validation)

    if len(cv_pmids) != 500:
        raise ValueError("The ten validation folds must cover 500 unique rows.")
    if cv_pmids & test_pmids:
        raise ValueError("Cross-validation and final-test PMIDs overlap.")
    if cv_pmids | test_pmids != set(original):
        raise ValueError("The CV and test sets do not cover all PQA-L rows.")

    cv = {pmid: original[pmid] for pmid in cv_pmids}
    print("Split                 Total   Yes    No  Maybe")
    for name, data in (("Cross-validation", cv), ("Final test", test)):
        counts = label_counts(data)
        unknown = set(counts) - set(LABELS)
        if unknown:
            raise ValueError(f"Unexpected labels: {sorted(unknown)}")
        print(
            f"{name:<21}{len(data):>5}{counts['yes']:>6}"
            f"{counts['no']:>6}{counts['maybe']:>7}"
        )
    print("\nDataset split validation passed.")


if __name__ == "__main__":
    main()
