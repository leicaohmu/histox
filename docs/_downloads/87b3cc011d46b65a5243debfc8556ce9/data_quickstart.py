"""Run the HistoX public-data quickstart from discovery to provenance."""

import argparse
import json
from pathlib import Path

import histox as hx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("histox-data"),
        help="Dataset cache root (default: ./histox-data)",
    )
    args = parser.parse_args()

    print("Available datasets:")
    for available in hx.data.list_datasets():
        print("-", available.name, available.version, available.access)

    record = hx.data.get_dataset("histox-download-fixture")
    plan = hx.data.plan_download(record, path=args.path)
    print("\nPlan:")
    print(json.dumps(plan.to_dict(), indent=2))

    result = hx.data.download_dataset(record, path=args.path)
    print("\nResult:")
    print("downloaded:", result.downloaded_count)
    print("reused:", result.reused_count)
    print("provenance:", result.provenance_path)


if __name__ == "__main__":
    main()
