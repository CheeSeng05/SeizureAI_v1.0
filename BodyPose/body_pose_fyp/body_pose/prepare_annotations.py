from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .annotation_adapters.factory import create_adapter
from .utils import active_labels, ensure_parent, load_config

LOGGER = logging.getLogger(__name__)


def prepare(config: dict) -> Path:
    adapter_config = config.get("annotation_adapter")
    if not adapter_config:
        raise ValueError("config.annotation_adapter is required.")
    paths = config.get("paths", {})
    source = adapter_config.get("source", paths.get("raw_annotations"))
    if source is None:
        raise ValueError("Set annotation_adapter.source or paths.raw_annotations.")
    output_value = paths.get("canonical_annotations", paths.get("annotations"))
    if output_value is None:
        raise ValueError("Set paths.canonical_annotations.")

    adapter = create_adapter(
        adapter_type=adapter_config.get("type", "generic_tabular"),
        source=source,
        config=adapter_config,
    )
    canonical = adapter.canonical(active_labels=active_labels(config))
    output = ensure_parent(output_value)
    canonical.to_csv(output, index=False)
    LOGGER.info(
        "Saved %d canonical annotation rows from %d patients to %s",
        len(canonical),
        canonical["patient_id"].nunique(),
        output,
    )
    LOGGER.info("Canonical label counts:\n%s", canonical["label"].value_counts())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw dataset annotations to the shared canonical schema."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prepare(load_config(args.config))


if __name__ == "__main__":
    main()
