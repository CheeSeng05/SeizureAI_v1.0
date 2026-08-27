from pathlib import Path

from fyp_data_platform.ingest.bids import discover_bids_files


def test_discovers_filtered_bids_and_annex_pointer(bids_root: Path) -> None:
    files = discover_bids_files(bids_root, subjects={"001"})
    assert {item.subject_id for item in files if item.subject_id} == {"sub-001"}
    edf = next(item for item in files if item.path.suffix == ".edf")
    assert edf.payload_state == "annex-pointer"
    assert not edf.materializable
    assert next(item for item in files if item.path.name.endswith("_events.tsv")).role == "annotation"
