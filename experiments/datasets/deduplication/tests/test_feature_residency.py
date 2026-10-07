from __future__ import annotations

from collections import OrderedDict

from deduplication.verification import BoundedFeatureStore


def test_resident_query_and_transient_reference_transfer_once(tmp_path):
    store = object.__new__(BoundedFeatureStore)
    store.extractor = None
    store.device = "cuda"
    store.feature_dir = tmp_path
    store.capacity = 2
    store.memory = OrderedDict()
    store.device_memory = {"query": "query-gpu"}
    store.transient_device = None
    loads = []
    transfers = []
    store._load = lambda path: loads.append(path.name) or "reference-cpu"
    store._to_device = lambda value, device: transfers.append((value, device)) or "reference-gpu"
    store._remember = lambda key, value: store.memory.__setitem__(key, value)
    (tmp_path / "reference.pt").write_bytes(b"fixture")

    for _ in range(10):
        assert store.get("query", tmp_path / "unused.jpg") == "query-gpu"
        assert store.get("reference", tmp_path / "unused.jpg") == "reference-gpu"
    assert loads == ["reference.pt"]
    assert transfers == [("reference-cpu", "cuda")]
