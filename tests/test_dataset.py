from pathlib import Path

from jssp_yafs.data.loader import load_instance_txt, parse_instance_corpus
from jssp_yafs.data.prepare import prepare_subsets


def test_parse_corpus_has_known_instances() -> None:
    corpus = parse_instance_corpus("data/raw/instance_data.txt")
    assert "ft06" in corpus
    assert "la01" in corpus
    assert corpus["ft06"].n_jobs == 6
    assert corpus["ft06"].n_machines == 6



def test_prepare_and_load_subset(tmp_path: Path) -> None:
    prepare_subsets(
        source_file="data/raw/instance_data.txt",
        quick_instances=["ft06"],
        full_instances=["la01"],
        out_root=tmp_path,
    )
    inst = load_instance_txt(tmp_path / "quick" / "ft06.txt", name="ft06")
    assert inst.n_jobs == 6
    assert inst.n_machines == 6
