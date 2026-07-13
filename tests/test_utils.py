import networkx as nx
import numpy as np
import pytest

from shapcrn.exceptions import NetworkVisualizationError
from shapcrn.utils.graph import all_simple_paths_from_target, plot_network
from shapcrn.utils.utils import z_score_normalize


def test_paths_are_generated_from_target():
    graph = nx.DiGraph([("A", "B"), ("B", "C")])
    paths = list(all_simple_paths_from_target(graph, "A"))
    assert ["A"] in paths
    assert ["A", "B"] in paths
    assert ["A", "B", "C"] in paths


def test_z_score_normalization_divides_by_standard_deviation():
    values = np.array([[1.0, 4.0], [3.0, 8.0]])
    normalized = z_score_normalize(values)
    np.testing.assert_allclose(normalized.mean(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(normalized.std(axis=0), [1.0, 1.0])


def test_network_rendering_reports_optional_dependency(monkeypatch, tmp_path):
    def missing_dependency(_graph):
        raise ImportError("pygraphviz is missing")

    monkeypatch.setattr(nx.nx_agraph, "to_agraph", missing_dependency)
    with pytest.raises(NetworkVisualizationError, match=r"shapcrn\[network\]"):
        plot_network(nx.DiGraph(), tmp_path / "images", tmp_path / "dot")


@pytest.mark.network
def test_network_rendering_with_extra(tmp_path):
    pytest.importorskip("pygraphviz")
    graph = nx.DiGraph()
    graph.add_node("S1", type="species")
    plot_network(graph, tmp_path / "images", tmp_path / "dot")
    assert (tmp_path / "images" / "network.png").is_file()
    assert (tmp_path / "dot" / "network.gv").is_file()
