import os
import pandas as pd
import pytest

from load_graph import get_nodes_df, get_edges_df, get_chargers_df, load_graph

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


@pytest.fixture(scope="session", autouse=True)
def _set_cwd_to_project_root():
    prev = os.getcwd()
    os.chdir(_PROJECT_ROOT)
    yield
    os.chdir(prev)


@pytest.fixture
def nodes_df():
    return get_nodes_df()


@pytest.fixture
def edges_df():
    return get_edges_df()


@pytest.fixture
def chargers_df():
    return get_chargers_df()


@pytest.fixture
def graph():
    return load_graph()


def test_nodes_dataframe_has_expected_columns(nodes_df):
    assert not nodes_df.empty
    expected_node_cols = {'node_id', 'City', 'Latitude', 'Longitude'}
    assert expected_node_cols <= set(nodes_df.columns)


def test_edges_dataframe_has_expected_columns(edges_df):
    assert not edges_df.empty
    expected_edge_cols = {'from_id', 'to_id', 'distance_km', 'is_bidirectional'}
    assert expected_edge_cols <= set(edges_df.columns)


def test_chargers_dataframe_has_expected_columns(chargers_df):
    assert not chargers_df.empty
    expected_charger_cols = {'node_id', 'charge_rate_kw', 'price_per_kwh'}
    assert expected_charger_cols <= set(chargers_df.columns)


def test_expected_nodes(nodes_df, graph):
    assert len(graph.nodes) == len(nodes_df)


def test_expected_edges(edges_df, graph):
    assert len(graph.edges) == 2 * len(edges_df[edges_df['is_bidirectional']]) + len(edges_df[~edges_df['is_bidirectional']])


def test_expected_chargers(chargers_df, graph):
    assert sum(1 for _, d in graph.nodes(data=True) if d.get('is_charger')) == len(chargers_df)


def test_invalid_charger_node_is_skipped(monkeypatch, nodes_df):
    invalid_node_id = nodes_df['node_id'].max() + 1
    fake_chargers_df = pd.DataFrame({'node_id': [invalid_node_id], 'charge_rate_kw': [50], 'price_per_kwh': [0.15]})
    monkeypatch.setattr('load_graph.get_chargers_df', lambda: fake_chargers_df)

    graph = load_graph()
    assert invalid_node_id not in graph.nodes
    assert len(graph.nodes) == len(nodes_df)