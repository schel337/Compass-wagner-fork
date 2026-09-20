"""
Unit tests for compass.utils.
"""
import os

import numpy as np
import pandas as pd

import compass
from compass import utils

TEST_DATA_DIR = os.path.join(os.path.dirname(compass.__file__), "Resources", "Test-Data")


def test_read_data_tsv():
    path = os.path.join(TEST_DATA_DIR, "tsv_format", "expression.tsv")
    data = utils.read_data([path])
    assert isinstance(data, pd.DataFrame)
    # 1 gene column + 40 cells (the Th17 example dataset)
    assert data.shape[0] > 40000
    assert data.shape[1] == 40
    assert np.issubdtype(data.values.dtype, np.floating)


def test_read_data_mtx():
    d = os.path.join(TEST_DATA_DIR, "mtx_format")
    data = utils.read_data([
        os.path.join(d, "matrix.mtx"),
        os.path.join(d, "genes.tsv"),
        os.path.join(d, "barcodes.tsv"),
    ])
    assert isinstance(data, pd.DataFrame)
    with open(os.path.join(d, "barcodes.tsv")) as f:
        n_cells = sum(1 for _ in f)
    assert data.shape[1] == n_cells
    # Genes file has a header-less column of names
    with open(os.path.join(d, "genes.tsv")) as f:
        n_genes = sum(1 for _ in f)
    assert data.shape[0] == n_genes


def test_read_data_h5ad(tmp_path):
    import anndata as ad

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=["CELL_0", "CELL_1"]),
                       var=pd.DataFrame(index=["GENE_A", "GENE_B"]))
    path = tmp_path / "test.h5ad"
    adata.write_h5ad(str(path))

    data = utils.read_data([str(path)])
    assert isinstance(data, pd.DataFrame)
    # read_data returns genes x cells (transposed)
    assert data.shape == (2, 2)
    assert list(data.columns) == ["CELL_0", "CELL_1"]
    assert list(data.index) == ["GENE_A", "GENE_B"]
    np.testing.assert_array_equal(data.values, X.T)


def test_parse_gurobi_license_file(tmp_path):
    p = tmp_path / "gurobi.lic"
    p.write_text(
        "WLSACCESSID=123456\n"
        "WLSSECRET=abcdef\n"
        "LICENSEID=789012\n"
    )
    creds = utils.parse_gurobi_license_file(str(p))
    assert creds["WLSACCESSID"] == 123456
    assert creds["WLSSECRET"] == "abcdef"
    assert creds["LICENSEID"] == 789012


def test_parse_gurobi_license_file_missing_file(tmp_path):
    try:
        utils.parse_gurobi_license_file(str(tmp_path / "nonexistent.lic"))
    except Exception:
        pass
    else:
        raise AssertionError("Expected error for missing license file")
