import ast
import json
import os
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_DIR = ROOT / 'Feature Engineer' / 'Operator'
sys.path.insert(0, str(OPERATOR_DIR))


def notebook_function(path, name, namespace):
    notebook = json.loads(path.read_text())
    source = ''.join(notebook['cells'][0]['source'])
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def test_price_fill_is_chronological_and_does_not_cross_dates(tmp_path):
    load = notebook_function(ROOT / 'Feature Engineer/Factor/Factor/Factor.ipynb', 'load_real_data', {'pd': pd, 'os': os})
    path = tmp_path / 'train.csv'
    pd.DataFrame({
        'stock_id': [1, 1, 1, 1],
        'date_id': [0, 1, 0, 1],
        'seconds_in_bucket': [0, 0, 10, 10],
        'far_price': [float('nan'), 22., float('nan'), float('nan')],
        'near_price': [1., 3., float('nan'), float('nan')],
    }).to_csv(path, index=False)
    result = load(path)
    assert list(zip(result.date_id, result.seconds_in_bucket)) == [(0, 0), (0, 10), (1, 0), (1, 10)]
    assert result.far_price.isna().tolist() == [True, True, False, False]
    assert result.near_price.tolist() == [1., 1., 3., 3.]


def test_training_subset_and_sort_preserve_full_panel():
    from src.feature_pipeline import sort_panel, training_subset
    df = pl.DataFrame({'stock_id': [1, 1, 1, 2], 'date_id': [401, 0, 400, 0], 'seconds_in_bucket': [0, 10, 0, 0], 'v': [4., 1., 3., 2.]})
    ordered = sort_panel(df)
    assert ordered['v'].to_list() == [1., 3., 4., 2.]
    assert training_subset(ordered)['v'].to_list() == [1., 3., 2.]
    assert ordered.height == 4


def test_merge_only_explicit_files_by_identity(tmp_path):
    from src.feature_pipeline import merge_operator_files
    base = pl.DataFrame({'stock_id': [1, 2], 'date_id': [0, 0], 'seconds_in_bucket': [0, 0], 'target': [1., 2.], 'base': [3., 4.]})
    source = tmp_path / 'selected.parquet'
    pl.DataFrame({'stock_id': [2, 1], 'date_id': [0, 0], 'seconds_in_bucket': [0, 0], 'new': [20., 10.]}).write_parquet(source)
    pl.DataFrame({'unwanted': [1]}).write_parquet(tmp_path / 'stale.parquet')
    result = merge_operator_files(base, [source]).collect()
    assert result['new'].to_list() == [10., 20.]
    assert 'unwanted' not in result.columns
    assert result['base'].to_list() == [3., 4.]


def test_merge_rejects_duplicate_or_missing_keys(tmp_path):
    from src.feature_pipeline import merge_operator_files
    base = pl.DataFrame({'stock_id': [1, 2], 'date_id': [0, 0], 'seconds_in_bucket': [0, 0]})
    path = tmp_path / 'bad.parquet'
    pl.DataFrame({'stock_id': [1, 1], 'date_id': [0, 0], 'seconds_in_bucket': [0, 0], 'new': [1, 2]}).write_parquet(path)
    with pytest.raises(ValueError, match='duplicate'):
        merge_operator_files(base, [path])
    pl.DataFrame({'stock_id': [1], 'date_id': [0], 'seconds_in_bucket': [0], 'new': [1]}).write_parquet(path)
    with pytest.raises(ValueError, match='keys'):
        merge_operator_files(base, [path])


@pytest.mark.parametrize('relative_path', [
    'Feature Engineer/Operator/src/single_factor_test.py',
    'Feature Engineer/Factor/Factor/single_factor_test.py',
])
def test_backtest_runs_with_locked_polars(relative_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('aft_single_factor_test', ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    df = pl.DataFrame({
        'date_id': [0, 0, 0, 1, 1, 1],
        'seconds_in_bucket': [0, 0, 0, 0, 0, 0],
        'factor': [1., 2., 3., 1., 2., 3.],
        'target': [1., 2., 3., 1., 2., 3.],
    })
    result = module.backtest(df, ['factor'])
    if isinstance(result, tuple):
        _, result = result
    assert result.loc[0, 'ic'] == pytest.approx(1.)
    assert result.loc[0, 'rankic'] == pytest.approx(1.)


def test_correlation_screening_supports_pandas_copy_on_write():
    from src.utils import filter_correlated_factors
    df = pl.DataFrame({'date_id': [0, 0, 0, 0, 0, 0, 0, 0], 'a': [1., 2., 3., 4., 5., 6., 7., 8.], 'b': [2., 4., 6., 8., 10., 12., 14., 16.]})
    ic = pd.DataFrame({'factor': ['a', 'b'], 'ic': [0.2, 0.1]})
    for method in ('corr_csall', 'corr_cs1'):
        kept, removed = filter_correlated_factors(df, ['a', 'b'], ic, corr_threshold=0.8, method=method)
        assert kept == ['a']
        assert removed == ['b']


def test_rolling_operators_work_with_locked_polars():
    from operators import Operators
    df = pl.DataFrame({'stock_id': [1, 1, 1, 2], 'a': [1., 3., 5., 10.]})
    result = Operators.ts_mean(df, ['a'], windows=[2])
    assert result['ts_mean_a_2'].to_list() == [1., 2., 4., 10.]
    result = Operators.ts_std(df, ['a'], windows=[2])
    assert result['ts_std_a_2'][1] == pytest.approx(2 ** 0.5)


def test_operator_notebook_does_not_silently_skip_failed_operator(tmp_path, monkeypatch):
    notebook = json.loads((OPERATOR_DIR / 'test.ipynb').read_text())
    source = ''.join(notebook['cells'][18]['source'])
    class BrokenOps:
        @staticmethod
        def fail(*args, **kwargs):
            raise ValueError('broken operator')
        ts_rsi = ts_ema = ts_momentum = ts_mean = fail
    monkeypatch.chdir(tmp_path)
    env = {'ops': BrokenOps, 'df': object(), 'cols': ['a'], 'pd': pd, 'Path': Path}
    with pytest.raises(RuntimeError, match='ts_rsi'):
        exec(source, env)


def test_factor_notebook_loads_only_dates_0_through_400(tmp_path):
    load = notebook_function(ROOT / 'Feature Engineer/Factor/Factor/Factor.ipynb', 'load_real_data', {'pd': pd, 'os': os})
    path = tmp_path / 'dates.csv'
    pd.DataFrame({'stock_id': [1, 1, 1, 1], 'date_id': [-1, 0, 400, 401], 'seconds_in_bucket': [0, 0, 0, 0], 'near_price': [1., 2., 3., 4.], 'far_price': [1., 2., 3., 4.]}).to_csv(path,index=False)
    assert load(path).date_id.tolist() == [0, 400]


def test_operator_notebook_loads_only_dates_0_through_400(tmp_path, monkeypatch):
    from src.feature_pipeline import sort_panel
    notebook = json.loads((OPERATOR_DIR/'test.ipynb').read_text())
    factor = tmp_path/'Factor/Factor'; operator = tmp_path/'Operator'
    factor.mkdir(parents=True); operator.mkdir()
    pl.DataFrame({'stock_id':[1,1,1,1], 'date_id':[-1,0,400,401], 'seconds_in_bucket':[0,0,0,0], 'target':[0.,0.,0.,0.], 'signal':[1.,2.,3.,4.]}).write_parquet(factor/'train_auction_features.parquet')
    monkeypatch.chdir(operator)
    env={'pl':pl,'Path':Path,'sort_panel':sort_panel,'clip_extreme_values':lambda df,cols:df,'DEFAULT_EXCLUDE_COLS':['stock_id','date_id','seconds_in_bucket','target','time_id','row_id'],'display':lambda *_:None}
    exec(''.join(notebook['cells'][11]['source']),env)
    assert env['df']['date_id'].to_list() == [0,400]
