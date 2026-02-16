from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence

import pandas as pd

from factors import FactorDefinition

SeriesFn = Callable[[pd.DataFrame], pd.Series]
PreStep = Callable[[pd.DataFrame], pd.DataFrame]
KwargFn = Callable[[str], Mapping[str, object]]
ColumnMapping = Mapping[str, str]


def ensure_series_name(series: pd.Series, default: str = "factor") -> pd.Series:
    result = series.copy()
    result.name = series.name or default
    return result


def default_series_builder(
    entity_col: str,
    builder: Callable[..., pd.DataFrame],
    *,
    pre_steps: Sequence[PreStep] | None = None,
    kwarg_fn: KwargFn | None = None,
) -> Callable[[str], SeriesFn]:
    steps = tuple(pre_steps or ())

    def factory(column_name: str) -> SeriesFn:
        def fn(df: pd.DataFrame) -> pd.Series:
            tmp = df.copy()
            for step in steps:
                tmp = step(tmp)
            kwargs = {"group_col": entity_col}
            if kwarg_fn:
                kwargs.update(kwarg_fn(column_name))
            tmp = builder(tmp, **kwargs)
            return ensure_series_name(tmp[column_name])

        return fn

    return factory


def register_columns(
    factory: Callable[[str], SeriesFn],
    column_mapping: ColumnMapping,
) -> Dict[str, FactorDefinition]:
    return {
        column: FactorDefinition(name=column, function=factory(column), description=description)
        for column, description in column_mapping.items()
    }


@dataclass(frozen=True)
class FactorGroupSpec:
    builder: Callable[..., pd.DataFrame]
    columns: ColumnMapping | Callable[[], ColumnMapping]
    pre_steps: Sequence[PreStep] | None = None
    pre_steps_factory: Callable[[str], Sequence[PreStep]] | None = None
    kwargs_factory: KwargFn | None = None
    inputs: Sequence[str] | None = None
    column_inputs: Mapping[str, Sequence[str]] | None = None


def build_factor_definitions_from_specs(
    specs: Sequence[FactorGroupSpec],
    *,
    entity_col: str,
) -> Dict[str, FactorDefinition]:
    registry: Dict[str, FactorDefinition] = {}
    for spec in specs:
        pre_steps = spec.pre_steps_factory(entity_col) if spec.pre_steps_factory else spec.pre_steps
        factory = default_series_builder(
            entity_col,
            spec.builder,
            pre_steps=pre_steps,
            kwarg_fn=spec.kwargs_factory,
        )
        column_mapping = spec.columns() if callable(spec.columns) else spec.columns
        registry.update(register_columns(factory, column_mapping))
    return registry


REGISTERED_GROUPS: Dict[str, List[FactorGroupSpec]] = defaultdict(list)
GROUP_DEPENDENCIES: Dict[str, Dict[str, List[str]]] = defaultdict(dict)


def register_factor_group(
    group_name: str,
    *,
    columns: ColumnMapping | Callable[[], ColumnMapping],
    pre_steps: Sequence[PreStep] | None = None,
    pre_steps_factory: Callable[[str], Sequence[PreStep]] | None = None,
    kwargs_factory: KwargFn | None = None,
    inputs: Sequence[str] | None = None,
    column_inputs: Mapping[str, Sequence[str]] | None = None,
) -> Callable[[Callable[..., pd.DataFrame]], Callable[..., pd.DataFrame]]:
    """Decorator to register a builder function under a named factor group."""

    def decorator(builder: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        REGISTERED_GROUPS[group_name].append(
            FactorGroupSpec(
                builder=builder,
                columns=columns,
                pre_steps=pre_steps,
                pre_steps_factory=pre_steps_factory,
                kwargs_factory=kwargs_factory,
                inputs=inputs,
                column_inputs=column_inputs,
            )
        )
        return builder

    return decorator


def build_factor_definitions_for_group(
    group_name: str,
    *,
    entity_col: str,
) -> Dict[str, FactorDefinition]:
    specs = REGISTERED_GROUPS.get(group_name, [])
    registry = build_factor_definitions_from_specs(specs, entity_col=entity_col)

    # refresh dependency graph entries for this group
    GROUP_DEPENDENCIES.pop(group_name, None)
    dependencies: Dict[str, List[str]] = {}
    for spec in specs:
        column_mapping = spec.columns() if callable(spec.columns) else spec.columns
        base_inputs = list(spec.inputs or [])
        per_column_inputs = spec.column_inputs or {}
        for column in column_mapping.keys():
            deps = per_column_inputs.get(column, base_inputs)
            dependencies[column] = list(deps) if deps else []
    GROUP_DEPENDENCIES[group_name] = dependencies

    return registry


def clear_factor_groups(group_name: str | None = None) -> None:
    if group_name is None:
        REGISTERED_GROUPS.clear()
        GROUP_DEPENDENCIES.clear()
    else:
        REGISTERED_GROUPS.pop(group_name, None)
        GROUP_DEPENDENCIES.pop(group_name, None)


def get_factor_dependencies(group_name: str) -> Dict[str, List[str]]:
    """Return the dependency mapping (factor -> referenced inputs) for a factor group."""

    return GROUP_DEPENDENCIES.get(group_name, {}).copy()
