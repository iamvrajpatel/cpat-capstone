"""Configuration schema, validation, and parameter-sweep helpers for the UI."""

from __future__ import annotations

import itertools
import json
import types
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import yaml
from pydantic import BaseModel, ValidationError

from cpat.config.loader import CPATConfig, load_config


WidgetType = Literal["checkbox", "selectbox", "slider", "number_input", "text_input", "list_editor"]
SweepMode = Literal["fixed", "list", "range", "scenario"]


@dataclass(frozen=True)
class UIFieldSpec:
    """Metadata required to render a config field dynamically."""

    path: str
    label: str
    section: str
    widget_type: WidgetType
    python_type: str
    default: Any
    value: Any
    required: bool
    description: str = ""
    options: tuple[Any, ...] = ()
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None


@dataclass(frozen=True)
class SweepFieldSpec:
    """UI overlay definition for one sweepable field."""

    path: str
    mode: SweepMode = "fixed"
    values: tuple[Any, ...] = ()
    start: float | int | None = None
    stop: float | int | None = None
    step: float | int | None = None
    scenario_name: str = ""

    def resolved_values(self, base_value: Any) -> list[Any]:
        """Return the concrete values implied by this overlay."""
        if self.mode == "fixed":
            return [base_value]
        if self.mode in {"list", "scenario"}:
            return list(self.values) or [base_value]
        if self.mode == "range":
            if self.start is None or self.stop is None or self.step in (None, 0):
                return [base_value]
            values: list[Any] = []
            current = self.start
            if isinstance(self.start, int) and isinstance(self.stop, int) and isinstance(self.step, int):
                while current <= self.stop:
                    values.append(int(current))
                    current += self.step
            else:
                while float(current) <= float(self.stop) + 1e-12:
                    values.append(round(float(current), 10))
                    current = float(current) + float(self.step)
            return values or [base_value]
        return [base_value]


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation response for UI save/run flows."""

    is_valid: bool
    config: CPATConfig | None = None
    errors: tuple[str, ...] = ()


class ConfigService:
    """Configuration and schema utilities for the Streamlit UI."""

    def __init__(self, config_path: str | Path = "config/settings.yaml") -> None:
        self._config_path = Path(config_path)

    @property
    def config_path(self) -> Path:
        """Return the canonical config path."""
        return self._config_path

    def load_raw_config(self, path: str | Path | None = None) -> dict[str, Any]:
        """Load raw YAML config into a plain nested dict."""
        target = Path(path) if path else self._config_path
        with target.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def load_config(self, path: str | Path | None = None) -> CPATConfig:
        """Load the typed CPAT config."""
        return load_config(path or self._config_path)

    def save_config(self, raw_config: dict[str, Any], path: str | Path | None = None) -> Path:
        """Persist a raw config dict back to YAML."""
        target = Path(path) if path else self._config_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(raw_config, handle, sort_keys=False)
        return target

    def export_preset(self, raw_config: dict[str, Any]) -> str:
        """Export config to a YAML string for downloads."""
        return yaml.safe_dump(raw_config, sort_keys=False)

    def validate_config(self, raw_config: dict[str, Any]) -> ValidationResult:
        """Validate a raw config dict against the typed schema."""
        try:
            config = CPATConfig.model_validate(raw_config)
            return ValidationResult(is_valid=True, config=config)
        except ValidationError as exc:
            errors = tuple(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}"
                for err in exc.errors()
            )
            return ValidationResult(is_valid=False, errors=errors)

    def flatten_config(
        self,
        data: dict[str, Any],
        prefix: str = "",
    ) -> dict[str, Any]:
        """Flatten a nested config dict into dotted paths."""
        flat: dict[str, Any] = {}
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self.flatten_config(value, path))
            else:
                flat[path] = value
        return flat

    def unflatten_config(self, flat: dict[str, Any]) -> dict[str, Any]:
        """Expand dotted-path overrides back into a nested dict."""
        nested: dict[str, Any] = {}
        for path, value in flat.items():
            cursor = nested
            parts = path.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
        return nested

    def apply_overrides(
        self,
        base_config: dict[str, Any],
        overrides_flat: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a deep-copied config with dotted-path overrides applied."""
        merged = deepcopy(base_config)
        for path, value in overrides_flat.items():
            cursor = merged
            parts = path.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
        return merged

    def build_ui_schema(
        self,
        raw_config: dict[str, Any] | None = None,
        model_cls: type[BaseModel] = CPATConfig,
    ) -> list[UIFieldSpec]:
        """Build a leaf-field UI schema by recursively inspecting the config model."""
        current_values = raw_config or self.load_raw_config()
        return self._build_model_schema(model_cls, current_values)

    def generate_parameter_grid(
        self,
        base_config: dict[str, Any],
        sweep_specs: list[SweepFieldSpec],
    ) -> list[dict[str, Any]]:
        """Generate Cartesian config combinations from UI overlay specs."""
        if not sweep_specs:
            return [deepcopy(base_config)]

        base_flat = self.flatten_config(base_config)
        variable_paths: list[str] = []
        value_lists: list[list[Any]] = []
        for spec in sweep_specs:
            base_value = base_flat.get(spec.path)
            values = spec.resolved_values(base_value)
            variable_paths.append(spec.path)
            value_lists.append(values)

        combinations: list[dict[str, Any]] = []
        for combo in itertools.product(*value_lists):
            overrides = dict(zip(variable_paths, combo))
            combinations.append(self.apply_overrides(base_config, overrides))
        return combinations

    def export_sweep_specs(self, sweep_specs: dict[str, dict[str, Any]]) -> str:
        """Serialise sweep overlays for download."""
        return json.dumps(sweep_specs, indent=2, sort_keys=True)

    def parse_list_value(self, raw_value: str, target_type: str) -> list[Any]:
        """Parse a multiline/token list editor value into typed scalars."""
        tokens = [
            token.strip()
            for line in raw_value.splitlines()
            for token in line.split(",")
            if token.strip()
        ]
        if target_type == "int":
            return [int(token) for token in tokens]
        if target_type == "float":
            return [float(token) for token in tokens]
        if target_type == "bool":
            return [token.lower() in {"1", "true", "yes", "on"} for token in tokens]
        return tokens

    def _build_model_schema(
        self,
        model_cls: type[BaseModel],
        values: dict[str, Any],
        prefix: str = "",
        section: str = "",
    ) -> list[UIFieldSpec]:
        fields: list[UIFieldSpec] = []
        for name, field in model_cls.model_fields.items():
            path = f"{prefix}.{name}" if prefix else name
            field_section = section or prefix.split(".")[0] if prefix else path.split(".")[0]
            annotation = field.annotation
            raw_value = values.get(name, field.default)
            nested_model = self._resolve_model_type(annotation)
            if nested_model is not None:
                nested_values = raw_value if isinstance(raw_value, dict) else {}
                fields.extend(
                    self._build_model_schema(
                        nested_model,
                        nested_values,
                        prefix=path,
                        section=field_section if prefix else name,
                    )
                )
                continue

            options = self._extract_options(annotation)
            widget_type = self._infer_widget_type(annotation, raw_value, options)
            minimum, maximum = self._extract_bounds(field.metadata)
            step = self._infer_step(annotation, raw_value)
            python_type = self._python_type_name(annotation, raw_value)
            default = None if field.default is Ellipsis else field.default
            required = field.is_required()
            description = field.description or ""
            fields.append(
                UIFieldSpec(
                    path=path,
                    label=name.replace("_", " ").title(),
                    section=field_section if field_section else path.split(".")[0],
                    widget_type=widget_type,
                    python_type=python_type,
                    default=default,
                    value=raw_value,
                    required=required,
                    description=description,
                    options=tuple(options),
                    minimum=minimum,
                    maximum=maximum,
                    step=step,
                )
            )
        return fields

    def _resolve_model_type(self, annotation: Any) -> type[BaseModel] | None:
        """Return a nested Pydantic model type if the annotation represents one."""
        origin = get_origin(annotation)
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        candidate = args[0] if origin in {types.UnionType, getattr(__import__("typing"), "Union")} and args else annotation
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
        return None

    def _extract_options(self, annotation: Any) -> list[Any]:
        """Extract enum/literal options if available."""
        origin = get_origin(annotation)
        if origin is Literal:
            return list(get_args(annotation))
        if isinstance(annotation, type) and issubclass(annotation, str) and hasattr(annotation, "__members__"):
            return list(annotation.__members__.keys())
        return []

    def _infer_widget_type(self, annotation: Any, value: Any, options: list[Any]) -> WidgetType:
        """Infer a Streamlit widget type for the field."""
        if options:
            return "selectbox"
        origin = get_origin(annotation)
        if origin in {list, tuple}:
            return "list_editor"
        if isinstance(value, bool):
            return "checkbox"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return "number_input"
        return "text_input"

    def _extract_bounds(self, metadata: list[Any]) -> tuple[float | int | None, float | int | None]:
        """Extract numeric constraints from Pydantic field metadata."""
        minimum: float | int | None = None
        maximum: float | int | None = None
        for item in metadata:
            for attr in ("ge", "gt"):
                bound = getattr(item, attr, None)
                if bound is not None:
                    minimum = bound
            for attr in ("le", "lt"):
                bound = getattr(item, attr, None)
                if bound is not None:
                    maximum = bound
        return minimum, maximum

    def _infer_step(self, annotation: Any, value: Any) -> float | int | None:
        """Infer a reasonable widget step size for numeric fields."""
        target_type = self._python_type_name(annotation, value)
        if target_type == "int":
            return 1
        if target_type == "float":
            return 0.01
        return None

    def _python_type_name(self, annotation: Any, value: Any) -> str:
        """Return a coarse-grained type label used by components."""
        origin = get_origin(annotation)
        if origin in {list, tuple}:
            args = get_args(annotation)
            if args:
                return self._python_type_name(args[0], value[0] if isinstance(value, list) and value else "")
            return "str"
        if annotation in {int, float, bool, str}:
            return annotation.__name__
        if value is not None:
            if isinstance(value, bool):
                return "bool"
            if isinstance(value, int):
                return "int"
            if isinstance(value, float):
                return "float"
            if isinstance(value, list):
                if value and isinstance(value[0], int):
                    return "int"
                if value and isinstance(value[0], float):
                    return "float"
                return "str"
        return "str"
