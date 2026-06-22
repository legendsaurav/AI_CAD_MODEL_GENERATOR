"""
shared_schemas/versioning.py — Schema Versioning Infrastructure
================================================================
Provides a versioned base class for all schemas in the AI CAD OS.

Every schema that crosses a repository boundary MUST inherit from
VersionedSchema to guarantee forward/backward compatibility checks
and automated migration support.
"""
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound="VersionedSchema")


class SchemaVersion(BaseModel):
    """Semantic version triple attached to every schema instance."""

    major: int = Field(
        ..., ge=0, description="Major version — incremented on breaking changes."
    )
    minor: int = Field(
        ..., ge=0, description="Minor version — incremented on backward-compatible additions."
    )
    patch: int = Field(
        ..., ge=0, description="Patch version — incremented on backward-compatible fixes."
    )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def as_tuple(self) -> Tuple[int, int, int]:
        """Return the version as a comparable ``(major, minor, patch)`` tuple."""
        return (self.major, self.minor, self.patch)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "SchemaVersion":
        """Deserialize from a JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def parse(cls, version_str: str) -> "SchemaVersion":
        """Parse a ``'major.minor.patch'`` string into a :class:`SchemaVersion`.

        Raises:
            ValueError: If *version_str* does not contain exactly three
                dot-separated non-negative integers.
        """
        parts = version_str.strip().split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Expected 'major.minor.patch' format, got '{version_str}'"
            )
        try:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError as exc:
            raise ValueError(
                f"Version components must be integers, got '{version_str}'"
            ) from exc
        return cls(major=major, minor=minor, patch=patch)


# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------

# Keys are ``(schema_name, from_version_str, to_version_str)``.
_MIGRATION_REGISTRY: Dict[
    Tuple[str, str, str], Callable[[Dict[str, Any]], Dict[str, Any]]
] = {}


def register_migration(
    schema_name: str,
    from_version: str,
    to_version: str,
    fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> None:
    """Register a migration function that transforms a raw dict payload.

    Parameters:
        schema_name: Logical name of the schema (e.g. ``'reason_graph'``).
        from_version: Source version string (``'major.minor.patch'``).
        to_version: Target version string.
        fn: A callable that accepts the raw ``dict`` representation of the
            *from_version* schema and returns a ``dict`` compatible with
            *to_version*.
    """
    _MIGRATION_REGISTRY[(schema_name, from_version, to_version)] = fn


def get_migration(
    schema_name: str,
    from_version: str,
    to_version: str,
) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """Look up a previously registered migration, or return ``None``."""
    return _MIGRATION_REGISTRY.get((schema_name, from_version, to_version))


def list_migrations(
    schema_name: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """Return all registered migration keys, optionally filtered by *schema_name*."""
    if schema_name is None:
        return list(_MIGRATION_REGISTRY.keys())
    return [k for k in _MIGRATION_REGISTRY if k[0] == schema_name]


# ---------------------------------------------------------------------------
# Versioned base class
# ---------------------------------------------------------------------------


class VersionedSchema(BaseModel):
    """Abstract base class that adds semantic versioning to any Pydantic model.

    Subclasses should override :pyattr:`_SCHEMA_NAME` and
    :pyattr:`_CURRENT_VERSION` to declare their identity.

    Example::

        class MySchema(VersionedSchema):
            _SCHEMA_NAME: str = "my_schema"
            _CURRENT_VERSION: SchemaVersion = SchemaVersion(major=1, minor=0, patch=0)
            ...
    """

    schema_version: SchemaVersion = Field(
        default_factory=lambda: SchemaVersion(major=1, minor=0, patch=0),
        description="Semantic version of this schema instance.",
    )

    # Subclasses override these class-level sentinels.
    _SCHEMA_NAME: str = "base"
    _CURRENT_VERSION: SchemaVersion = SchemaVersion(major=1, minor=0, patch=0)

    # -- Compatibility helpers ------------------------------------------------

    def backward_compatible(self, other_version: SchemaVersion) -> bool:
        """Return ``True`` if *other_version* is backward-compatible with this instance.

        Compatibility rules (semver):
        * Same major version **and** other minor >= self minor → compatible.
        * Different major → incompatible.
        """
        return (
            self.schema_version.major == other_version.major
            and other_version.minor >= self.schema_version.minor
        )

    def is_current_version(self) -> bool:
        """Return ``True`` if this instance matches the class-level current version."""
        return self.schema_version.as_tuple() == self._CURRENT_VERSION.as_tuple()

    # -- Migration helpers ----------------------------------------------------

    def migrate_to(self, target_version: SchemaVersion) -> Dict[str, Any]:
        """Attempt to migrate the raw dict payload to *target_version*.

        Looks up a registered migration function and applies it.

        Returns:
            The migrated ``dict`` payload.

        Raises:
            ValueError: If no migration path is registered.
        """
        from_str = str(self.schema_version)
        to_str = str(target_version)
        fn = get_migration(self._SCHEMA_NAME, from_str, to_str)
        if fn is None:
            raise ValueError(
                f"No migration registered for {self._SCHEMA_NAME} "
                f"from {from_str} to {to_str}."
            )
        return fn(self.model_dump())

    @classmethod
    def migrate_and_load(
        cls: Type[T],
        raw: Dict[str, Any],
        target_version: Optional[SchemaVersion] = None,
    ) -> T:
        """Deserialize *raw*, migrating if the embedded version differs.

        Parameters:
            raw: A dict (e.g. from ``json.loads``).
            target_version: The version to migrate to.  Defaults to
                ``cls._CURRENT_VERSION``.

        Returns:
            A validated instance of the subclass.
        """
        target = target_version or cls._CURRENT_VERSION
        embedded = raw.get("schema_version", {})
        embedded_ver = SchemaVersion(**embedded) if embedded else cls._CURRENT_VERSION

        if embedded_ver.as_tuple() != target.as_tuple():
            fn = get_migration(
                cls._SCHEMA_NAME, str(embedded_ver), str(target)
            )
            if fn is not None:
                raw = fn(raw)
                raw["schema_version"] = target.model_dump()

        return cls.model_validate(raw)

    # -- Serialization --------------------------------------------------------

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize from a JSON string."""
        return cls.model_validate_json(json_str)
