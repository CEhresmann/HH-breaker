from typing import Type

from hr_breaker.filters.base import BaseFilter


class FilterRegistry:
    """Registry for filter plugins."""

    _filters: dict[str, Type[BaseFilter]] = {}

    @classmethod
    def register(cls, filter_class: Type[BaseFilter]) -> Type[BaseFilter]:
        """Decorator to register a filter class."""
        cls._filters[filter_class.name] = filter_class
        return filter_class

    @classmethod
    def get(cls, name: str) -> Type[BaseFilter] | None:
        return cls._filters.get(name)

    @classmethod
    def all(cls) -> list[Type[BaseFilter]]:
        return list(cls._filters.values())

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._filters.keys())
