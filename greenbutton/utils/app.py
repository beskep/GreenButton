from __future__ import annotations

import enum
import itertools
from typing import TYPE_CHECKING, Any

import attrs
import cyclopts

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def _is_helper(name: str | Iterable[str] | None) -> bool:
    helpers = {'-h', '--help', '--version'}

    if name is None:
        return False

    if isinstance(name, str):
        return name in helpers

    return any(n in helpers for n in name)


class RegisteredOrder(enum.Enum):
    token = 0


class RemovePrefix(enum.Enum):
    token = 0


REGISTERED_ORDER = RegisteredOrder.token
REMOVE_PREFIX = RemovePrefix.token


@attrs.define
class App(cyclopts.App):
    _count: itertools.count = attrs.field(factory=itertools.count)

    def _remove_prefix(self, s: str):
        if name := self.name:
            s = s.removeprefix(name[0])

        return cyclopts.default_name_transform(s)

    def command(  # type: ignore[override]
        self,
        obj: Callable | None = None,
        name: str | Iterable[str] | None = None,
        sort_key: Any = REGISTERED_ORDER,
        name_transform: Callable[[str], str] | RemovePrefix | None = REMOVE_PREFIX,
        **kwargs: object,
    ):
        if _is_helper(name):
            sort_key = None
        elif sort_key is REGISTERED_ORDER:
            sort_key = next(self._count)

        if isinstance(obj, cyclopts.App):
            obj._sort_key = sort_key  # noqa: SLF001
        else:
            if name_transform is REMOVE_PREFIX:
                name_transform = self._remove_prefix

            kwargs['sort_key'] = sort_key
            kwargs['name_transform'] = name_transform

        return super().command(obj=obj, name=name, **kwargs)
