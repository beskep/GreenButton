from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import attrs
import cyclopts

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cyclopts.core import T


class _Sentinel:
    def __init__(self, v: str, /):
        self.v = v

    def __repr__(self):
        return f'<{self.v}>'


def _is_helper(name: str | Iterable[str] | None) -> bool:
    helpers = {'-h', '--help', '--version'}

    if name is None:
        return False

    if isinstance(name, str):
        return name in helpers

    return any(n in helpers for n in name)


REGISTERED_ORDER = _Sentinel('REGISTERED_ORDER')


@attrs.define
class App(cyclopts.App):
    _count: itertools.count = attrs.field(factory=itertools.count)

    def command(
        self,
        obj: T | None = None,
        name: str | Iterable[str] | None = None,
        sort_key: Any = REGISTERED_ORDER,
        **kwargs: object,
    ):
        if _is_helper(name):
            sort_key = None
        elif sort_key is REGISTERED_ORDER:
            sort_key = next(self._count)

        if isinstance(obj, cyclopts.App):
            obj._sort_key = sort_key  # noqa: SLF001
        else:
            kwargs['sort_key'] = sort_key

        return super().command(obj=obj, name=name, **kwargs)
