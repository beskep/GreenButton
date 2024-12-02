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


def _is_help_or_version(name: str | Iterable[str] | None) -> bool:
    hv = {'-h', '--help', '--version'}

    if name is None:
        return False

    if isinstance(name, str):
        return name in hv

    return any(n in hv for n in name)


REGISTERED_ORDER = _Sentinel('REGISTERED_ORDER')
_count = itertools.count()


@attrs.define
class App(cyclopts.App):
    _sort_key: Any = attrs.field(
        default=REGISTERED_ORDER,
        alias='sort_key',
        kw_only=True,
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        if self._sort_key is REGISTERED_ORDER:
            self._sort_key = next(_count)

    def command(
        self,
        obj: T | None = None,
        name: str | Iterable[str] | None = None,
        sort_key: Any = REGISTERED_ORDER,
        **kwargs: object,
    ):
        if sort_key is REGISTERED_ORDER:
            sort_key = None if _is_help_or_version(name) else next(_count)

        return super().command(obj=obj, name=name, sort_key=sort_key, **kwargs)
