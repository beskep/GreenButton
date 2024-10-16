from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import rich
from loguru import logger
from rich import progress
from rich.logging import RichHandler
from rich.theme import Theme

if TYPE_CHECKING:
    from logging import LogRecord


console = rich.get_console()
console.push_theme(Theme({'logging.level.success': 'bold blue'}))


class _RichHandler(RichHandler):
    BLANK_NO = 21
    _NEW_LVLS: ClassVar[dict[int, str]] = {5: 'TRACE', 25: 'SUCCESS', BLANK_NO: ''}

    def emit(self, record: LogRecord) -> None:
        if name := self._NEW_LVLS.get(record.levelno, None):
            record.levelname = name

        return super().emit(record)


def set_logger(level: int | str = 20, *, rich_tracebacks=False, **kwargs):
    handler = _RichHandler(
        console=console,
        markup=True,
        log_time_format='[%X]',
        rich_tracebacks=rich_tracebacks,
    )

    logger.remove()
    logger.add(handler, level=level, format='{message}', **kwargs)


class Progress(progress.Progress):
    @classmethod
    def get_default_columns(cls) -> tuple[progress.ProgressColumn, ...]:
        return (
            progress.TextColumn('[progress.description]{task.description}'),
            progress.BarColumn(bar_width=60),
            progress.MofNCompleteColumn(),
            progress.TaskProgressColumn(),
            progress.TimeRemainingColumn(compact=True, elapsed_when_finished=True),
        )


if __name__ == '__main__':
    set_logger(1, rich_tracebacks=True)

    logger.trace('trace')
    logger.debug('debug')
    logger.info('info')
    logger.success('success')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')

    try:
        x = 1 / 0
    except ZeroDivisionError:
        logger.exception('exception')
