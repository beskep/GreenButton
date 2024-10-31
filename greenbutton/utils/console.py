from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeVar

import rich
from deprecated import deprecated
from loguru import logger
from rich import progress
from rich.logging import RichHandler
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from logging import LogRecord

T = TypeVar('T')

console = rich.get_console()
console.push_theme(Theme({'logging.level.success': 'bold blue'}))


class LogHandler(RichHandler):
    _NEW_LEVELS: ClassVar[dict[int, str]] = {5: 'TRACE', 25: 'SUCCESS'}

    def emit(self, record: LogRecord) -> None:
        if name := self._NEW_LEVELS.get(record.levelno):
            record.levelname = name

        return super().emit(record)

    @classmethod
    def set(
        cls,
        level: int | str = 20,
        *,
        rich_tracebacks: bool = False,
        remove: bool = True,
        **kwargs,
    ):
        handler = cls(
            console=console,
            markup=True,
            log_time_format='[%X]',
            rich_tracebacks=rich_tracebacks,
        )

        if remove:
            logger.remove()

        logger.add(handler, level=level, format='{message}', **kwargs)


@deprecated
def set_logger(level: int | str = 20, *, rich_tracebacks=False, **kwargs):
    handler = LogHandler(
        console=console,
        markup=True,
        log_time_format='[%X]',
        rich_tracebacks=rich_tracebacks,
    )

    logger.remove()
    logger.add(handler, level=level, format='{message}', **kwargs)


class ProgressColumn(progress.TaskProgressColumn):
    def __init__(
        self,
        *,
        style='progress.download',
        highlighter=None,
        table_column=None,
        show_speed=False,
    ):
        super().__init__(
            text_format='',
            text_format_no_percentage='',
            style=style,
            justify='left',
            markup=True,
            highlighter=highlighter,
            table_column=table_column,
            show_speed=show_speed,
        )

    @staticmethod
    def text(task: progress.Task):
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else '?'
        width = len(str(total))
        return f'{completed:{width}d}/{total}={task.percentage:>3.0f}%'

    def render(self, task: progress.Task):
        if task.total is None and self.show_speed:
            return self.render_speed(task.finished_speed or task.speed)

        s = self.text(task=task)
        text = Text(s, style=self.style, justify=self.justify)

        if self.highlighter:
            self.highlighter.highlight(text)

        return text


class Progress(progress.Progress):
    @classmethod
    def get_default_columns(cls) -> tuple[progress.ProgressColumn, ...]:
        return (
            progress.TextColumn('[progress.description]{task.description}'),
            progress.BarColumn(bar_width=60),
            ProgressColumn(show_speed=True),
            progress.TimeRemainingColumn(compact=True, elapsed_when_finished=True),
        )

    @classmethod
    def trace(
        cls,
        sequence: Sequence[T] | Iterable[T],
        *,
        description: str = 'Working...',
        total: float | None = None,
        completed: int = 0,
        transient: bool = False,
    ) -> Iterable[T]:
        with cls(transient=transient) as p:
            yield from p.track(
                sequence,
                total=total,
                completed=completed,
                description=description,
            )


if __name__ == '__main__':
    import time

    for _ in Progress.trace(list(range(10))):
        time.sleep(0.1)

    LogHandler.set(1, rich_tracebacks=False)

    logger.trace('Trace')
    logger.debug('Debug')
    logger.info('Info')
    logger.success('Success')
    logger.warning('Warning')
    logger.error('Error')
    logger.critical('Critical')

    try:
        x = 1 / 0
    except ZeroDivisionError as e:
        logger.exception(e)
