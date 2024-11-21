# Copyright (c) 2023 Nordic Semiconductor ASA
# SPDX-License-Identifier: Apache-2.0

"""
Internal basic profiling functionality.
"""

from __future__ import annotations

import os
from time import perf_counter_ns
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    NoReturn,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import Concatenate, ParamSpec, Self

PROFILING_ENABLED = bool(os.getenv("PROFILING", default=False))

T = TypeVar("T")
TimingReport = Dict[str, Union[None, float, List[float]]]

if TYPE_CHECKING or PROFILING_ENABLED:

    def create_object_report(obj: object) -> TimingReport:
        """Get a dictionary containing execution times for each timed method in the given object."""
        report = {}
        times: Union[None, float, List[float]]

        for method, timer_instance in obj.__dict__.get("_timedmethods", {}).items():
            if timer_instance.max_times <= 1:
                if timer_instance.times:
                    times = _ns_to_ms(timer_instance.times[0])
                else:
                    times = None
            else:
                times = [_ns_to_ms(t) for t in timer_instance.times]

            report[method.__name__] = times

        return report

    def timed_method(
        max_times: int = 10,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """
        Decorator that records the execution times of a method.
        The execution times are stored in a circular buffer storing up to max times entries.

        :param max_times: Number of most recent times to keep.
        :return: TimedMethod object that wraps the method and records the execution
        time each time it is called.
        """

        def inner(method: Callable[P, T]) -> Callable[P, T]:
            return TimedMethod(method, max_times)  # type: ignore

        return inner

    def _ns_to_ms(time_ns: int) -> float:
        """Convert nanoseconds to miliseconds."""
        return time_ns / 1_000_000

    P = ParamSpec("P")
    O = TypeVar("O")

    class TimedMethod(Generic[O, P, T]):
        """
        Object wrapper around a method that records the most recent execution times
        of that method.
        """

        def __init__(
            self, method: Callable[Concatenate[object, P], T], max_times: int
        ) -> None:
            self._method: Callable[Concatenate[object, P], T] = method
            self._max_times: int = max_times

        def __call__(self, obj: O) -> T:
            """
            Call method that enables e.g. functools.cached_property to call the method
            through this object directly.
            """
            return self._get_instance(obj)()

        @overload
        def __get__(self, obj: Literal[None], owner: Optional[Type] = None) -> Self:
            ...

        @overload
        def __get__(
            self, obj: O, owner: Optional[Type] = None
        ) -> TimedMethodInstance[O, P, T]:
            ...

        def __get__(
            self, obj: Optional[O], owner: Any = None
        ) -> Union[TimedMethodInstance[O, P, T], Self]:
            """Get the child instance wrapper."""
            if obj is None:
                return self
            return self._get_instance(obj)

        def _get_instance(self, obj: O) -> TimedMethodInstance[O, P, T]:
            """Get the child wrapper instance, constructing it if accessed the first time."""
            obj_timed_methods = obj.__dict__.setdefault("_timedmethods", {})

            try:
                return obj_timed_methods[self._method]
            except KeyError:
                # First call; initialize the bound method
                instance = TimedMethodInstance(obj, self._method, self._max_times)
                obj_timed_methods[self._method] = instance
                return instance

    class TimedMethodInstance(Generic[O, P, T]):
        """
        Object wrapper around a method that records the most recent execution times
        of that method.
        """

        def __init__(
            self,
            obj: O,
            method: Callable[Concatenate[O, P], T],
            max_times: int,
        ) -> None:
            self._obj: O = obj
            self._method: Callable[Concatenate[O, P], T] = method
            self._max_times: int = max_times
            self._times: List[int] = []
            # Zero is the initial circular buffer head when the time buffer reaches its max capacity
            self._head: int = 0

        @property
        def times(self) -> List[int]:
            """
            A list of the most recent execution times, ordered from oldest to newest,
            with a size bounded by max_times.
            """
            if len(self._times) >= self._max_times:
                return self._times[self._head :] + self._times[: self._head]
            else:
                return self._times

        @property
        def max_times(self) -> int:
            """Maximum number of most recent execution times recorded."""
            return self._max_times

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
            """Call the wrapped method and record the execution time."""
            t_start = perf_counter_ns()

            result = self._method(self._obj, *args, **kwargs)

            t_end = perf_counter_ns()
            self._add_time(t_end - t_start)

            return result

        def _add_time(self, time_ns: int) -> None:
            """
            Add a new time to the time buffer.
            The time buffer acts as a regular list before it reaches its capacity,
            and is used a circular buffer after that.
            """
            if len(self._times) >= self._max_times:
                self._times[self._head] = time_ns
                self._head = (self._head + 1) % self._max_times
            else:
                self._times.append(time_ns)

else:

    def create_object_report(obj: Any) -> NoReturn:
        raise RuntimeError(
            "Profiling info is not available as profiling is not enabled."
            "Enable profiling by setting the environment variable 'PROFILING'=1."
        )

    def timed_method(*args: Any, **kwargs: Any) -> Callable[[T], T]:
        # Just return the method back again
        return lambda x: x
