"""Anatomy Inspector: find what is wrong before a tab repaints it.

    report = inspect(image, "Realistic")

The rest of the package is internal. See README, "Anatomy Inspector".
"""

from .findings import Finding, Person, Report
from .inspect import inspect
from .profiles import NAMES as PROFILE_NAMES

__all__ = ["Finding", "Person", "Report", "inspect", "PROFILE_NAMES"]
