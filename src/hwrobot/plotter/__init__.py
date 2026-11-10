"""Plotter backends behind one interface.

- :class:`~hwrobot.plotter.base.Plotter` — the interface
- :class:`~hwrobot.plotter.simulator.SimulatorPlotter` — renders to PNG/SVG (default)
- ``grbl.py`` — serial backend for real hardware (later)
"""

from hwrobot.plotter.base import OutOfBoundsError, Plotter
from hwrobot.plotter.simulator import SimulatorPlotter

__all__ = ["OutOfBoundsError", "Plotter", "SimulatorPlotter"]
