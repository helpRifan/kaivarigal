"""GRBL serial backend (Phase 5).

Will stream G-code from :mod:`hwrobot.plotter.gcode` to an Arduino running
GRBL (servo-pen fork) over USB serial with pyserial, using GRBL's
character-counting flow control. Must import cleanly with no hardware attached.
"""
