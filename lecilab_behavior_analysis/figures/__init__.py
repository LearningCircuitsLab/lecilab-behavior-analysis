"""
Figures made of panels: render.py (how a figure is drawn), panels.py (each
plot) and layouts.py (which panels make each figure).
"""
from lecilab_behavior_analysis.figures.layouts import (
    SESSION_SUMMARY,
    SUBJECT_PROGRESS,
    session_summary,
    subject_progress,
)
from lecilab_behavior_analysis.figures.render import (
    FigureData,
    Layout,
    NotApplicable,
    Panel,
    PanelProblem,
    Row,
    panel,
    render,
    split_cell,
)
