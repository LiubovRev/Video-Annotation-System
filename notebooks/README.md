# Exploratory notebooks

These notebooks are the original exploration that the `vas` package grew out
of. They are kept for reference only and **predate the fixes in v0.2.0** — in
particular they name MediaPipe landmarks with the OpenPose BODY_25 layout and
compute temporal features grouped by the annotation label. Use the package
(`vas …`) for anything you want to report.

Outputs were stripped (they could contain frames from clinical recordings).
`pre-commit` runs `nbstripout` so outputs are not committed again.
