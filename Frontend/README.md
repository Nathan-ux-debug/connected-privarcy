# PrivARcy Frontend (rebuilt)

PyQt5 desktop app. Entry point: `Frontend/main.py`. Pages live in `ui/pages/`; every page talks only to `Backend.service.BackendService`.
The previous frontend is preserved untouched in `Frontend/legacy/` (not imported).
Design tokens: `ui/theme.py`. Icons are drawn in code (`ui/icons.py`). Light workspace + compact dark sidebar; single theme.
