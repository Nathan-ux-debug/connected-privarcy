"""Application shell: sidebar + page stack + loading / error states."""

from __future__ import annotations

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget,
)

from . import theme
from .icons import make_icon
from .widgets import button, hline, label
from .workers import BackendLoader

NAV = [
    ("Dashboard", "dashboard"), ("Video Processing", "video"), ("Live Camera", "camera"), ("Review Queue", "review"),
    ("Face Registry", "face"), ("Correction Log", "log"), ("History & Outputs", "history"),
]
BOTTOM_NAV = [("Settings", "settings")]
SIDEBAR_WIDTH = 196


class MainWindow(QMainWindow):
    def __init__(self, logo_path: str = "") -> None:
        super().__init__()
        self.setWindowTitle("PrivARcy")
        self.resize(1360, 820)
        self.setMinimumSize(980, 640)
        self.logo_path = logo_path
        if logo_path:
            pm = QPixmap(logo_path)
            if not pm.isNull():
                self.setWindowIcon(QIcon(pm))
        self.settings = QSettings("PrivARcy", "Dashboard2")
        self.service = None
        self.pages: dict = {}
        self.nav_buttons: dict = {}
        self.current = "Dashboard"
        self._busy = 0
        self._loader = None

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        self.top_stack = QStackedWidget()
        outer.addWidget(self.top_stack)
        self.top_stack.addWidget(self._build_loading())   # 0
        self.top_stack.addWidget(self._build_error())     # 1
        self.top_stack.addWidget(self._build_app())       # 2
        self.start_loading()

    # ------------------------------------------------------------ load states
    def _build_loading(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(10)
        if self.logo_path:
            logo = QLabel()
            pm = QPixmap(self.logo_path)
            if not pm.isNull():
                logo.setPixmap(pm.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                logo.setAlignment(Qt.AlignCenter)
                lay.addWidget(logo)
        t = label("Starting PrivARcy", "PageTitle")
        t.setAlignment(Qt.AlignCenter)
        lay.addWidget(t)
        self.loading_text = label("Loading the detection model and preparing storage…", "Muted")
        self.loading_text.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.loading_text)
        bar = QProgressBar()
        bar.setRange(0, 0)
        bar.setFixedWidth(260)
        lay.addWidget(bar, alignment=Qt.AlignHCenter)
        return w

    def _build_error(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(12)
        t = label("PrivARcy could not start", "PageTitle")
        t.setAlignment(Qt.AlignCenter)
        lay.addWidget(t)
        self.error_text = label("", "Muted", wrap=True)
        self.error_text.setMaximumWidth(720)
        self.error_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.error_text, alignment=Qt.AlignHCenter)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(button("Try again", "Primary", callback=self.start_loading))
        row.addWidget(button("Quit", "", callback=self.close))
        row.addStretch()
        lay.addLayout(row)
        return w

    def start_loading(self) -> None:
        self.top_stack.setCurrentIndex(0)
        self._loader = BackendLoader()
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        self._loader.start()

    def _on_failed(self, message: str) -> None:
        self.error_text.setText(message)
        self.top_stack.setCurrentIndex(1)

    def _on_loaded(self, service) -> None:
        self.service = service
        self._restore_settings()
        from .pages.dashboard import DashboardPage
        from .pages.video import VideoPage
        from .pages.live import LivePage
        from .pages.review import ReviewPage
        from .pages.faces import FacesPage
        from .pages.corrections import CorrectionsPage
        from .pages.history import HistoryPage
        from .pages.settings import SettingsPage

        classes = {"Dashboard": DashboardPage, "Video Processing": VideoPage, "Live Camera": LivePage, "Review Queue": ReviewPage,
                   "Face Registry": FacesPage, "Correction Log": CorrectionsPage, "History & Outputs": HistoryPage, "Settings": SettingsPage}
        for name, cls in classes.items():
            page = cls(self)
            self.pages[name] = page
            self.page_stack.addWidget(page)
        self.top_stack.setCurrentIndex(2)
        self.navigate("Dashboard")
        self.refresh_badges()

    # ------------------------------------------------------------------ layout
    def _build_app(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(10, 12, 10, 12)
        sl.setSpacing(3)

        brand = QFrame()
        brand.setObjectName("SidebarBrand")
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(8, 7, 8, 7)
        bl.setSpacing(8)
        mark = QLabel()
        mark.setFixedSize(22, 22)
        mark.setAlignment(Qt.AlignCenter)
        pm = QPixmap(self.logo_path) if self.logo_path else QPixmap()
        if not pm.isNull():
            mark.setPixmap(pm.scaled(22, 22, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            mark.setPixmap(make_icon("shield", "#FFFFFF", 20).pixmap(20, 20))
        bl.addWidget(mark)
        name = QLabel("PrivARcy")
        name.setObjectName("BrandName")
        bl.addWidget(name, 1)
        sl.addWidget(brand)
        sl.addSpacing(8)

        for text, icon in NAV:
            sl.addWidget(self._nav_button(text, icon))
        sl.addStretch(1)
        d = QFrame()
        d.setObjectName("SidebarDivider")
        sl.addWidget(d)
        sl.addSpacing(4)
        for text, icon in BOTTOM_NAV:
            sl.addWidget(self._nav_button(text, icon))
        self.busy_label = QLabel("Local processing")
        self.busy_label.setObjectName("SidebarMeta")
        self.busy_label.setContentsMargins(10, 6, 0, 0)
        sl.addWidget(self.busy_label)

        self.page_stack = QStackedWidget()
        lay.addWidget(sidebar)
        lay.addWidget(self.page_stack, 1)
        self.page_stack.setStyleSheet("QStackedWidget { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #F3EEFF, stop:0.22 #FAFAFC, stop:1 #FAFAFC); }")
        return w

    def _nav_button(self, text: str, icon: str) -> QPushButton:
        b = QPushButton("  " + text.replace("&", "&&"))
        b.setObjectName("NavButton")
        b.setCheckable(True)
        b.setCursor(Qt.PointingHandCursor)
        b.setIcon(make_icon(icon, theme.SIDEBAR_TEXT, 17))
        b.setProperty("icon_name", icon)
        b.setProperty("page", text)
        b.clicked.connect(lambda _=False, t=text: self.navigate(t))
        self.nav_buttons[text] = b
        return b

    # -------------------------------------------------------------- navigation
    def navigate(self, name: str) -> None:
        if self.service is None or name not in self.pages:
            return
        old = self.pages.get(self.current)
        if old is not None and self.current != name:
            old.on_hide()
        self.current = name
        self.page_stack.setCurrentWidget(self.pages[name])
        for text, b in self.nav_buttons.items():
            active = text == name
            b.setChecked(active)
            b.setIcon(make_icon(b.property("icon_name"), "#FFFFFF" if active else theme.SIDEBAR_TEXT, 17))
        self.pages[name].on_show()

    def open_video(self, path: str) -> None:
        self.navigate("Video Processing")
        self.pages["Video Processing"].set_video(path)

    def set_busy(self, busy: bool) -> None:
        self._busy += 1 if busy else -1
        self._busy = max(0, self._busy)
        self.busy_label.setText("● Working…" if self._busy else "Local processing")
        self.busy_label.setStyleSheet("color: #8C74FF;" if self._busy else "")

    def refresh_badges(self) -> None:
        if self.service is None:
            return
        n = self.service.overview()["pending_review"]
        self.nav_buttons["Review Queue"].setText("  Review Queue" + (f"   {n}" if n else ""))

    def settings_changed(self) -> None:
        settings_page = self.pages.get("Settings")
        if settings_page is not None and self.current != "Settings":
            settings_page._load()

    # -------------------------------------------------------------- persistence
    _KEYS = ("redaction_method", "registered_face_policy", "screen_policy", "hold_missed_detections", "temporal_smoothing",
             "ocr_enabled", "ocr_model", "detection_interval", "live_ocr_enabled")

    def persist_settings(self) -> None:
        c = self.service.config
        for k in self._KEYS:
            self.settings.setValue(k, getattr(c, k))
        self.settings.setValue("enabled_classes", sorted(c.enabled_classes))

    def _restore_settings(self) -> None:
        def b(v): return v if isinstance(v, bool) else str(v).lower() == "true"
        s, values = self.settings, {}
        casts = {"hold_missed_detections": b, "ocr_enabled": b, "live_ocr_enabled": b, "temporal_smoothing": int, "detection_interval": int}
        for k in self._KEYS:
            if s.contains(k):
                try:
                    v = s.value(k)
                    values[k] = casts.get(k, str)(v)
                except (TypeError, ValueError):
                    pass
        if s.contains("enabled_classes"):
            v = s.value("enabled_classes")
            values["enabled_classes"] = set(v if isinstance(v, list) else [v])
        if values:
            self.service.update_settings(**values)

    # ------------------------------------------------------------------- close
    def closeEvent(self, event) -> None:
        if self._busy:
            ans = QMessageBox.question(self, "Quit PrivARcy", "Work is still running. Stop it and quit?")
            if ans != QMessageBox.Yes:
                event.ignore()
                return
        for page in self.pages.values():
            try:
                page.shutdown()
            except Exception:
                pass
        if self.service is not None:
            self.service.shutdown()
        super().closeEvent(event)
