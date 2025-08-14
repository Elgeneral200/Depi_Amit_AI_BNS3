# UI.py
import sys, json, uuid
from datetime import datetime, date
from typing import Optional, List, Dict, Tuple

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QGroupBox, QFormLayout, QLineEdit,
    QSpinBox, QPlainTextEdit, QPushButton, QMessageBox, QTabWidget, QCheckBox,
    QInputDialog, QDialog, QDialogButtonBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QDateTimeEdit, QDateEdit
)
from PySide6.QtCore import Qt, QDateTime, QDate, QObject, Signal, QSettings
from PySide6.QtGui import QFont, QPalette, QColor, QBrush

# Charts (اختياري: لو غير متاحة، هنستخدم بديل نصي)
try:
    from PySide6.QtCharts import (
        QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis
    )
    HAS_QTCHARTS = True
except Exception:
    HAS_QTCHARTS = False

# Domain imports (ملفاتك)
from hospital import Hospital
from department import Department
from patient import Patient
from staff import Staff


# ================= I18N (Arabic/English) =================
class I18nManager(QObject):
    language_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.settings = QSettings("HospitalApp", "UI")
        self.lang = self.settings.value("language", "ar")
        self.strings: Dict[str, Dict[str, str]] = {
            # App / Menu
            "app.title": {"ar": "إدارة المستشفى - {name}", "en": "Hospital Management - {name}"},
            "menu.file": {"ar": "ملف", "en": "File"},
            "menu.file.new": {"ar": "جديد", "en": "New"},
            "menu.file.open": {"ar": "فتح...", "en": "Open..."},
            "menu.file.save": {"ar": "حفظ", "en": "Save"},
            "menu.file.saveas": {"ar": "حفظ باسم...", "en": "Save As..."},
            "menu.language": {"ar": "اللغة", "en": "Language"},
            "menu.language.ar": {"ar": "العربية", "en": "Arabic"},
            "menu.language.en": {"ar": "الإنجليزية", "en": "English"},
            "menu.theme": {"ar": "المظهر", "en": "Theme"},
            "menu.theme.light": {"ar": "فاتح", "en": "Light"},
            "menu.theme.dark": {"ar": "داكن", "en": "Dark"},
            "filter.all": {"ar": "الكل", "en": "All"},

            # Tabs
            "tab.dashboard": {"ar": "لوحة المعلومات", "en": "Dashboard"},
            "tab.patients": {"ar": "المرضى", "en": "Patients"},
            "tab.staff": {"ar": "الطاقم", "en": "Staff"},
            "tab.search": {"ar": "بحث", "en": "Search"},
            "tab.appointments": {"ar": "المواعيد", "en": "Appointments"},

            # Left panel (Departments)
            "label.departments": {"ar": "الأقسام", "en": "Departments"},
            "group.add_dept": {"ar": "إضافة قسم", "en": "Add Department"},
            "field.name": {"ar": "الاسم", "en": "Name"},
            "field.capacity": {"ar": "السعة", "en": "Capacity"},
            "btn.add": {"ar": "إضافة", "en": "Add"},

            # Info panel
            "group.dept_info": {"ar": "معلومات القسم", "en": "Department Info"},
            "label.name:": {"ar": "الاسم:", "en": "Name:"},
            "label.code:": {"ar": "الكود:", "en": "Code:"},
            "label.capacity:": {"ar": "السعة:", "en": "Capacity:"},
            "label.patients_count:": {"ar": "عدد المرضى:", "en": "Patients:"},
            "label.staff_count:": {"ar": "عدد الطاقم:", "en": "Staff:"},
            "unit.years": {"ar": "سنة", "en": "yr"},

            # Patients tab
            "group.add_patient": {"ar": "إضافة مريض", "en": "Add Patient"},
            "field.age": {"ar": "السن", "en": "Age"},
            "field.medical_record": {"ar": "السجل الطبي", "en": "Medical Record"},
            "ph.medical_record": {"ar": "الملاحظات/السجل الطبي", "en": "Notes / medical record"},
            "btn.add_patient": {"ar": "إضافة مريض", "en": "Add Patient"},
            "group.patients_list": {"ar": "قائمة المرضى", "en": "Patients List"},
            "chk.only_active": {"ar": "عرض غير المخروجين فقط", "en": "Show only non-discharged"},
            "btn.discharge_selected": {"ar": "خروج المريض المحدد", "en": "Discharge selected patient"},
            "btn.move": {"ar": "نقل المريض لقسم آخر", "en": "Move patient to another department"},
            "btn.refresh": {"ar": "تحديث", "en": "Refresh"},
            "status.admitted": {"ar": "مقيم", "en": "Admitted"},
            "status.discharged": {"ar": "مخروج", "en": "Discharged"},

            # Staff tab
            "group.add_staff": {"ar": "إضافة موظف", "en": "Add Staff"},
            "field.position": {"ar": "المنصب", "en": "Position"},
            "ph.position_hint": {"ar": "مثال: Cardiologist / Nurse / Technician", "en": "Example: Cardiologist / Nurse / Technician"},
            "btn.add_staff": {"ar": "إضافة موظف", "en": "Add Staff"},
            "group.staff_list": {"ar": "قائمة الطاقم", "en": "Staff List"},
            "ph.position_filter": {"ar": "فلتر بالمنصب (اختياري)", "en": "Filter by position (optional)"},
            "btn.toggle_active": {"ar": "تبديل حالة التفعيل", "en": "Toggle active state"},
            "staff.active": {"ar": "نشط", "en": "Active"},
            "staff.inactive": {"ar": "موقّف", "en": "Inactive"},

            # Search tab
            "group.search": {"ar": "بحث عن مريض في كل الأقسام", "en": "Search patient across all departments"},
            "search.field.label": {"ar": "كلمة البحث", "en": "Search term"},
            "ph.search_all": {"ar": "اكتب اسم المريض أو Patient ID", "en": "Type patient name or Patient ID"},

            # Appointments tab
            "group.create_appt": {"ar": "إنشاء موعد", "en": "Create appointment"},
            "field.dept": {"ar": "القسم", "en": "Department"},
            "field.patient": {"ar": "المريض", "en": "Patient"},
            "field.staff": {"ar": "الموظف", "en": "Staff"},
            "field.start": {"ar": "بداية", "en": "Start"},
            "field.end": {"ar": "نهاية", "en": "End"},
            "field.notes": {"ar": "ملاحظات", "en": "Notes"},
            "btn.add_appt": {"ar": "إضافة موعد", "en": "Add appointment"},
            "group.appt_list": {"ar": "قائمة المواعيد", "en": "Appointments List"},
            "filter.date": {"ar": "التاريخ:", "en": "Date:"},
            "filter.dept": {"ar": "القسم:", "en": "Department:"},
            "filter.status": {"ar": "الحالة:", "en": "Status:"},

            "col.id": {"ar": "#", "en": "#"},
            "col.dept": {"ar": "القسم", "en": "Department"},
            "col.patient": {"ar": "المريض", "en": "Patient"},
            "col.staff": {"ar": "الموظف", "en": "Staff"},
            "col.start": {"ar": "بداية", "en": "Start"},
            "col.end": {"ar": "نهاية", "en": "End"},
            "col.status": {"ar": "الحالة", "en": "Status"},
            "col.notes": {"ar": "ملاحظات", "en": "Notes"},
            "col.conflict": {"ar": "تعارض", "en": "Conflict"},

            "btn.change_status": {"ar": "تغيير حالة الموعد", "en": "Change appointment status"},
            "btn.delete_appt": {"ar": "حذف الموعد المحدد", "en": "Delete selected appointment"},

            "appt.status.scheduled": {"ar": "مجدول", "en": "Scheduled"},
            "appt.status.checked_in": {"ar": "حضر", "en": "Checked-in"},
            "appt.status.completed": {"ar": "مكتمل", "en": "Completed"},
            "appt.status.cancelled": {"ar": "ملغي", "en": "Cancelled"},

            # Dashboard
            "dash.stats": {"ar": "إحصائيات", "en": "Statistics"},
            "dash.total_depts": {"ar": "عدد الأقسام", "en": "Departments"},
            "dash.total_patients": {"ar": "إجمالي المرضى", "en": "Total patients"},
            "dash.active_patients": {"ar": "مرضى مقيمون", "en": "Active (admitted)"},
            "dash.total_staff": {"ar": "عدد الطاقم", "en": "Staff"},
            "dash.appts_today": {"ar": "مواعيد اليوم", "en": "Appointments today"},
            "dash.refresh": {"ar": "تحديث", "en": "Refresh"},
            "dash.chart.appts_by_status": {"ar": "مواعيد اليوم حسب الحالة", "en": "Today's appointments by status"},
            "dash.no_qtcharts": {"ar": "QtCharts غير متاحة - لا يمكن عرض الرسم", "en": "QtCharts not available - cannot display chart"},

            # Messages (titles)
            "msg.info.title": {"ar": "تم", "en": "Info"},
            "msg.warning.title": {"ar": "تنبيه", "en": "Warning"},
            "msg.error.title": {"ar": "خطأ", "en": "Error"},
            "msg.confirm.title": {"ar": "تأكيد", "en": "Confirm"},

            # Messages (content)
            "msg.new.confirm": {"ar": "بدء ملف جديد؟ قد تفقد تغييرات غير محفوظة.", "en": "Start new file? Unsaved changes may be lost."},
            "dialog.open.title": {"ar": "فتح ملف بيانات", "en": "Open data file"},
            "dialog.save.title": {"ar": "حفظ الملف", "en": "Save file"},
            "dialog.json.filter": {"ar": "JSON (*.json)", "en": "JSON (*.json)"},
            "msg.loaded_ok": {"ar": "تم تحميل البيانات بنجاح", "en": "Data loaded successfully"},
            "msg.open.fail": {"ar": "فشل التحميل:\n{err}", "en": "Failed to load:\n{err}"},
            "msg.save.ok": {"ar": "تم الحفظ", "en": "Saved"},
            "msg.save.fail": {"ar": "فشل الحفظ:\n{err}", "en": "Failed to save:\n{err}"},

            "msg.select_department_first": {"ar": "اختار قسم أولًا", "en": "Select a department first"},
            "msg.patient.name_required": {"ar": "اسم المريض مطلوب", "en": "Patient name is required"},
            "msg.dept.full": {"ar": "لا يمكن إدخال {name}، القسم ممتلئ", "en": "Cannot admit {name}, department is full"},
            "msg.patient.added": {"ar": "تم إدخال المريض: {pid}", "en": "Patient admitted: {pid}"},
            "msg.select_patient": {"ar": "اختار مريض من القائمة", "en": "Select a patient from the list"},
            "msg.patient.already_discharged": {"ar": "المريض مخروج بالفعل", "en": "Patient already discharged"},
            "dialog.discharge.notes_title": {"ar": "ملاحظات الخروج", "en": "Discharge notes"},
            "dialog.discharge.notes_prompt": {"ar": "أدخل ملاحظات الخروج (اختياري):", "en": "Enter discharge notes (optional):"},
            "msg.patient.discharged": {"ar": "تم خروج المريض {name}", "en": "Patient {name} discharged"},

            "msg.no_other_departments": {"ar": "لا يوجد أقسام أخرى", "en": "No other departments available"},
            "dialog.move_patient.title": {"ar": "نقل مريض", "en": "Move patient"},
            "dialog.move_patient.prompt": {"ar": "اختر القسم الهدف:", "en": "Select target department:"},
            "msg.dept.full_with_name": {"ar": "القسم {name} ممتلئ", "en": "Department {name} is full"},
            "msg.patient.moved_to": {"ar": "تم نقل {pname} إلى {dname}", "en": "{pname} moved to {dname}"},

            "msg.staff.name_position_required": {"ar": "الاسم والمنصب مطلوبان", "en": "Name and position are required"},
            "msg.staff.added": {"ar": "تمت إضافة الموظف: {sid}", "en": "Staff added: {sid}"},

            "msg.select_dept": {"ar": "اختر قسم", "en": "Select department"},
            "msg.select_patient_for_appt": {"ar": "اختر مريض", "en": "Select a patient"},
            "msg.end_after_start": {"ar": "وقت النهاية يجب أن يكون بعد البداية", "en": "End time must be after start"},
            "msg.appt.created": {"ar": "تم إنشاء الموعد #{id}", "en": "Appointment #{id} created"},
            "msg.select_appt": {"ar": "اختر موعدًا من الجدول", "en": "Select an appointment from the table"},
            "dialog.change_status.title": {"ar": "تغيير الحالة", "en": "Change status"},
            "dialog.change_status.prompt": {"ar": "اختر الحالة:", "en": "Pick a status:"},
            "dialog.delete_appt.confirm": {"ar": "حذف الموعد #{id}؟", "en": "Delete appointment #{id}?"},
            "msg.appt.conflict.title": {"ar": "تعارض مواعيد", "en": "Appointment conflict"},
            "msg.appt.conflict.body": {"ar": "يوجد تعارض مع المواعيد:\n{details}", "en": "Conflicts with existing appointments:\n{details}"},
            "msg.appt.conflict.row": {"ar": "- #{id} | {dept} | {start}-{end} | {who}", "en": "- #{id} | {dept} | {start}-{end} | {who}"},
            "who.patient": {"ar": "نفس المريض", "en": "same patient"},
            "who.staff": {"ar": "نفس الموظف", "en": "same staff"},

            # Patient details dialog
            "dialog.patient_details.title": {"ar": "تفاصيل المريض - {id}", "en": "Patient Details - {id}"},
            "label.patient_id:": {"ar": "رقم المريض:", "en": "Patient ID:"},
            "label.state:": {"ar": "الحالة:", "en": "State:"},
            "group.dates": {"ar": "تواريخ", "en": "Dates"},
            "label.admission_date:": {"ar": "تاريخ الدخول:", "en": "Admission date:"},
            "label.discharge_date:": {"ar": "تاريخ الخروج:", "en": "Discharge date:"},
            "btn.save": {"ar": "حفظ التعديلات", "en": "Save changes"},
            "btn.close": {"ar": "إغلاق", "en": "Close"},
            "btn.discharge": {"ar": "خروج المريض", "en": "Discharge"},
        }

    def t(self, key: str, **kwargs) -> str:
        data = self.strings.get(key, {})
        text = data.get(self.lang) or data.get("en") or key
        return text.format(**kwargs)

    def set_language(self, lang: str):
        if lang not in ("ar", "en"): return
        if lang == self.lang: return
        self.lang = lang
        self.settings.setValue("language", lang)
        self.language_changed.emit(lang)


I18N = I18nManager()


# ================= Theme Manager (Dark/Light) =================
DARK_QSS = """
QWidget { background-color: #121212; color: #eaeaea; }
QMenuBar { background-color: #121212; color: #eaeaea; }
QMenuBar::item { padding: 6px 10px; }
QMenuBar::item:selected { background: #2a2a2a; color: #eaeaea; }
QMenu { background-color: #1b1b1b; color: #eaeaea; border: 1px solid #2d2d2d; }
QMenu::item { padding: 6px 18px; }
QMenu::item:selected { background-color: #2c2c2c; }
QStatusBar { background: #121212; color: #bbbbbb; }
QToolTip { color: #000; background: #ffffdd; border: 1px solid #bfbf7f; }

QGroupBox { border: 1px solid #2a2a2a; border-radius: 6px; margin-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top right; padding: 3px 8px; }

QLineEdit, QPlainTextEdit, QSpinBox, QDateTimeEdit, QDateEdit, QComboBox {
    background-color: #1e1e1e; border: 1px solid #3a3a3a; padding: 6px 8px; border-radius: 4px; color: #eaeaea;
}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDateTimeEdit:focus, QDateEdit:focus, QComboBox:focus {
    border: 1px solid #4a90e2;
}

QPushButton { background-color: #2b2b2b; border: 1px solid #3a3a3a; padding: 7px 12px; border-radius: 4px; }
QPushButton:hover { background-color: #343434; }
QPushButton:pressed { background-color: #3c3c3c; }

QListWidget, QTableWidget { background-color: #1a1a1a; border: 1px solid #333; alternate-background-color: #161616; }
QHeaderView::section { background: #1f1f1f; color: #ddd; border: 1px solid #3a3a3a; padding: 6px; }
QTabWidget::pane { border-top: 1px solid #2a2a2a; }
QTabBar::tab { background: #1e1e1e; padding: 8px 14px; border: 1px solid #3a3a3a; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-left: 2px; }
QTabBar::tab:selected { background: #2a2a2a; }

QScrollBar:vertical { background: #1c1c1c; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #3a3a3a; min-height: 24px; border-radius: 6px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #1c1c1c; height: 12px; margin: 0; }
QScrollBar::handle:horizontal { background: #3a3a3a; min-width: 24px; border-radius: 6px; }

QCheckBox { spacing: 6px; }

QComboBox QAbstractItemView {
    background: #1e1e1e; color: #eaeaea;
    selection-background-color: #2b2b2b; selection-color: #eaeaea;
}
"""

LIGHT_QSS = """
QWidget { background-color: #ffffff; color: #111; }
QMenuBar { background-color: #ffffff; color: #111; }
QMenuBar::item { padding: 6px 10px; }
QMenuBar::item:selected { background: #e6f0ff; color: #111; }
QMenu { background-color: #ffffff; color: #111; border: 1px solid #cfcfcf; }
QMenu::item { padding: 6px 18px; }
QMenu::item:selected { background-color: #e6f0ff; color: #111; }
QStatusBar { background: #ffffff; color: #666; }
QToolTip { color: #111; background: #ffffdd; border: 1px solid #bfbf7f; }

QGroupBox { border: 1px solid #e0e0e0; border-radius: 6px; margin-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top right; padding: 3px 8px; }

QLineEdit, QPlainTextEdit, QSpinBox, QDateTimeEdit, QDateEdit, QComboBox {
    background-color: #ffffff; border: 1px solid #cfcfcf; padding: 6px 8px; border-radius: 4px; color: #111;
}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDateTimeEdit:focus, QDateEdit:focus, QComboBox:focus {
    border: 1px solid #4a90e2;
}

QPushButton { background-color: #f6f6f6; border: 1px solid #cfcfcf; padding: 7px 12px; border-radius: 4px; color: #111; }
QPushButton:hover { background-color: #efefef; }
QPushButton:pressed { background-color: #e8e8e8; }

QListWidget, QTableWidget { background-color: #ffffff; border: 1px solid #dddddd; alternate-background-color: #fafafa; }
QHeaderView::section { background: #f4f4f4; color: #222; border: 1px solid #e0e0e0; padding: 6px; }
QTabWidget::pane { border-top: 1px solid #e0e0e0; }
QTabBar::tab { background: #f7f7f7; padding: 8px 14px; border: 1px solid #e0e0e0; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-left: 2px; }
QTabBar::tab:selected { background: #ffffff; }

QScrollBar:vertical { background: #f0f0f0; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #cfcfcf; min-height: 24px; border-radius: 6px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #f0f0f0; height: 12px; margin: 0; }
QScrollBar::handle:horizontal { background: #cfcfcf; min-width: 24px; border-radius: 6px; }

QComboBox QAbstractItemView {
    background: #ffffff; color: #111;
    selection-background-color: #e6f0ff; selection-color: #111;
}
"""

def make_dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, QColor(18,18,18))
    p.setColor(QPalette.WindowText, QColor(234,234,234))
    p.setColor(QPalette.Base, QColor(30,30,30))
    p.setColor(QPalette.AlternateBase, QColor(25,25,25))
    p.setColor(QPalette.ToolTipBase, QColor(255,255,220))
    p.setColor(QPalette.ToolTipText, QColor(0,0,0))
    p.setColor(QPalette.Text, QColor(230,230,230))
    p.setColor(QPalette.Button, QColor(43,43,43))
    p.setColor(QPalette.ButtonText, QColor(230,230,230))
    p.setColor(QPalette.BrightText, QColor(255,0,0))
    p.setColor(QPalette.Link, QColor(42,130,218))
    p.setColor(QPalette.Highlight, QColor(56,120,217))
    p.setColor(QPalette.HighlightedText, QColor(0,0,0))
    try:
        p.setColor(QPalette.PlaceholderText, QColor(200,200,200,128))
    except Exception:
        pass
    return p

def make_light_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, QColor("#ffffff"))
    p.setColor(QPalette.WindowText, QColor("#111111"))
    p.setColor(QPalette.Base, QColor("#ffffff"))
    p.setColor(QPalette.AlternateBase, QColor("#fafafa"))
    p.setColor(QPalette.ToolTipBase, QColor("#ffffdd"))
    p.setColor(QPalette.ToolTipText, QColor("#111111"))
    p.setColor(QPalette.Text, QColor("#111111"))
    p.setColor(QPalette.Button, QColor("#f6f6f6"))
    p.setColor(QPalette.ButtonText, QColor("#111111"))
    p.setColor(QPalette.BrightText, QColor("#ff0000"))
    p.setColor(QPalette.Link, QColor("#2a82da"))
    p.setColor(QPalette.Highlight, QColor("#cde2ff"))
    p.setColor(QPalette.HighlightedText, QColor("#111111"))
    try:
        p.setColor(QPalette.PlaceholderText, QColor(0,0,0,128))
    except Exception:
        pass
    return p

class ThemeManager(QObject):
    theme_changed = Signal(str)
    def __init__(self):
        super().__init__()
        self.settings = QSettings("HospitalApp", "UI")
        self.theme = self.settings.value("theme", "light")
    def set_theme(self, theme: str):
        theme = "dark" if str(theme).lower().startswith("d") else "light"
        if theme == self.theme: return
        self.theme = theme
        self.settings.setValue("theme", self.theme)
        self.apply(QApplication.instance())
        self.theme_changed.emit(theme)
    def apply(self, app: QApplication | None):
        if not app: return
        if self.theme == "dark":
            app.setPalette(make_dark_palette())
            app.setStyleSheet(DARK_QSS)
        else:
            app.setPalette(make_light_palette())
            app.setStyleSheet(LIGHT_QSS)

THEME = ThemeManager()


# ================= Helpers: Date/Time JSON =================
def dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat(timespec="seconds") if dt else None

def dt_from_str(s: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(s) if s else None


# ================= JSON Serializer (No changes to domain) =================
def hospital_to_dict(h: Hospital) -> dict:
    return {
        "name": h.name,
        "location": h.location,
        "departments": [
            {
                "name": d.name,
                "capacity": d.capacity,
                "patients": [
                    {
                        "id": p.id,
                        "patient_id": p.patient_id,
                        "name": p.name,
                        "age": p.age,
                        "created_at": dt_to_str(getattr(p, "created_at", None)),
                        "medical_record": p.medical_record,
                        "admission_date": dt_to_str(getattr(p, "admission_date", None)),
                        "is_discharged": p.is_discharged,
                        "discharge_date": dt_to_str(getattr(p, "discharge_date", None)),
                    }
                    for p in d.patients
                ],
                "staff": [
                    {
                        "id": s.id,
                        "staff_id": s.staff_id,
                        "name": s.name,
                        "age": s.age,
                        "position": s.position,
                        "department": s.department,
                        "is_active": s.is_active,
                        "created_at": dt_to_str(getattr(s, "created_at", None)),
                    }
                    for s in d.staff
                ],
            }
            for d in h.departments.values()
        ],
    }

def hospital_from_dict(data: dict) -> Hospital:
    h = Hospital(data["name"], data["location"])
    h.departments = {}
    for d in data.get("departments", []):
        dept = Department(d["name"], int(d.get("capacity", 50)))
        dept.patients = []
        dept.staff = []

        for pd in d.get("patients", []):
            p = Patient(pd["name"], int(pd["age"]), pd.get("medical_record", ""))
            p.id = pd.get("id", p.id)
            p.patient_id = pd.get("patient_id", p.patient_id)
            p.created_at = dt_from_str(pd.get("created_at"))
            p.admission_date = dt_from_str(pd.get("admission_date"))
            p.is_discharged = bool(pd.get("is_discharged", False))
            p.discharge_date = dt_from_str(pd.get("discharge_date"))
            dept.patients.append(p)

        for sd in d.get("staff", []):
            s = Staff(sd["name"], int(sd["age"]), sd["position"], dept.name)
            s.id = sd.get("id", s.id)
            s.staff_id = sd.get("staff_id", s.staff_id)
            s.created_at = dt_from_str(sd.get("created_at"))
            s.is_active = bool(sd.get("is_active", True))
            s.department = dept.name
            dept.staff.append(s)

        h.departments[dept.name] = dept
    return h


# ================= Appointments with conflict detection =================
class AppointmentStatus:
    SCHEDULED = "scheduled"
    CHECKED_IN = "checked-in"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

    @staticmethod
    def all():
        return [AppointmentStatus.SCHEDULED, AppointmentStatus.CHECKED_IN,
                AppointmentStatus.COMPLETED, AppointmentStatus.CANCELLED]

    @staticmethod
    def label(status: str) -> str:
        key = {
            AppointmentStatus.SCHEDULED: "appt.status.scheduled",
            AppointmentStatus.CHECKED_IN: "appt.status.checked_in",
            AppointmentStatus.COMPLETED: "appt.status.completed",
            AppointmentStatus.CANCELLED: "appt.status.cancelled",
        }.get(status, status)
        return I18N.t(key) if isinstance(key, str) else key

    @staticmethod
    def from_label(label: str) -> str:
        reverse = {AppointmentStatus.label(s): s for s in AppointmentStatus.all()}
        return reverse.get(label, label)

    @staticmethod
    def is_active(status: str) -> bool:
        return status in (AppointmentStatus.SCHEDULED, AppointmentStatus.CHECKED_IN)


class Appointment:
    def __init__(self, patient_person_id: str, staff_person_id: Optional[str],
                 dept_name: str, start: datetime, end: datetime,
                 status: str = AppointmentStatus.SCHEDULED, notes: str = "",
                 appt_id: Optional[str] = None):
        self.id = appt_id or uuid.uuid4().hex[:10]
        self.patient_person_id = patient_person_id
        self.staff_person_id = staff_person_id
        self.dept_name = dept_name
        self.start = start
        self.end = end
        self.status = status
        self.notes = notes

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "patient_person_id": self.patient_person_id,
            "staff_person_id": self.staff_person_id,
            "dept_name": self.dept_name,
            "start": dt_to_str(self.start),
            "end": dt_to_str(self.end),
            "status": self.status,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "Appointment":
        return Appointment(
            patient_person_id=d["patient_person_id"],
            staff_person_id=d.get("staff_person_id"),
            dept_name=d["dept_name"],
            start=dt_from_str(d["start"]),
            end=dt_from_str(d["end"]),
            status=d.get("status", AppointmentStatus.SCHEDULED),
            notes=d.get("notes", ""),
            appt_id=d.get("id"),
        )


class AppointmentManager:
    def __init__(self, hospital: Hospital):
        self.hospital = hospital
        self.items: List[Appointment] = []
        self._patient_index: Dict[str, Patient] = {}
        self._staff_index: Dict[str, Staff] = {}
        self.rebuild_indexes()

    def bind_hospital(self, hospital: Hospital):
        self.hospital = hospital
        self.rebuild_indexes()

    def rebuild_indexes(self):
        self._patient_index.clear()
        self._staff_index.clear()
        for d in self.hospital.departments.values():
            for p in d.patients:
                self._patient_index[p.id] = p
            for s in d.staff:
                self._staff_index[s.id] = s

    # ---- Conflicts ----
    @staticmethod
    def _overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
        return (a_start < b_end) and (a_end > b_start)

    def find_conflicts(self, patient_id: str, staff_id: Optional[str],
                       start: datetime, end: datetime, ignore_id: Optional[str] = None
                       ) -> List[Tuple[Appointment, List[str]]]:
        conflicts: List[Tuple[Appointment, List[str]]] = []
        for a in self.items:
            if ignore_id and a.id == ignore_id:
                continue
            if not AppointmentStatus.is_active(a.status):
                continue
            if not self._overlap(start, end, a.start, a.end):
                continue
            reasons: List[str] = []
            if patient_id == a.patient_person_id:
                reasons.append("patient")
            if staff_id and a.staff_person_id and staff_id == a.staff_person_id:
                reasons.append("staff")
            if reasons:
                conflicts.append((a, reasons))
        return conflicts

    # CRUD
    def add(self, patient: Patient, dept: Department, start: datetime,
            end: datetime, staff: Optional[Staff] = None, notes: str = "") -> Appointment:
        appt = Appointment(
            patient_person_id=patient.id,
            staff_person_id=(staff.id if staff else None),
            dept_name=dept.name,
            start=start, end=end,
            status=AppointmentStatus.SCHEDULED,
            notes=notes
        )
        self.items.append(appt)
        return appt

    def remove(self, appt_id: str):
        self.items = [a for a in self.items if a.id != appt_id]

    def update_status(self, appt_id: str, status: str):
        for a in self.items:
            if a.id == appt_id:
                a.status = status
                return a
        return None

    # Queries
    def list_filtered(self, day: Optional[date] = None,
                      dept_name: Optional[str] = None,
                      status: Optional[str] = None) -> List[Appointment]:
        out = self.items
        if day:
            out = [a for a in out if a.start.date() == day]
        if dept_name and dept_name != "__ALL__":
            out = [a for a in out if a.dept_name == dept_name]
        if status and status != "__ALL__":
            out = [a for a in out if a.status == status]
        return sorted(out, key=lambda a: a.start)

    # Resolve for display
    def patient_of(self, a: Appointment) -> Optional[Patient]:
        return self._patient_index.get(a.patient_person_id)
    def staff_of(self, a: Appointment) -> Optional[Staff]:
        return self._staff_index.get(a.staff_person_id) if a.staff_person_id else None

    # Serialization
    def to_dict(self) -> dict:
        return {"items": [a.to_dict() for a in self.items]}
    def from_dict(self, d: dict):
        self.items = [Appointment.from_dict(x) for x in d.get("items", [])]


# ================= Patient Details Dialog =================
class PatientDetailsDialog(QDialog):
    def __init__(self, patient: Patient, parent=None):
        super().__init__(parent)
        self.patient = patient
        self.setLayoutDirection(Qt.RightToLeft if I18N.lang == "ar" else Qt.LeftToRight)
        try:
            self.setFont(QFont("Cairo", 11))
        except Exception:
            pass

        self.form = QFormLayout(self)

        # Labels
        self.lbl_patient_id_label = QLabel("")
        self.lbl_state_label = QLabel("")
        self.lbl_name_label = QLabel("")
        self.lbl_age_label = QLabel("")
        self.lbl_med_label = QLabel("")

        self.id_val = QLabel(patient.patient_id)
        self.state_val = QLabel("")
        self.name_in = QLineEdit(patient.name)
        self.age_in = QSpinBox(); self.age_in.setRange(1, 120); self.age_in.setValue(int(patient.age))
        self.med_in = QPlainTextEdit(patient.medical_record)

        self.form.addRow(self.lbl_patient_id_label, self.id_val)
        self.form.addRow(self.lbl_state_label, self.state_val)
        self.form.addRow(self.lbl_name_label, self.name_in)
        self.form.addRow(self.lbl_age_label, self.age_in)
        self.form.addRow(self.lbl_med_label, self.med_in)

        self.info_box = QGroupBox("")
        self.info_form = QFormLayout(self.info_box)
        self.lbl_adm_label = QLabel("")
        self.lbl_dis_label = QLabel("")
        self.adm_val = QLabel(str(getattr(patient, "admission_date", "-")))
        self.dis_val = QLabel(str(getattr(patient, "discharge_date", "-")) if patient.is_discharged else "-")
        self.info_form.addRow(self.lbl_adm_label, self.adm_val)
        self.info_form.addRow(self.lbl_dis_label, self.dis_val)
        self.form.addRow(self.info_box)

        self.buttons = QDialogButtonBox()
        self.btn_save = self.buttons.addButton("", QDialogButtonBox.AcceptRole)
        self.btn_close = self.buttons.addButton("", QDialogButtonBox.RejectRole)
        self.btn_discharge = None
        if not patient.is_discharged:
            self.btn_discharge = self.buttons.addButton("", QDialogButtonBox.DestructiveRole)

        self.btn_save.clicked.connect(self.handle_save)
        self.btn_close.clicked.connect(self.reject)
        if self.btn_discharge:
            self.btn_discharge.clicked.connect(self.handle_discharge)

        self.form.addRow(self.buttons)

        I18N.language_changed.connect(self._on_lang)
        self.retranslate_ui()

    def _on_lang(self, lang: str):
        self.setLayoutDirection(Qt.RightToLeft if lang == "ar" else Qt.LeftToRight)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.setWindowTitle(I18N.t("dialog.patient_details.title", id=self.patient.patient_id))
        self.lbl_patient_id_label.setText(I18N.t("label.patient_id:"))
        self.lbl_state_label.setText(I18N.t("label.state:"))
        self.lbl_name_label.setText(I18N.t("field.name"))
        self.lbl_age_label.setText(I18N.t("field.age"))
        self.lbl_med_label.setText(I18N.t("field.medical_record"))
        self.state_val.setText(I18N.t("status.discharged") if self.patient.is_discharged else I18N.t("status.admitted"))
        self.info_box.setTitle(I18N.t("group.dates"))
        self.lbl_adm_label.setText(I18N.t("label.admission_date:"))
        self.lbl_dis_label.setText(I18N.t("label.discharge_date:"))
        if self.btn_discharge:
            self.btn_discharge.setText(I18N.t("btn.discharge"))
        self.btn_save.setText(I18N.t("btn.save"))
        self.btn_close.setText(I18N.t("btn.close"))

    def handle_save(self):
        name = self.name_in.text().strip()
        age = int(self.age_in.value())
        if not name:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.patient.name_required"))
            return
        try:
            self.patient.name = name
            self.patient.age = age
            self.patient.medical_record = self.med_in.toPlainText().strip()
            QMessageBox.information(self, I18N.t("msg.info.title"), I18N.t("msg.save.ok"))
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, I18N.t("msg.error.title"), str(e))

    def handle_discharge(self):
        notes, ok = QInputDialog.getMultiLineText(self, I18N.t("dialog.discharge.notes_title"),
                                                  I18N.t("dialog.discharge.notes_prompt"), "")
        if not ok:
            return
        try:
            self.patient.discharge(notes)
            self.state_val.setText(I18N.t("status.discharged"))
            QMessageBox.information(self, I18N.t("msg.info.title"),
                                    I18N.t("msg.patient.discharged", name=self.patient.name))
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, I18N.t("msg.error.title"), str(e))


# ================= Main Window =================
class HospitalWindow(QMainWindow):
    def __init__(self, hospital: Hospital):
        super().__init__()
        self.hospital = hospital
        self.appts = AppointmentManager(self.hospital)
        self.current_file_path: Optional[str] = None

        self.setLayoutDirection(Qt.RightToLeft if I18N.lang == "ar" else Qt.LeftToRight)
        self.setMinimumSize(1300, 780)
        try:
            self.setFont(QFont("Cairo", 11))
        except Exception:
            pass

        self.page = MainPage(self.hospital, self.appts)
        self.setCentralWidget(self.page)

        self._build_menu()
        I18N.language_changed.connect(self._on_lang)
        THEME.theme_changed.connect(self._on_theme_changed)
        self.retranslate_ui()
        self._on_theme_changed(THEME.theme)
        self.statusBar().showMessage("")

    def _on_lang(self, lang: str):
        self.setLayoutDirection(Qt.RightToLeft if lang == "ar" else Qt.LeftToRight)
        self.retranslate_ui()
        self.page.retranslate_ui()
        self.statusBar().showMessage("Language changed" if lang == "en" else "تم تغيير اللغة", 2000)

    def _on_theme_changed(self, theme: str):
        if hasattr(self, "act_theme_light"):
            self.act_theme_light.setChecked(theme == "light")
            self.act_theme_dark.setChecked(theme == "dark")
        self.statusBar().showMessage("Dark theme applied" if theme == "dark" else "Light theme applied", 2000)

    def _build_menu(self):
        menubar = self.menuBar()
        self.file_menu = menubar.addMenu("")
        self.act_new = self.file_menu.addAction("")
        self.act_open = self.file_menu.addAction("")
        self.act_save = self.file_menu.addAction("")
        self.act_save_as = self.file_menu.addAction("")
        self.act_new.triggered.connect(self.handle_new)
        self.act_open.triggered.connect(self.handle_open)
        self.act_save.triggered.connect(self.handle_save)
        self.act_save_as.triggered.connect(self.handle_save_as)

        self.lang_menu = menubar.addMenu("")
        self.act_lang_ar = self.lang_menu.addAction("")
        self.act_lang_en = self.lang_menu.addAction("")
        self.act_lang_ar.triggered.connect(lambda: I18N.set_language("ar"))
        self.act_lang_en.triggered.connect(lambda: I18N.set_language("en"))

        self.theme_menu = menubar.addMenu("")
        self.act_theme_light = self.theme_menu.addAction("")
        self.act_theme_dark = self.theme_menu.addAction("")
        self.act_theme_light.setCheckable(True)
        self.act_theme_dark.setCheckable(True)
        self.act_theme_light.triggered.connect(lambda: THEME.set_theme("light"))
        self.act_theme_dark.triggered.connect(lambda: THEME.set_theme("dark"))

    def retranslate_ui(self):
        self.setWindowTitle(I18N.t("app.title", name=self.hospital.name))
        self.file_menu.setTitle(I18N.t("menu.file"))
        self.act_new.setText(I18N.t("menu.file.new"))
        self.act_open.setText(I18N.t("menu.file.open"))
        self.act_save.setText(I18N.t("menu.file.save"))
        self.act_save_as.setText(I18N.t("menu.file.saveas"))
        self.lang_menu.setTitle(I18N.t("menu.language"))
        self.act_lang_ar.setText(I18N.t("menu.language.ar"))
        self.act_lang_en.setText(I18N.t("menu.language.en"))
        self.theme_menu.setTitle(I18N.t("menu.theme"))
        self.act_theme_light.setText(I18N.t("menu.theme.light"))
        self.act_theme_dark.setText(I18N.t("menu.theme.dark"))

    # ----- File ops -----
    def handle_new(self):
        if QMessageBox.question(self, I18N.t("msg.confirm.title"), I18N.t("msg.new.confirm")) != QMessageBox.Yes:
            return
        self.hospital = Hospital("New Hospital", "Unknown")
        self.appts = AppointmentManager(self.hospital)
        self.current_file_path = None
        self._reload_page()

    def handle_open(self):
        path, _ = QFileDialog.getOpenFileName(self, I18N.t("dialog.open.title"), "", I18N.t("dialog.json.filter"))
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            h = hospital_from_dict(data["hospital"])
            ap = AppointmentManager(h)
            ap.from_dict(data.get("appointments", {}))
            ap.bind_hospital(h)
            self.hospital = h
            self.appts = ap
            self.current_file_path = path
            self._reload_page()
            QMessageBox.information(self, I18N.t("msg.info.title"), I18N.t("msg.loaded_ok"))
        except Exception as e:
            QMessageBox.critical(self, I18N.t("msg.error.title"), I18N.t("msg.open.fail", err=e))

    def handle_save(self):
        if not self.current_file_path:
            return self.handle_save_as()
        try:
            data = {"version": 1, "hospital": hospital_to_dict(self.hospital), "appointments": self.appts.to_dict()}
            with open(self.current_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.statusBar().showMessage(I18N.t("msg.save.ok"), 3000)
        except Exception as e:
            QMessageBox.critical(self, I18N.t("msg.error.title"), I18N.t("msg.save.fail", err=e))

    def handle_save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, I18N.t("dialog.save.title"), "hospital_data.json", I18N.t("dialog.json.filter"))
        if not path:
            return
        self.current_file_path = path
        self.handle_save()

    def _reload_page(self):
        self.page.setParent(None)
        self.page = MainPage(self.hospital, self.appts)
        self.setCentralWidget(self.page)
        self.retranslate_ui()


# ================= Main Page with all tabs =================
class MainPage(QWidget):
    def __init__(self, hospital: Hospital, appts: AppointmentManager):
        super().__init__()
        self.hospital = hospital
        self.appts = appts
        self._build_ui()
        I18N.language_changed.connect(self._on_lang)
        THEME.theme_changed.connect(self._on_theme)
        self.retranslate_ui()

    def _on_lang(self, lang: str):
        self.setLayoutDirection(Qt.RightToLeft if lang == "ar" else Qt.LeftToRight)
        self.retranslate_ui()
        self._fill_appt_filter_combos()
        self.refresh_appt_table()
        self.refresh_dashboard()

    def _on_theme(self, theme: str):
        # إعادة تهيئة الرسم البياني للون الثيم
        self.refresh_dashboard()

    def retranslate_ui(self):
        # Tabs titles
        self.tabs.setTabText(self.tab_idx_dashboard, I18N.t("tab.dashboard"))
        self.tabs.setTabText(self.tab_idx_patients, I18N.t("tab.patients"))
        self.tabs.setTabText(self.tab_idx_staff, I18N.t("tab.staff"))
        self.tabs.setTabText(self.tab_idx_search, I18N.t("tab.search"))
        self.tabs.setTabText(self.tab_idx_appts, I18N.t("tab.appointments"))

        # Left
        self.lbl_depts.setText(I18N.t("label.departments"))
        self.add_dept_group.setTitle(I18N.t("group.add_dept"))
        self.add_dept_form_label_name.setText(I18N.t("field.name"))
        self.add_dept_form_label_cap.setText(I18N.t("field.capacity"))
        self.btn_add_dept.setText(I18N.t("btn.add"))

        # Info
        self.info_group.setTitle(I18N.t("group.dept_info"))
        self.info_label_name.setText(I18N.t("label.name:"))
        self.info_label_code.setText(I18N.t("label.code:"))
        self.info_label_cap.setText(I18N.t("label.capacity:"))
        self.info_label_patcnt.setText(I18N.t("label.patients_count:"))
        self.info_label_staffcnt.setText(I18N.t("label.staff_count:"))

        # Patients tab
        self.add_patient_group.setTitle(I18N.t("group.add_patient"))
        self.add_patient_label_name.setText(I18N.t("field.name"))
        self.add_patient_label_age.setText(I18N.t("field.age"))
        self.add_patient_label_med.setText(I18N.t("field.medical_record"))
        self.p_med_rec_in.setPlaceholderText(I18N.t("ph.medical_record"))
        self.btn_add_patient.setText(I18N.t("btn.add_patient"))
        self.patients_group.setTitle(I18N.t("group.patients_list"))
        self.only_active_chk.setText(I18N.t("chk.only_active"))
        self.btn_discharge.setText(I18N.t("btn.discharge_selected"))
        self.btn_move.setText(I18N.t("btn.move"))
        self.btn_p_refresh.setText(I18N.t("btn.refresh"))

        # Staff tab
        self.add_staff_group.setTitle(I18N.t("group.add_staff"))
        self.add_staff_label_name.setText(I18N.t("field.name"))
        self.add_staff_label_age.setText(I18N.t("field.age"))
        self.add_staff_label_pos.setText(I18N.t("field.position"))
        self.s_position_in.setPlaceholderText(I18N.t("ph.position_hint"))
        self.btn_add_staff.setText(I18N.t("btn.add_staff"))
        self.staff_group.setTitle(I18N.t("group.staff_list"))
        self.position_filter_in.setPlaceholderText(I18N.t("ph.position_filter"))
        self.btn_toggle_active.setText(I18N.t("btn.toggle_active"))
        self.btn_s_refresh.setText(I18N.t("btn.refresh"))

        # Search tab
        self.search_group.setTitle(I18N.t("group.search"))
        self.search_label_term.setText(I18N.t("search.field.label"))
        self.search_all_in.setPlaceholderText(I18N.t("ph.search_all"))

        # Appointments tab
        self.create_appt_group.setTitle(I18N.t("group.create_appt"))
        self.ap_label_dept.setText(I18N.t("field.dept"))
        self.ap_label_patient.setText(I18N.t("field.patient"))
        self.ap_label_staff.setText(I18N.t("field.staff"))
        self.ap_label_start.setText(I18N.t("field.start"))
        self.ap_label_end.setText(I18N.t("field.end"))
        self.ap_label_notes.setText(I18N.t("field.notes"))
        self.btn_add_appt.setText(I18N.t("btn.add_appt"))
        self.appt_list_group.setTitle(I18N.t("group.appt_list"))
        self.ap_filter_label_date.setText(I18N.t("filter.date"))
        self.ap_filter_label_dept.setText(I18N.t("filter.dept"))
        self.ap_filter_label_status.setText(I18N.t("filter.status"))
        headers = [
            I18N.t("col.id"), I18N.t("col.dept"), I18N.t("col.patient"),
            I18N.t("col.staff"), I18N.t("col.start"), I18N.t("col.end"),
            I18N.t("col.status"), I18N.t("col.notes"), I18N.t("col.conflict")
        ]
        self.ap_table.setHorizontalHeaderLabels(headers)
        self.btn_appt_status.setText(I18N.t("btn.change_status"))
        self.btn_appt_delete.setText(I18N.t("btn.delete_appt"))
        self.btn_a_refresh.setText(I18N.t("btn.refresh"))

        # Dashboard
        self.dash_group_stats.setTitle(I18N.t("dash.stats"))
        self.dash_label_total_depts.setText(I18N.t("dash.total_depts"))
        self.dash_label_total_patients.setText(I18N.t("dash.total_patients"))
        self.dash_label_active_patients.setText(I18N.t("dash.active_patients"))
        self.dash_label_total_staff.setText(I18N.t("dash.total_staff"))
        self.dash_label_appts_today.setText(I18N.t("dash.appts_today"))
        self.btn_dash_refresh.setText(I18N.t("dash.refresh"))
        self.dash_group_chart.setTitle(I18N.t("dash.chart.appts_by_status"))
        if not HAS_QTCHARTS:
            self.dash_chart_placeholder.setText(I18N.t("dash.no_qtcharts"))

        # Refresh data-bound views
        self.refresh_department_list()
        self.refresh_patients_list()
        self.refresh_staff_list()
        self._fill_dept_combos()
        self._ap_on_dept_changed()
        self._fill_appt_filter_combos()
        self.refresh_appt_table()
        self.refresh_dashboard()

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left: Departments panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.lbl_depts = QLabel("")
        self.dept_list = QListWidget()
        self.dept_list.setAlternatingRowColors(True)
        self.dept_list.currentItemChanged.connect(self.on_dept_changed)

        self.add_dept_group = QGroupBox("")
        self.add_dept_form = QFormLayout(self.add_dept_group)
        self.dept_name_in = QLineEdit()
        self.dept_cap_in = QSpinBox(); self.dept_cap_in.setRange(10, 2000); self.dept_cap_in.setValue(50)
        self.btn_add_dept = QPushButton("")
        self.btn_add_dept.clicked.connect(self.handle_add_department)

        self.add_dept_form_label_name = QLabel("")
        self.add_dept_form_label_cap = QLabel("")
        self.add_dept_form.addRow(self.add_dept_form_label_name, self.dept_name_in)
        self.add_dept_form.addRow(self.add_dept_form_label_cap, self.dept_cap_in)
        self.add_dept_form.addRow(self.btn_add_dept)

        left_layout.addWidget(self.lbl_depts)
        left_layout.addWidget(self.dept_list, 1)
        left_layout.addWidget(self.add_dept_group, 0)

        # Right: Info + Tabs
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.info_group = QGroupBox("")
        info_layout = QFormLayout(self.info_group)
        self.lbl_dept_name = QLabel("-")
        self.lbl_dept_code = QLabel("-")
        self.lbl_dept_capacity = QLabel("-")
        self.lbl_patients_count = QLabel("-")
        self.lbl_staff_count = QLabel("-")
        self.info_label_name = QLabel("")
        self.info_label_code = QLabel("")
        self.info_label_cap = QLabel("")
        self.info_label_patcnt = QLabel("")
        self.info_label_staffcnt = QLabel("")
        info_layout.addRow(self.info_label_name, self.lbl_dept_name)
        info_layout.addRow(self.info_label_code, self.lbl_dept_code)
        info_layout.addRow(self.info_label_cap, self.lbl_dept_capacity)
        info_layout.addRow(self.info_label_patcnt, self.lbl_patients_count)
        info_layout.addRow(self.info_label_staffcnt, self.lbl_staff_count)

        self.tabs = QTabWidget()
        self._build_dashboard_tab()
        self._build_patients_tab()
        self._build_staff_tab()
        self._build_search_tab()
        self._build_appointments_tab()

        right_layout.addWidget(self.info_group)
        right_layout.addWidget(self.tabs, 1)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([320, 980])

        self.refresh_department_list()

    # ---------- Dashboard tab ----------
    def _build_dashboard_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Stats
        self.dash_group_stats = QGroupBox("")
        stats_form = QFormLayout(self.dash_group_stats)
        self.dash_label_total_depts = QLabel("")
        self.dash_label_total_patients = QLabel("")
        self.dash_label_active_patients = QLabel("")
        self.dash_label_total_staff = QLabel("")
        self.dash_label_appts_today = QLabel("")
        self.dash_val_total_depts = QLabel("-")
        self.dash_val_total_patients = QLabel("-")
        self.dash_val_active_patients = QLabel("-")
        self.dash_val_total_staff = QLabel("-")
        self.dash_val_appts_today = QLabel("-")
        stats_form.addRow(self.dash_label_total_depts, self.dash_val_total_depts)
        stats_form.addRow(self.dash_label_total_patients, self.dash_val_total_patients)
        stats_form.addRow(self.dash_label_active_patients, self.dash_val_active_patients)
        stats_form.addRow(self.dash_label_total_staff, self.dash_val_total_staff)
        stats_form.addRow(self.dash_label_appts_today, self.dash_val_appts_today)

        # Chart
        self.dash_group_chart = QGroupBox("")
        chart_vbox = QVBoxLayout(self.dash_group_chart)
        self.btn_dash_refresh = QPushButton("")
        self.btn_dash_refresh.clicked.connect(self.refresh_dashboard)
        chart_vbox.addWidget(self.btn_dash_refresh, alignment=Qt.AlignLeft)

        if HAS_QTCHARTS:
            self.chart = QChart()
            self.chart_view = QChartView(self.chart)
            self.chart_view.setRenderHint(self.chart_view.renderHints())
            chart_vbox.addWidget(self.chart_view, 1)
        else:
            self.dash_chart_placeholder = QLabel("")
            self.dash_chart_placeholder.setAlignment(Qt.AlignCenter)
            chart_vbox.addWidget(self.dash_chart_placeholder, 1)

        layout.addWidget(self.dash_group_stats)
        layout.addWidget(self.dash_group_chart, 1)

        self.tab_idx_dashboard = self.tabs.addTab(tab, "")

    def refresh_dashboard(self):
        # Stats
        total_depts = len(self.hospital.departments)
        total_patients = sum(len(d.patients) for d in self.hospital.departments.values())
        active_patients = sum(len(d.get_active_patients()) for d in self.hospital.departments.values())
        total_staff = sum(len(d.staff) for d in self.hospital.departments.values())
        today = date.today()
        appts_today = sum(1 for a in self.appts.items if a.start.date() == today)

        self.dash_val_total_depts.setText(str(total_depts))
        self.dash_val_total_patients.setText(str(total_patients))
        self.dash_val_active_patients.setText(str(active_patients))
        self.dash_val_total_staff.setText(str(total_staff))
        self.dash_val_appts_today.setText(str(appts_today))

        # Chart
        if not HAS_QTCHARTS:
            return

        # Counts by status for today
        statuses = AppointmentStatus.all()
        counts = {s: 0 for s in statuses}
        for a in self.appts.items:
            if a.start.date() == today:
                counts[a.status] = counts.get(a.status, 0) + 1

        series = QBarSeries()
        set0 = QBarSet("")
        set0.append([counts[s] for s in statuses])
        series.append(set0)

        chart = QChart()
        chart.addSeries(series)
        categories = [AppointmentStatus.label(s) for s in statuses]
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        axis_y = QValueAxis()
        axis_y.setRange(0, max([0] + list(counts.values())))
        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
        chart.setTitle(I18N.t("dash.chart.appts_by_status"))

        # Theme adapts to app theme
        try:
            chart.setTheme(QChart.ChartThemeDark if THEME.theme == "dark" else QChart.ChartThemeLight)
        except Exception:
            pass

        self.chart = chart
        self.chart_view.setChart(self.chart)

    # ---------- Patients tab ----------
    def _build_patients_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.add_patient_group = QGroupBox("")
        form = QFormLayout(self.add_patient_group)
        self.p_name_in = QLineEdit()
        self.p_age_in = QSpinBox(); self.p_age_in.setRange(1, 120)
        self.p_med_rec_in = QPlainTextEdit()
        self.btn_add_patient = QPushButton("")
        self.btn_add_patient.clicked.connect(self.handle_add_patient)

        self.add_patient_label_name = QLabel("")
        self.add_patient_label_age = QLabel("")
        self.add_patient_label_med = QLabel("")
        form.addRow(self.add_patient_label_name, self.p_name_in)
        form.addRow(self.add_patient_label_age, self.p_age_in)
        form.addRow(self.add_patient_label_med, self.p_med_rec_in)
        form.addRow(self.btn_add_patient)

        self.patients_group = QGroupBox("")
        list_layout = QVBoxLayout(self.patients_group)
        self.only_active_chk = QCheckBox("")
        self.only_active_chk.setChecked(True)
        self.only_active_chk.stateChanged.connect(self.refresh_patients_list)

        self.patients_list = QListWidget()
        self.patients_list.setAlternatingRowColors(True)
        self.patients_list.itemDoubleClicked.connect(self.handle_open_patient_details)

        btns_row = QHBoxLayout()
        self.btn_discharge = QPushButton("")
        self.btn_move = QPushButton("")
        self.btn_p_refresh = QPushButton("")
        self.btn_discharge.clicked.connect(self.handle_discharge_patient)
        self.btn_move.clicked.connect(self.handle_move_patient)
        self.btn_p_refresh.clicked.connect(self.refresh_patients_list)
        btns_row.addWidget(self.btn_discharge)
        btns_row.addWidget(self.btn_move)
        btns_row.addWidget(self.btn_p_refresh)
        btns_row.addStretch()

        list_layout.addWidget(self.only_active_chk)
        list_layout.addWidget(self.patients_list, 1)
        list_layout.addLayout(btns_row)

        layout.addWidget(self.add_patient_group)
        layout.addWidget(self.patients_group, 1)

        self.tab_idx_patients = self.tabs.addTab(tab, "")

    # ---------- Staff tab ----------
    def _build_staff_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.add_staff_group = QGroupBox("")
        form = QFormLayout(self.add_staff_group)
        self.s_name_in = QLineEdit()
        self.s_age_in = QSpinBox(); self.s_age_in.setRange(18, 100)
        self.s_position_in = QLineEdit()
        self.btn_add_staff = QPushButton("")
        self.btn_add_staff.clicked.connect(self.handle_add_staff)

        self.add_staff_label_name = QLabel("")
        self.add_staff_label_age = QLabel("")
        self.add_staff_label_pos = QLabel("")
        form.addRow(self.add_staff_label_name, self.s_name_in)
        form.addRow(self.add_staff_label_age, self.s_age_in)
        form.addRow(self.add_staff_label_pos, self.s_position_in)
        form.addRow(self.btn_add_staff)

        self.staff_group = QGroupBox("")
        list_layout = QVBoxLayout(self.staff_group)
        self.position_filter_in = QLineEdit()
        self.position_filter_in.textChanged.connect(self.refresh_staff_list)

        self.staff_list = QListWidget()
        self.staff_list.setAlternatingRowColors(True)
        btns_row = QHBoxLayout()
        self.btn_toggle_active = QPushButton("")
        self.btn_toggle_active.clicked.connect(self.handle_toggle_staff)
        self.btn_s_refresh = QPushButton("")
        self.btn_s_refresh.clicked.connect(self.refresh_staff_list)
        btns_row.addWidget(self.btn_toggle_active)
        btns_row.addWidget(self.btn_s_refresh)
        btns_row.addStretch()

        list_layout.addWidget(self.position_filter_in)
        list_layout.addWidget(self.staff_list, 1)
        list_layout.addLayout(btns_row)

        layout.addWidget(self.add_staff_group)
        layout.addWidget(self.staff_group, 1)

        self.tab_idx_staff = self.tabs.addTab(tab, "")

    # ---------- Search tab ----------
    def _build_search_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.search_group = QGroupBox("")
        form = QFormLayout(self.search_group)
        self.search_all_in = QLineEdit()
        self.search_label_term = QLabel("")
        self.search_all_in.textChanged.connect(self.handle_search_all)
        form.addRow(self.search_label_term, self.search_all_in)

        self.search_results = QListWidget()
        self.search_results.setAlternatingRowColors(True)
        self.search_results.itemDoubleClicked.connect(self.handle_search_navigate)

        layout.addWidget(self.search_group)
        layout.addWidget(self.search_results, 1)

        self.tab_idx_search = self.tabs.addTab(tab, "")

    # ---------- Appointments tab ----------
    def _build_appointments_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.create_appt_group = QGroupBox("")
        form = QFormLayout(self.create_appt_group)

        self.ap_dept_combo = QComboBox()
        self.ap_dept_combo.currentIndexChanged.connect(self._ap_on_dept_changed)
        self.ap_patient_combo = QComboBox()
        self.ap_staff_combo = QComboBox()

        self.ap_start_dt = QDateTimeEdit(QDateTime.currentDateTime()); self.ap_start_dt.setCalendarPopup(True)
        self.ap_end_dt = QDateTimeEdit(QDateTime.currentDateTime().addSecs(1800)); self.ap_end_dt.setCalendarPopup(True)
        self.ap_notes_in = QLineEdit()
        self.btn_add_appt = QPushButton("")
        self.btn_add_appt.clicked.connect(self.handle_add_appointment)

        self.ap_label_dept = QLabel("")
        self.ap_label_patient = QLabel("")
        self.ap_label_staff = QLabel("")
        self.ap_label_start = QLabel("")
        self.ap_label_end = QLabel("")
        self.ap_label_notes = QLabel("")

        form.addRow(self.ap_label_dept, self.ap_dept_combo)
        form.addRow(self.ap_label_patient, self.ap_patient_combo)
        form.addRow(self.ap_label_staff, self.ap_staff_combo)
        form.addRow(self.ap_label_start, self.ap_start_dt)
        form.addRow(self.ap_label_end, self.ap_end_dt)
        form.addRow(self.ap_label_notes, self.ap_notes_in)
        form.addRow(self.btn_add_appt)

        self.appt_list_group = QGroupBox("")
        v = QVBoxLayout(self.appt_list_group)
        filters = QHBoxLayout()
        self.ap_filter_label_date = QLabel("")
        self.ap_filter_label_dept = QLabel("")
        self.ap_filter_label_status = QLabel("")
        self.ap_filter_date = QDateEdit(QDate.currentDate()); self.ap_filter_date.setCalendarPopup(True)
        self.ap_filter_date.dateChanged.connect(self.refresh_appt_table)
        self.ap_filter_dept = QComboBox(); self.ap_filter_dept.currentIndexChanged.connect(self.refresh_appt_table)
        self.ap_filter_status = QComboBox(); self.ap_filter_status.currentIndexChanged.connect(self.refresh_appt_table)

        filters.addWidget(self.ap_filter_label_date); filters.addWidget(self.ap_filter_date)
        filters.addWidget(self.ap_filter_label_dept); filters.addWidget(self.ap_filter_dept)
        filters.addWidget(self.ap_filter_label_status); filters.addWidget(self.ap_filter_status)
        filters.addStretch()

        self.ap_table = QTableWidget(0, 9)  # +1 عمود للتعارض
        self.ap_table.setHorizontalHeaderLabels([""]*9)
        self.ap_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ap_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.ap_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ap_table.setAlternatingRowColors(True)

        btns = QHBoxLayout()
        self.btn_appt_status = QPushButton("")
        self.btn_appt_status.clicked.connect(self.handle_change_appt_status)
        self.btn_appt_delete = QPushButton("")
        self.btn_appt_delete.clicked.connect(self.handle_delete_appt)
        self.btn_a_refresh = QPushButton("")
        self.btn_a_refresh.clicked.connect(self.refresh_appt_table)
        btns.addWidget(self.btn_appt_status); btns.addWidget(self.btn_appt_delete); btns.addWidget(self.btn_a_refresh); btns.addStretch()

        v.addLayout(filters)
        v.addWidget(self.ap_table, 1)
        v.addLayout(btns)

        layout.addWidget(self.create_appt_group)
        layout.addWidget(self.appt_list_group, 1)

        self.tab_idx_appts = self.tabs.addTab(tab, "")

        self._fill_dept_combos()
        self._ap_on_dept_changed()
        self._fill_appt_filter_combos()
        self.refresh_appt_table()

    # ---------- Helpers ----------
    def refresh_department_list(self):
        self.dept_list.clear()
        for name, dept in self.hospital.departments.items():
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, dept)
            self.dept_list.addItem(item)
        if self.dept_list.count() > 0 and self.dept_list.currentRow() == -1:
            self.dept_list.setCurrentRow(0)

    def on_dept_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        self.update_dept_info()
        self.refresh_patients_list()
        self.refresh_staff_list()
        self._fill_dept_combos_if_needed()

    def current_department(self) -> Optional[Department]:
        item = self.dept_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def update_dept_info(self):
        dept = self.current_department()
        if not dept:
            self.lbl_dept_name.setText("-")
            self.lbl_dept_code.setText("-")
            self.lbl_dept_capacity.setText("-")
            self.lbl_patients_count.setText("-")
            self.lbl_staff_count.setText("-")
            return
        self.lbl_dept_name.setText(dept.name)
        self.lbl_dept_code.setText(dept.dept_code)
        self.lbl_dept_capacity.setText(str(dept.capacity))
        self.lbl_patients_count.setText(f"{len(dept.patients)} ({I18N.t('chk.only_active')}: {len(dept.get_active_patients())})")
        self.lbl_staff_count.setText(str(len(dept.staff)))

    def refresh_patients_list(self):
        self.patients_list.clear()
        dept = self.current_department()
        if not dept: return
        patients = dept.get_active_patients() if self.only_active_chk.isChecked() else dept.patients
        for p in patients:
            status = I18N.t("status.admitted") if not p.is_discharged else I18N.t("status.discharged")
            text = f"{p.patient_id} | {p.name} | {p.age} {I18N.t('unit.years')} | {status}"
            item = QListWidgetItem(text); item.setData(Qt.UserRole, p)
            self.patients_list.addItem(item)
        self.update_dept_info()
        self.refresh_dashboard()

    def refresh_staff_list(self):
        self.staff_list.clear()
        dept = self.current_department()
        if not dept: return
        pos_filter = self.position_filter_in.text().strip()
        staff_list = dept.get_staff_by_position(pos_filter) if pos_filter else dept.staff
        for s in staff_list:
            status = I18N.t("staff.active") if s.is_active else I18N.t("staff.inactive")
            text = f"{s.staff_id} | {s.name} | {s.position} | {status}"
            item = QListWidgetItem(text); item.setData(Qt.UserRole, s)
            self.staff_list.addItem(item)
        self.update_dept_info()
        self.refresh_dashboard()

    # ---------- Departments ----------
    def handle_add_department(self):
        name = self.dept_name_in.text().strip()
        cap = int(self.dept_cap_in.value())
        if not name:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("field.name") + " ?")
            return
        try:
            if name in self.hospital.departments:
                raise ValueError(f"Department {name} already exists")
            d = Department(name, cap)
            self.hospital.departments[name] = d
            self.dept_name_in.clear(); self.dept_cap_in.setValue(50)
            self.refresh_department_list()
            self._fill_dept_combos_if_needed()
            QMessageBox.information(self, I18N.t("msg.info.title"), I18N.t("btn.add") + " ✓")
        except Exception as e:
            QMessageBox.critical(self, I18N.t("msg.error.title"), str(e))

    # ---------- Patients ----------
    def handle_add_patient(self):
        dept = self.current_department()
        if not dept:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.select_department_first")); return
        name = self.p_name_in.text().strip()
        age = int(self.p_age_in.value())
        med = self.p_med_rec_in.toPlainText().strip()
        if not name:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.patient.name_required")); return
        try:
            patient = Patient(name, age, med)
            if len(dept.patients) >= dept.capacity:
                QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.dept.full", name=name)); return
            dept.patients.append(patient)
            self.p_name_in.clear(); self.p_age_in.setValue(1); self.p_med_rec_in.clear()
            self.refresh_patients_list()
            self.appts.rebuild_indexes()
            self._ap_on_dept_changed()
            QMessageBox.information(self, I18N.t("msg.info.title"), I18N.t("msg.patient.added", pid=patient.patient_id))
        except Exception as e:
            QMessageBox.critical(self, I18N.t("msg.error.title"), str(e))

    def handle_discharge_patient(self):
        item = self.patients_list.currentItem()
        if not item:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.select_patient")); return
        patient: Patient = item.data(Qt.UserRole)
        if patient.is_discharged:
            QMessageBox.information(self, I18N.t("msg.warning.title"), I18N.t("msg.patient.already_discharged")); return
        notes, ok = QInputDialog.getMultiLineText(self, I18N.t("dialog.discharge.notes_title"),
                                                  I18N.t("dialog.discharge.notes_prompt"), "")
        if not ok: return
        try:
            patient.discharge(notes)
            self.refresh_patients_list()
            QMessageBox.information(self, I18N.t("msg.info.title"), I18N.t("msg.patient.discharged", name=patient.name))
        except Exception as e:
            QMessageBox.critical(self, I18N.t("msg.error.title"), str(e))

    def handle_open_patient_details(self, item: QListWidgetItem):
        patient: Patient = item.data(Qt.UserRole)
        dlg = PatientDetailsDialog(patient, self)
        if dlg.exec():
            self.refresh_patients_list()

    def handle_move_patient(self):
        item = self.patients_list.currentItem()
        dept_from = self.current_department()
        if not item or not dept_from:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.select_patient")); return
        patient: Patient = item.data(Qt.UserRole)
        if patient.is_discharged:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.patient.already_discharged")); return

        depts = [d for d in self.hospital.departments.values() if d is not dept_from]
        if not depts:
            QMessageBox.information(self, I18N.t("msg.warning.title"), I18N.t("msg.no_other_departments")); return
        names = [d.name for d in depts]
        name, ok = QInputDialog.getItem(self, I18N.t("dialog.move_patient.title"),
                                        I18N.t("dialog.move_patient.prompt"), names, 0, False)
        if not ok: return
        dept_to = self.hospital.departments.get(name)
        if len(dept_to.patients) >= dept_to.capacity:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.dept.full_with_name", name=dept_to.name)); return

        try:
            dept_from.patients = [p for p in dept_from.patients if p is not patient]
            dept_to.patients.append(patient)
            self.refresh_patients_list()
            self.update_dept_info()
            self.appts.rebuild_indexes()
            moved = 0
            for a in self.appts.items:
                if a.patient_person_id == patient.id and a.status in (AppointmentStatus.SCHEDULED, AppointmentStatus.CHECKED_IN):
                    a.dept_name = dept_to.name
                    moved += 1
            if moved:
                self.refresh_appt_table()
            QMessageBox.information(self, I18N.t("msg.info.title"),
                                    I18N.t("msg.patient.moved_to", pname=patient.name, dname=dept_to.name))
        except Exception as e:
            QMessageBox.critical(self, I18N.t("msg.error.title"), str(e))

    # ---------- Staff ----------
    def handle_add_staff(self):
        dept = self.current_department()
        if not dept:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.select_department_first")); return
        name = self.s_name_in.text().strip()
        age = int(self.s_age_in.value())
        position = self.s_position_in.text().strip()
        if not name or not position:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.staff.name_position_required")); return
        try:
            staff = Staff(name, age, position, dept.name)
            dept.staff.append(staff)
            self.s_name_in.clear(); self.s_age_in.setValue(18); self.s_position_in.clear()
            self.refresh_staff_list()
            self.appts.rebuild_indexes()
            self._ap_on_dept_changed()
            QMessageBox.information(self, I18N.t("msg.info.title"), I18N.t("msg.staff.added", sid=staff.staff_id))
        except Exception as e:
            QMessageBox.critical(self, I18N.t("msg.error.title"), str(e))

    def handle_toggle_staff(self):
        item = self.staff_list.currentItem()
        if not item:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.select_patient")); return
        s: Staff = item.data(Qt.UserRole)
        try:
            s.is_active = not s.is_active
            self.refresh_staff_list()
        except Exception as e:
            QMessageBox.critical(self, I18N.t("msg.error.title"), str(e))

    # ---------- Global Search ----------
    def handle_search_all(self, text: str):
        text = text.strip().lower(); self.search_results.clear()
        if not text: return
        for dept in self.hospital.departments.values():
            for p in dept.patients:
                hay = f"{p.name} {p.patient_id}".lower()
                if text in hay:
                    status = I18N.t("status.admitted") if not p.is_discharged else I18N.t("status.discharged")
                    item = QListWidgetItem(f"[{dept.name}] {p.patient_id} | {p.name} | {status}")
                    item.setData(Qt.UserRole, (dept, p))
                    self.search_results.addItem(item)

    def handle_search_navigate(self, item: QListWidgetItem):
        data = item.data(Qt.UserRole)
        if not data: return
        dept, patient = data
        self.goto_dept_and_select_patient(dept, patient)

    def goto_dept_and_select_patient(self, dept: Department, patient: Patient):
        for i in range(self.dept_list.count()):
            it = self.dept_list.item(i)
            if it.data(Qt.UserRole) is dept:
                self.dept_list.setCurrentRow(i)
                break
        self.tabs.setCurrentIndex(self.tab_idx_patients)
        self.refresh_patients_list()
        for i in range(self.patients_list.count()):
            it = self.patients_list.item(i)
            if it.data(Qt.UserRole) is patient:
                self.patients_list.setCurrentRow(i)
                break

    # ---------- Appointments: UI helpers ----------
    def _fill_dept_combos(self):
        self.ap_dept_combo.blockSignals(True)
        self.ap_dept_combo.clear()
        for d in self.hospital.departments.values():
            self.ap_dept_combo.addItem(d.name, d)
        self.ap_dept_combo.blockSignals(False)

    def _fill_dept_combos_if_needed(self):
        self._fill_dept_combos()
        self._ap_on_dept_changed()
        self._fill_appt_filter_combos()
        self.refresh_appt_table()
        self.refresh_dashboard()

    def _ap_on_dept_changed(self):
        dept = self.ap_dept_combo.currentData()
        self.ap_patient_combo.clear()
        self.ap_staff_combo.clear()
        if not dept: return
        for p in dept.patients:
            self.ap_patient_combo.addItem(f"{p.patient_id} - {p.name}", p)
        for s in dept.staff:
            if s.is_active:
                self.ap_staff_combo.addItem(f"{s.staff_id} - {s.name} ({s.position})", s)

    def _fill_appt_filter_combos(self):
        self.ap_filter_dept.blockSignals(True)
        self.ap_filter_status.blockSignals(True)
        self.ap_filter_dept.clear(); self.ap_filter_status.clear()

        self.ap_filter_dept.addItem(I18N.t("filter.all"), "__ALL__")
        for d in self.hospital.departments.values():
            self.ap_filter_dept.addItem(d.name, d.name)

        self.ap_filter_status.addItem(I18N.t("filter.all"), "__ALL__")
        for s in AppointmentStatus.all():
            self.ap_filter_status.addItem(AppointmentStatus.label(s), s)

        self.ap_filter_dept.blockSignals(False)
        self.ap_filter_status.blockSignals(False)

    # ---------- Appointments: Handlers ----------
    def handle_add_appointment(self):
        dept: Department = self.ap_dept_combo.currentData()
        if not dept:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.select_dept")); return
        p: Patient = self.ap_patient_combo.currentData()
        s: Optional[Staff] = self.ap_staff_combo.currentData()
        if not p:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.select_patient_for_appt")); return

        def to_py(dt: QDateTime) -> datetime:
            try:
                return dt.toPython()
            except Exception:
                secs = dt.toSecsSinceEpoch()
                return datetime.fromtimestamp(secs)

        start = to_py(self.ap_start_dt.dateTime())
        end = to_py(self.ap_end_dt.dateTime())
        if end <= start:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.end_after_start")); return

        # Conflict check
        conflicts = self.appts.find_conflicts(p.id, (s.id if s else None), start, end)
        if conflicts:
            details = []
            for a, reasons in conflicts:
                who_parts = []
                if "patient" in reasons:
                    who_parts.append(I18N.t("who.patient"))
                if "staff" in reasons:
                    who_parts.append(I18N.t("who.staff"))
                who = " & ".join(who_parts) if I18N.lang == "en" else " و ".join(who_parts)
                details.append(I18N.t("msg.appt.conflict.row", id=a.id, dept=a.dept_name,
                                      start=a.start.strftime("%H:%M"), end=a.end.strftime("%H:%M"), who=who))
            QMessageBox.critical(self, I18N.t("msg.appt.conflict.title"),
                                 I18N.t("msg.appt.conflict.body", details="\n".join(details)))
            return

        a = self.appts.add(p, dept, start, end, s, self.ap_notes_in.text().strip())
        self.ap_notes_in.clear()
        self.refresh_appt_table()
        self.refresh_dashboard()
        QMessageBox.information(self, I18N.t("msg.info.title"), I18N.t("msg.appt.created", id=a.id))

    def refresh_appt_table(self):
        # Determine selected date
        if hasattr(self.ap_filter_date.date(), "toPython"):
            day = self.ap_filter_date.date().toPython()
        else:
            d = self.ap_filter_date.date()
            day = date(d.year(), d.month(), d.day())

        dept_name = self.ap_filter_dept.currentData() if self.ap_filter_dept.count() else "__ALL__"
        status = self.ap_filter_status.currentData() if self.ap_filter_status.count() else "__ALL__"
        items = self.appts.list_filtered(day=day, dept_name=dept_name, status=status)

        self.ap_table.setRowCount(0)
        for a in items:
            row = self.ap_table.rowCount()
            self.ap_table.insertRow(row)
            p = self.appts.patient_of(a)
            s = self.appts.staff_of(a)

            def set_item(col, text):
                it = QTableWidgetItem(text)
                if col == 0:
                    it.setData(Qt.UserRole, a.id)
                self.ap_table.setItem(row, col, it)

            set_item(0, a.id)
            set_item(1, a.dept_name)
            set_item(2, f"{p.patient_id} - {p.name}" if p else "(N/A)")
            set_item(3, f"{s.staff_id} - {s.name}" if s else I18N.t("filter.all"))
            set_item(4, a.start.strftime("%Y-%m-%d %H:%M"))
            set_item(5, a.end.strftime("%Y-%m-%d %H:%M"))
            set_item(6, AppointmentStatus.label(a.status))
            set_item(7, a.notes or "")

            # Conflict highlight
            conflict_list = self.appts.find_conflicts(a.patient_person_id, a.staff_person_id, a.start, a.end, ignore_id=a.id)
            has_conflict = AppointmentStatus.is_active(a.status) and len(conflict_list) > 0
            conflict_symbol = "⚠" if has_conflict else ""
            set_item(8, conflict_symbol)

            if has_conflict:
                tooltip_lines = []
                for c, reasons in conflict_list:
                    who_parts = []
                    if "patient" in reasons: who_parts.append(I18N.t("who.patient"))
                    if "staff" in reasons: who_parts.append(I18N.t("who.staff"))
                    who = " & ".join(who_parts) if I18N.lang == "en" else " و ".join(who_parts)
                    tooltip_lines.append(f"#{c.id} | {c.dept_name} | {c.start.strftime('%H:%M')}-{c.end.strftime('%H:%M')} | {who}")
                tooltip = "\n".join(tooltip_lines)
                for col in range(0, 9):
                    it = self.ap_table.item(row, col)
                    if it:
                        it.setBackground(QBrush(QColor(255, 235, 205)))  # خفيف برتقالي
                        if col == 8 or col == 0:
                            it.setToolTip(tooltip)

    def _current_selected_appt(self) -> Optional[Appointment]:
        row = self.ap_table.currentRow()
        if row < 0: return None
        appt_id = self.ap_table.item(row, 0).text()
        for a in self.appts.items:
            if a.id == appt_id:
                return a
        return None

    def handle_change_appt_status(self):
        a = self._current_selected_appt()
        if not a:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.select_appt")); return
        choices = [AppointmentStatus.label(s) for s in AppointmentStatus.all()]
        current_label = AppointmentStatus.label(a.status)
        label, ok = QInputDialog.getItem(self, I18N.t("dialog.change_status.title"),
                                         I18N.t("dialog.change_status.prompt"),
                                         choices, choices.index(current_label) if current_label in choices else 0, False)
        if not ok: return
        new_status = AppointmentStatus.from_label(label)
        self.appts.update_status(a.id, new_status)
        self.refresh_appt_table()
        self.refresh_dashboard()

    def handle_delete_appt(self):
        a = self._current_selected_appt()
        if not a:
            QMessageBox.warning(self, I18N.t("msg.warning.title"), I18N.t("msg.select_appt")); return
        if QMessageBox.question(self, I18N.t("msg.confirm.title"), I18N.t("dialog.delete_appt.confirm", id=a.id)) != QMessageBox.Yes:
            return
        self.appts.remove(a.id)
        self.refresh_appt_table()
        self.refresh_dashboard()


# ================= main =================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # مظهر ثابت

    # لغة وثيم افتراضيين (من الإعدادات)
    I18N.set_language(I18N.lang)
    THEME.apply(app)

    # بيانات تجريبية
    hospital = Hospital("City General Hospital", "123 Main St")
    cardio = hospital.find_department("Cardiology")
    if cardio:
        try:
            p1 = Patient("Alice Johnson", 35, "Hypertension monitoring")
            p2 = Patient("Bob Smith", 52, "Post-operative care")
            cardio.patients.extend([p1, p2])
            cardio.staff.append(Staff("Dr. Sarah Miller", 42, "Cardiologist", "Cardiology"))
            cardio.staff.append(Staff("Emma Wilson", 28, "Head Nurse", "Cardiology"))
        except Exception:
            pass

    window = HospitalWindow(hospital)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()