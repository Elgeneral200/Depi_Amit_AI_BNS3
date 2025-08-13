import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QGroupBox, QFormLayout, QLineEdit,
    QSpinBox, QPlainTextEdit, QPushButton, QMessageBox, QTabWidget, QCheckBox, QInputDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from hospital import Hospital
from department import Department
from patient import Patient
from staff import Staff


class HospitalWindow(QMainWindow):
    def __init__(self, hospital: Hospital):
        super().__init__()
        self.hospital = hospital
        self.setWindowTitle(f"إدارة المستشفى - {self.hospital.name}")
        self.setLayoutDirection(Qt.RightToLeft)
        self.setMinimumSize(1100, 700)

        # خط مناسب للعربي (اختياري)
        try:
            self.setFont(QFont("Cairo", 11))
        except Exception:
            pass

        self.page = DepartmentsPage(self.hospital)
        self.setCentralWidget(self.page)


class DepartmentsPage(QWidget):
    def __init__(self, hospital: Hospital):
        super().__init__()
        self.hospital = hospital
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left panel: Departments list + add department
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        lbl_depts = QLabel("الأقسام")
        lbl_depts.setStyleSheet("font-weight: bold;")
        self.dept_list = QListWidget()
        self.dept_list.currentItemChanged.connect(self.on_dept_changed)

        add_group = QGroupBox("إضافة قسم")
        add_form = QFormLayout(add_group)
        self.dept_name_in = QLineEdit()
        self.dept_cap_in = QSpinBox()
        self.dept_cap_in.setRange(10, 1000)
        self.dept_cap_in.setValue(50)
        btn_add_dept = QPushButton("إضافة")
        btn_add_dept.clicked.connect(self.handle_add_department)
        add_form.addRow("الاسم", self.dept_name_in)
        add_form.addRow("السعة", self.dept_cap_in)
        add_form.addRow(btn_add_dept)

        left_layout.addWidget(lbl_depts)
        left_layout.addWidget(self.dept_list, 1)
        left_layout.addWidget(add_group, 0)

        # Right panel: Department info + tabs (Patients/Staff)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Department Info
        self.info_group = QGroupBox("معلومات القسم")
        info_layout = QFormLayout(self.info_group)
        self.lbl_dept_name = QLabel("-")
        self.lbl_dept_code = QLabel("-")
        self.lbl_dept_capacity = QLabel("-")
        self.lbl_patients_count = QLabel("-")
        self.lbl_staff_count = QLabel("-")
        info_layout.addRow("الاسم:", self.lbl_dept_name)
        info_layout.addRow("الكود:", self.lbl_dept_code)
        info_layout.addRow("السعة:", self.lbl_dept_capacity)
        info_layout.addRow("عدد المرضى:", self.lbl_patients_count)
        info_layout.addRow("عدد الطاقم:", self.lbl_staff_count)

        # Tabs
        self.tabs = QTabWidget()
        self._build_patients_tab()
        self._build_staff_tab()

        right_layout.addWidget(self.info_group)
        right_layout.addWidget(self.tabs, 1)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 800])

        self.refresh_department_list()

    # ---------- Patients tab ----------
    def _build_patients_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Add patient form
        add_group = QGroupBox("إضافة مريض")
        form = QFormLayout(add_group)
        self.p_name_in = QLineEdit()
        self.p_age_in = QSpinBox()
        self.p_age_in.setRange(1, 120)
        self.p_med_rec_in = QPlainTextEdit()
        self.p_med_rec_in.setPlaceholderText("الملاحظات/السجل الطبي")
        btn_add_patient = QPushButton("إضافة مريض")
        btn_add_patient.clicked.connect(self.handle_add_patient)
        form.addRow("الاسم", self.p_name_in)
        form.addRow("السن", self.p_age_in)
        form.addRow("السجل الطبي", self.p_med_rec_in)
        form.addRow(btn_add_patient)

        # Patients list
        list_group = QGroupBox("قائمة المرضى")
        list_layout = QVBoxLayout(list_group)
        self.only_active_chk = QCheckBox("عرض غير المخروجين فقط")
        self.only_active_chk.setChecked(True)
        self.only_active_chk.stateChanged.connect(self.refresh_patients_list)

        self.patients_list = QListWidget()
        btns_row = QHBoxLayout()
        self.btn_discharge = QPushButton("خروج المريض المحدد")
        self.btn_discharge.clicked.connect(self.handle_discharge_patient)
        btn_refresh = QPushButton("تحديث")
        btn_refresh.clicked.connect(self.refresh_patients_list)
        btns_row.addWidget(self.btn_discharge)
        btns_row.addWidget(btn_refresh)
        btns_row.addStretch()

        list_layout.addWidget(self.only_active_chk)
        list_layout.addWidget(self.patients_list, 1)
        list_layout.addLayout(btns_row)

        layout.addWidget(add_group)
        layout.addWidget(list_group, 1)

        self.tabs.addTab(tab, "المرضى")

    # ---------- Staff tab ----------
    def _build_staff_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        add_group = QGroupBox("إضافة موظف")
        form = QFormLayout(add_group)
        self.s_name_in = QLineEdit()
        self.s_age_in = QSpinBox()
        self.s_age_in.setRange(18, 100)
        self.s_position_in = QLineEdit()
        self.s_position_in.setPlaceholderText("مثال: Cardiologist / Nurse / Technician")
        btn_add_staff = QPushButton("إضافة موظف")
        btn_add_staff.clicked.connect(self.handle_add_staff)
        form.addRow("الاسم", self.s_name_in)
        form.addRow("السن", self.s_age_in)
        form.addRow("المنصب", self.s_position_in)
        form.addRow(btn_add_staff)

        list_group = QGroupBox("قائمة الطاقم")
        list_layout = QVBoxLayout(list_group)
        self.position_filter_in = QLineEdit()
        self.position_filter_in.setPlaceholderText("فلتر بالمنصب (اختياري)")
        self.position_filter_in.textChanged.connect(self.refresh_staff_list)

        self.staff_list = QListWidget()
        btns_row = QHBoxLayout()
        self.btn_toggle_active = QPushButton("تبديل حالة التفعيل")
        self.btn_toggle_active.clicked.connect(self.handle_toggle_staff)
        btn_refresh = QPushButton("تحديث")
        btn_refresh.clicked.connect(self.refresh_staff_list)
        btns_row.addWidget(self.btn_toggle_active)
        btns_row.addWidget(btn_refresh)
        btns_row.addStretch()

        list_layout.addWidget(self.position_filter_in)
        list_layout.addWidget(self.staff_list, 1)
        list_layout.addLayout(btns_row)

        layout.addWidget(add_group)
        layout.addWidget(list_group, 1)

        self.tabs.addTab(tab, "الطاقم")

    # ---------- Helpers ----------
    def refresh_department_list(self):
        self.dept_list.clear()
        for name, dept in self.hospital.departments.items():
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, dept)
            self.dept_list.addItem(item)

        # Select first dept if any
        if self.dept_list.count() > 0 and self.dept_list.currentRow() == -1:
            self.dept_list.setCurrentRow(0)

    def on_dept_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        self.update_dept_info()
        self.refresh_patients_list()
        self.refresh_staff_list()

    def current_department(self) -> Department | None:
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
        self.lbl_patients_count.setText(f"{len(dept.patients)} (غير المخروجين: {len(dept.get_active_patients())})")
        self.lbl_staff_count.setText(str(len(dept.staff)))

    def refresh_patients_list(self):
        self.patients_list.clear()
        dept = self.current_department()
        if not dept:
            return
        patients = dept.get_active_patients() if self.only_active_chk.isChecked() else dept.patients
        for p in patients:
            status = "مقيم" if not p.is_discharged else "مخروج"
            text = f"{p.patient_id} | {p.name} | {p.age} سنة | {status}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, p)
            self.patients_list.addItem(item)
        self.update_dept_info()

    def refresh_staff_list(self):
        self.staff_list.clear()
        dept = self.current_department()
        if not dept:
            return
        pos_filter = self.position_filter_in.text().strip()
        staff_list = dept.get_staff_by_position(pos_filter) if pos_filter else dept.staff
        for s in staff_list:
            status = "نشط" if s.is_active else "موقّف"
            text = f"{s.staff_id} | {s.name} | {s.position} | {status}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, s)
            self.staff_list.addItem(item)
        self.update_dept_info()

    # ---------- Handlers: Departments ----------
    def handle_add_department(self):
        name = self.dept_name_in.text().strip()
        cap = int(self.dept_cap_in.value())
        if not name:
            QMessageBox.warning(self, "تنبيه", "اسم القسم مطلوب")
            return
        try:
            self.hospital.add_department(Department(name, cap))
            self.dept_name_in.clear()
            self.dept_cap_in.setValue(50)
            self.refresh_department_list()
            QMessageBox.information(self, "تم", f"تمت إضافة القسم '{name}' بنجاح")
        except ValueError as e:
            QMessageBox.critical(self, "خطأ", str(e))

    # ---------- Handlers: Patients ----------
    def handle_add_patient(self):
        dept = self.current_department()
        if not dept:
            QMessageBox.warning(self, "تنبيه", "اختار قسم أولًا")
            return
        name = self.p_name_in.text().strip()
        age = int(self.p_age_in.value())
        med = self.p_med_rec_in.toPlainText().strip()

        if not name:
            QMessageBox.warning(self, "تنبيه", "اسم المريض مطلوب")
            return
        try:
            patient = Patient(name, age, med)
            ok = dept.add_patient(patient)
            if not ok:
                QMessageBox.warning(self, "السعة ممتلئة", f"لا يمكن إدخال {name}، القسم ممتلئ")
                return
            # clear inputs
            self.p_name_in.clear()
            self.p_age_in.setValue(1)
            self.p_med_rec_in.clear()
            self.refresh_patients_list()
            QMessageBox.information(self, "تم", f"تم إدخال المريض: {patient.patient_id}")
        except Exception as e:
            QMessageBox.critical(self, "خطأ", str(e))

    def handle_discharge_patient(self):
        item = self.patients_list.currentItem()
        if not item:
            QMessageBox.warning(self, "تنبيه", "اختار مريض من القائمة")
            return
        patient: Patient = item.data(Qt.UserRole)
        if patient.is_discharged:
            QMessageBox.information(self, "تنبيه", "المريض مخروج بالفعل")
            return
        notes, ok = QInputDialog.getMultiLineText(self, "ملاحظات الخروج", "أدخل ملاحظات الخروج (اختياري):", "")
        if not ok:
            return
        try:
            patient.discharge(notes)
            self.refresh_patients_list()
            QMessageBox.information(self, "تم", f"تم خروج المريض {patient.name}")
        except Exception as e:
            QMessageBox.critical(self, "خطأ", str(e))

    # ---------- Handlers: Staff ----------
    def handle_add_staff(self):
        dept = self.current_department()
        if not dept:
            QMessageBox.warning(self, "تنبيه", "اختار قسم أولًا")
            return
        name = self.s_name_in.text().strip()
        age = int(self.s_age_in.value())
        position = self.s_position_in.text().strip()
        if not name or not position:
            QMessageBox.warning(self, "تنبيه", "الاسم والمنصب مطلوبان")
            return
        try:
            staff = Staff(name, age, position, dept.name)
            dept.add_staff(staff)
            # clear
            self.s_name_in.clear()
            self.s_age_in.setValue(18)
            self.s_position_in.clear()
            self.refresh_staff_list()
            QMessageBox.information(self, "تم", f"تمت إضافة الموظف: {staff.staff_id}")
        except Exception as e:
            QMessageBox.critical(self, "خطأ", str(e))

    def handle_toggle_staff(self):
        item = self.staff_list.currentItem()
        if not item:
            QMessageBox.warning(self, "تنبيه", "اختار موظف من القائمة")
            return
        s: Staff = item.data(Qt.UserRole)
        try:
            s.toggle_active_status()
            self.refresh_staff_list()
        except Exception as e:
            QMessageBox.critical(self, "خطأ", str(e))


def main():
    app = QApplication(sys.argv)
    # مستشفى جديدة (الأقسام الافتراضية بتتضاف تلقائيًا من Hospital._initialize_default_departments)
    hospital = Hospital("City General Hospital", "123 Main St")

    # مثال صغير للتجربة: أضف بيانات
    cardio = hospital.find_department("Cardiology")
    if cardio:
        try:
            cardio.add_patient(Patient("Alice Johnson", 35, "Hypertension monitoring"))
            cardio.add_staff(Staff("Dr. Sarah Miller", 42, "Cardiologist", "Cardiology"))
        except Exception:
            pass

    window = HospitalWindow(hospital)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()