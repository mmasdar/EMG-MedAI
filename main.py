import sys
import numpy as np
import pandas as pd
import hashlib
from scipy.signal import find_peaks
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QListWidget, 
                             QFrame, QSplitter, QSizePolicy, QLineEdit, QGroupBox, QGridLayout, 
                             QFileDialog, QMessageBox, QDialog, QSlider)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # Remove top and right spines/borders for a cleaner look
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
        super(MplCanvas, self).__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

class MotorNCSBackend:
    def __init__(self):
        self.dt = 0.0781 # ms sample interval
        self.dist_m = 0.08 # fixed distance 80mm for now

    def load_data(self, filepath):
        try:
            with open(filepath, encoding="latin-1") as f:
                lines = f.readlines()
            
            metadata = {
                "name": "-",
                "id": "-",
                "test_item": "Motor NCS",
                "labels": ["Site 1", "Site 2"]
            }
            
            start = None
            data = []
            for i, line in enumerate(lines):
                parts = line.strip().split(";")
                if len(parts) >= 2:
                    key = parts[0].strip()
                    # Remove quotes and whitespace
                    values = [p.strip().replace('"', '') for p in parts[1:] if p.strip()]
                    
                    if key == "Patient Name" and values:
                        metadata["name"] = values[0]
                    elif key == "Patient ID" and values:
                        metadata["id"] = values[0]
                    elif key == "Trace Label" and len(values) >= 2:
                        metadata["labels"] = values[:2]

                if line.startswith("Trace Data"):
                    # Check if there is data on the same line
                    if len(parts) >= 3:
                        try:
                            # Assuming format: index; ch1; ch2
                            a = float(parts[1].replace(",", "."))
                            b = float(parts[2].replace(",", "."))
                            data.append([a, b])
                        except ValueError:
                            pass
                    start = i + 1
                    break
            
            if start is None:
                raise ValueError("Could not find 'Trace Data' section in file.")

            for line in lines[start:]:
                parts = line.strip().split(";")
                if len(parts) < 3:
                    break
                try:
                    # Assuming format: index; ch1; ch2
                    a = float(parts[1].replace(",", "."))
                    b = float(parts[2].replace(",", "."))
                    data.append([a, b])
                except ValueError:
                    continue

            df = pd.DataFrame(data, columns=["Ch1", "Ch2"])
            return df, metadata
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def auto_invert(self, signal):
        max_idx = np.argmax(np.abs(signal))
        if signal[max_idx] < 0:
            return -signal
        return signal

    def analyze(self, df):
        N = len(df)
        time = np.array([i * self.dt for i in range(N)])
        
        # Cut initial artifact
        start_index = 10
        if N <= start_index:
            return None

        # Slices
        time_cut = time[start_index:]
        # Use simple positional indexing instead of names
        trace1_raw = df.iloc[start_index:, 0].values
        trace2_raw = df.iloc[start_index:, 1].values

        # Auto-invert
        trace1 = self.auto_invert(trace1_raw)
        trace2 = self.auto_invert(trace2_raw)

        # Peak detection
        peaks1, _ = find_peaks(trace1, distance=50)
        peaks2, _ = find_peaks(trace2, distance=50)

        if len(peaks1) == 0 or len(peaks2) == 0:
            return {
                "time": time_cut,
                "trace1": trace1,
                "trace2": trace2,
                "p1_idx": None,
                "p2_idx": None,
                "latency": 0,
                "amp1": 0,
                "amp2": 0,
                "velocity": 0
            }

        # Taking the highest peak for simplicity (as per notebook logic)
        p1_local_idx = peaks1[np.argmax(trace1[peaks1])]
        p2_local_idx = peaks2[np.argmax(trace2[peaks2])]

        # Calculate metrics
        # Time values
        t1 = time_cut[p1_local_idx]
        t2 = time_cut[p2_local_idx]
        
        latency_ms = abs(t2 - t1)
        latency_sec = latency_ms / 1000.0 if latency_ms != 0 else 1e-9

        # Baseline at t = 0 (using time_cut if 0 is within range, else first available)
        zero_idx = np.argmin(np.abs(time_cut - 0))
        b1_val = trace1[zero_idx]
        b2_val = trace2[zero_idx]

        # Amplitudes (uV -> mV ?) Note notebook says / 1e3. 
        # Usually data is in uV, user wants mV.
        amp1_mV = trace1[p1_local_idx] / 1000.0
        amp2_mV = trace2[p2_local_idx] / 1000.0

        velocity = self.dist_m / latency_sec if latency_sec > 0 else 0

        return {
            "time": time_cut,
            "trace1": trace1,
            "trace2": trace2,
            "p1_t": t1, # Time at peak 1
            "p1_val": trace1[p1_local_idx],
            "p2_t": t2, # Time at peak 2
            "p2_val": trace2[p2_local_idx],
            "zero_idx": zero_idx,
            "b1_val": b1_val,
            "b2_val": b2_val,
            "latency_ms": latency_ms,
            "amp1_mV": amp1_mV,
            "amp2_mV": amp2_mV,
            "velocity": velocity
        }

class FWaveBackend:
    def __init__(self):
        self.dt = 0.0781 # ms
        self.manual_cut_time = 20.0 # ms

    def load_data(self, filepath):
        try:
            with open(filepath, encoding="latin-1") as f:
                lines = f.readlines()
            
            metadata = {
                "name": "-", 
                "id": "-", 
                "test_item": "F-Wave",
                "ms_per_sample": 0.0781
            }
            start = None
            data_lines = []
            
            for i, line in enumerate(lines):
                parts = line.strip().split(";")
                if len(parts) >= 2:
                    key = parts[0].strip()
                    val = parts[1].strip().replace('"', '')
                    if key == "Patient Name" and val: metadata["name"] = val
                    elif key == "Patient ID" and val: metadata["id"] = val
                    elif key == "Test Item" and val: metadata["test_item"] = val
                    elif key == "ms/Sample" and val: 
                        try:
                             metadata["ms_per_sample"] = float(val.replace(',', '.'))
                        except: pass

                if line.startswith("Trace Data"):
                    # Include the first sample if present
                    if len(parts) > 1:
                        data_lines.append(parts)
                    start = i + 1
                    break
            
            if start is None:
                return None, None

            # Read remaining lines
            data_lines.extend([l.strip().split(';') for l in lines[start:] if l.strip()])
            if not data_lines:
                return None, None
            
            # Create List of Numeric Data
            processed_data = []
            for row in data_lines:
                # Column 0 is usually Label/Empty, data starts from Column 1
                try:
                    # Clean and convert all elements from index 1 onwards
                    numeric_row = []
                    for val in row[1:]:
                        clean_val = val.strip().replace('"', '').replace(',', '.')
                        if not clean_val:
                            numeric_row.append(0.0) # Or skip? Let's use 0.0 or try to parse
                        else:
                            numeric_row.append(float(clean_val))
                    
                    if numeric_row:
                        processed_data.append(numeric_row)
                except (ValueError, IndexError):
                    continue
            
            if not processed_data:
                return None, None

            # Create DataFrame
            df = pd.DataFrame(processed_data)
            df.columns = [f"Tr {k+1}" for k in range(df.shape[1])]
            
            # Update dt if available
            self.dt = metadata.get("ms_per_sample", 0.0781)
            
            return df, metadata

        except Exception as e:
            print(f"Error loading F-Wave: {e}")
            return None, None

    def analyze(self, df):
        # We prepare data for plotting: split gain logic
        # Traces are columns Tr 1, Tr 2...
        # We assume Traces start from column 1 (ignoring Index at col 0)
        
        N = len(df)
        time = np.arange(N) * self.dt 
        
        processed_traces = []
        # Filter for "Tr X" columns
        trace_cols = [c for c in df.columns if c.startswith("Tr ")]
        
        # If no explicit "Tr " columns, take from index 1
        if not trace_cols and len(df.columns) > 1:
            trace_cols = df.columns[1:]
            
        for col in trace_cols[:10]: # Limit to 10
            # Data from CSV
            signal = df[col].values
            
            # User Logic: tr_current = tr_data[:N] * -1
            # Unconditional inversion
            signal_inverted = signal * -1
            
            processed_traces.append(signal_inverted)

        return {
            "time": time,
            "traces": processed_traces,
            "cut_time": self.manual_cut_time
        }

class SensoryBackend:
    def __init__(self):
        self.dt = 0.0313  # ms (default for sensory)
        self.dist_m = 0.14 # default distance 14cm for sural

    def load_data(self, filepath):
        try:
            with open(filepath, encoding="latin-1") as f:
                lines = f.readlines()
            
            metadata = {
                "name": "-", 
                "id": "-", 
                "test_item": "Sensory NCS",
                "ms_per_sample": 0.0313
            }
            start = None
            data = []
            for i, line in enumerate(lines):
                parts = line.strip().split(";")
                if len(parts) >= 2:
                    key = parts[0].strip()
                    val = parts[1].strip().replace('"', '')
                    if key == "Patient Name" and val: metadata["name"] = val
                    elif key == "Patient ID" and val: metadata["id"] = val
                    elif key == "Test Item" and val: metadata["test_item"] = val
                    elif key == "ms/Sample" and val: 
                        try:
                             metadata["ms_per_sample"] = float(val.replace(',', '.'))
                        except: pass

                if line.startswith("Trace Data"):
                    # Check if there is data on the same line
                    if len(parts) >= 2:
                        try:
                            # Based on format: ;Value (Column 1)
                            val = float(parts[1].replace(",", "."))
                            data.append(val)
                        except (ValueError, IndexError):
                            pass
                    start = i + 1
                    break
            
            if start is None:
                return None, None

            for line in lines[start:]:
                parts = line.strip().split(";")
                if len(parts) < 2:
                    continue
                try:
                    # Based on format: ;Value (Column 1)
                    val = float(parts[1].replace(",", "."))
                    data.append(val)
                except (ValueError, IndexError):
                    continue

            df = pd.DataFrame(data, columns=["Signal"])
            self.dt = metadata.get("ms_per_sample", 0.0313)
            return df, metadata
        except Exception as e:
            print(f"Error loading Sensory: {e}")
            return None, None

    def analyze(self, df):
        if df is None or df.empty:
            return None
            
        signal = df["Signal"].values
        N = len(signal)
        time = np.arange(N) * self.dt
        
        # Simple Peak detection
        p_idx = np.argmax(signal)
        n_idx = np.argmin(signal)
        
        peak_amp = signal[p_idx]
        trough_amp = signal[n_idx]
        p2p_amp = abs(peak_amp - trough_amp)
        
        # Onset detection (rough approximation: first point > 5% of peak)
        threshold = peak_amp * 0.1
        onset_idx = 0
        for i, v in enumerate(signal):
            if v > threshold:
                onset_idx = i
                break
        
        onset_lat = time[onset_idx]
        peak_lat = time[p_idx]
        
        # Velocity
        latency_sec = onset_lat / 1000.0 if onset_lat > 0 else 1e-9
        velocity = self.dist_m / latency_sec if latency_sec > 0 else 0
        
        return {
            "time": time,
            "signal": signal,
            "p_idx": p_idx,
            "n_idx": n_idx,
            "onset_idx": onset_idx,
            "onset_lat": onset_lat,
            "peak_lat": peak_lat,
            "p2p_amp": p2p_amp,
            "velocity": velocity
        }

class AccessLockDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NeuroDiag Access Lock")
        self.setFixedSize(400, 220)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout(self)
        
        icon_label = QLabel("🔐")
        icon_label.setFont(QFont("Segoe UI Emoji", 40))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        self.info_label = QLabel("Masukkan Kunci Akses untuk menjalankan aplikasi:")
        self.info_label.setStyleSheet("font-weight: bold; color: #212529; font-size: 14px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.Password)
        self.key_input.setPlaceholderText("Kunci Akses...")
        self.key_input.setStyleSheet("padding: 12px; font-size: 14px; border: 2px solid #dee2e6; border-radius: 6px;")
        layout.addWidget(self.key_input)
        
        btn_layout = QHBoxLayout()
        self.btn_login = QPushButton("Buka Akses")
        self.btn_login.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd; color: white; padding: 10px; font-weight: bold; border-radius: 5px;
            }
            QPushButton:hover { background-color: #0b5ed7; }
        """)
        self.btn_login.clicked.connect(self.verify_key)
        
        self.btn_exit = QPushButton("Keluar")
        self.btn_exit.setStyleSheet("""
            QPushButton {
                background-color: #6c757d; color: white; padding: 10px; border-radius: 5px;
            }
            QPushButton:hover { background-color: #5a6268; }
        """)
        self.btn_exit.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.btn_exit)
        btn_layout.addWidget(self.btn_login)
        layout.addLayout(btn_layout)
        
        self.master_hash = "92e43ee76839fad67687b3c863e5f9dfe1c6d6d1d9cfee2c5509502ca60db200"

    def verify_key(self):
        key = self.key_input.text().strip()
        if not key:
            return
            
        input_hash = hashlib.sha256(key.encode()).hexdigest()
        if input_hash == self.master_hash:
            self.accept()
        else:
            QMessageBox.critical(self, "Akses Ditolak", "Kunci akses salah! Silakan hubungi pengembang.")
            self.key_input.clear()

class NeuroDiagMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.motor_backend = MotorNCSBackend()
        self.fwave_backend = FWaveBackend()
        self.sensory_backend = SensoryBackend()
        
        self.setWindowTitle("NeuroDiag v0.1-alpha")
        self.showMaximized()
        self.current_labels = ["Site 1", "Site 2"] 
        self.active_mode = "MOTOR_NCS" # MOTOR_NCS, F_WAVE, SENSORY_NCS
        self.last_fwave_df = None
        
        # File storage for folder import
        self.patient_folder = None
        self.motor_files = []
        self.fwave_files = []
        self.sensory_files = []

        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Left Sidebar
        self.create_sidebar()
        main_layout.addWidget(self.sidebar_frame)

        # 2. Center Content (Graph)
        self.create_center_content()
        main_layout.addWidget(self.center_frame, stretch=2)

        # 3. Right Panel (Parameters)
        self.create_right_panel()
        main_layout.addWidget(self.right_frame)

        self.apply_styles()

    def on_menu_change(self, row):
        item = self.menu_list.item(row)
        if "Motor NCS" in item.text():
            self.switch_view("MOTOR_NCS")
        elif "F-Wave" in item.text():
            self.switch_view("F_WAVE")
        elif "Sensory NCS" in item.text():
            self.switch_view("SENSORY_NCS")
            
    def switch_view(self, mode):
        self.active_mode = mode
        self.canvas.axes.clear()
        self.canvas.draw()
        
        # Update file list in right panel based on mode
        self.update_file_list()
        
        if mode == "MOTOR_NCS":
            self.title_label.setText("Motor Nerve Conduction")
            self.right_frame.show() 
            self.motor_summary_group.show()
            self.cv_group.show()
            self.fwave_group.hide()
            self.site1_label.setText("Distal")
            self.site2_label.setText("Proximal")
        elif mode == "F_WAVE":
            self.title_label.setText("F-Wave Analysis")
            self.right_frame.show()
            self.motor_summary_group.hide()
            self.cv_group.hide()
            self.fwave_group.show()
        elif mode == "SENSORY_NCS":
            self.title_label.setText("Sensory Nerve Conduction")
            self.right_frame.show()
            self.motor_summary_group.show()
            self.cv_group.show()
            self.fwave_group.hide()
            self.site1_label.setText("Sensory")
            self.site2_label.setText("-")


    def update_file_list(self):
        """Update the file list in right panel based on current mode"""
        self.data_file_list.clear()
        import os
        
        if self.active_mode == "MOTOR_NCS":
            for f in self.motor_files:
                self.data_file_list.addItem(f"⚡ {os.path.basename(f)}")
        elif self.active_mode == "F_WAVE":
            for f in self.fwave_files:
                self.data_file_list.addItem(f"🌊 {os.path.basename(f)}")
        elif self.active_mode == "SENSORY_NCS":
            for f in self.sensory_files:
                self.data_file_list.addItem(f"👆 {os.path.basename(f)}")

    def on_file_selected(self, item):
        """Handle file selection from right panel list"""
        idx = self.data_file_list.row(item)
        
        if self.active_mode == "MOTOR_NCS" and 0 <= idx < len(self.motor_files):
            self.process_motor_ncs(self.motor_files[idx])
        elif self.active_mode == "F_WAVE" and 0 <= idx < len(self.fwave_files):
            self.process_f_wave(self.fwave_files[idx])
        elif self.active_mode == "SENSORY_NCS" and 0 <= idx < len(self.sensory_files):
            self.process_sensory_ncs(self.sensory_files[idx])

    def create_sidebar(self):
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setFixedWidth(250)
        self.sidebar_frame.setStyleSheet("background-color: #f8f9fa; border-right: 1px solid #dee2e6;")
        layout = QVBoxLayout(self.sidebar_frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # User Profile
        user_box = QFrame()
        user_box.setStyleSheet("background-color: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 10px;")
        user_box_layout = QVBoxLayout(user_box)
        user_box_layout.setContentsMargins(10, 10, 10, 10)
        user_box_layout.setSpacing(5)
        
        self.name_label = QLabel("Pasien: -")
        self.name_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #212529;")
        self.name_label.setWordWrap(True)
        user_box_layout.addWidget(self.name_label)
        
        self.id_label = QLabel("ID: -")
        self.id_label.setStyleSheet("color: #6c757d; font-size: 10px;")
        self.id_label.setWordWrap(True)
        user_box_layout.addWidget(self.id_label)
        
        layout.addWidget(user_box)
        
        edit_link = QLabel("✏️ Edit Data Pasien")
        edit_link.setStyleSheet("color: #0d6efd; font-size: 11px;")
        edit_link.setCursor(Qt.PointingHandCursor)
        layout.addWidget(edit_link)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #dee2e6;")
        layout.addWidget(line)

        # Review Section Header
        header_ncs = QLabel("PEMERIKSAAN NCS")
        header_ncs.setStyleSheet("color: #6c757d; font-size: 11px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(header_ncs)

        # Menu Items
        self.menu_list = QListWidget()
        self.menu_list.addItem("⚡ Motor NCS")
        self.menu_list.addItem("🌊 F-Wave Analysis")
        self.menu_list.addItem("👆 Sensory NCS")
        self.menu_list.setCurrentRow(0)
        self.menu_list.setMaximumHeight(110)
        self.menu_list.setStyleSheet("""
            QListWidget { border: none; background: transparent; font-size: 13px; }
            QListWidget::item { padding: 8px; border-radius: 4px; }
            QListWidget::item:selected { background-color: #e7f1ff; color: #0d6efd; font-weight: bold; }
            QListWidget::item:hover { background-color: #e9ecef; }
        """)
        self.menu_list.currentRowChanged.connect(self.on_menu_change)
        layout.addWidget(self.menu_list)

        # Results Section
        header_res = QLabel("HASIL")
        header_res.setStyleSheet("color: #6c757d; font-size: 11px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(header_res)
        
        self.res_list = QListWidget()
        self.res_list.addItem("➕ Diagnosis Otomatis")
        self.res_list.setStyleSheet("""
             QListWidget { border: none; background: transparent; font-size: 13px; }
             QListWidget::item { padding: 8px; border-radius: 4px; }
             QListWidget::item:hover { background-color: #e9ecef; }
        """)
        layout.addWidget(self.res_list)

        layout.addStretch()

    def create_center_content(self):
        self.center_frame = QFrame()
        self.center_frame.setStyleSheet("background-color: white;")
        layout = QVBoxLayout(self.center_frame)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header_layout = QHBoxLayout()
        self.title_label = QLabel("Motor Nerve Conduction")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #212529;")
        
        # Action Buttons
        btn_import = QPushButton("📂 Open Folder")
        btn_import.setStyleSheet("""
            QPushButton {
                background-color: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 4px; margin-right: 10px;
            }
            QPushButton:hover { background-color: #5a6268; }
        """)
        btn_import.clicked.connect(self.open_patient_folder)

        btn_stimulate = QPushButton("▶ Jalankan Stimulasi")
        btn_stimulate.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #0b5ed7; }
        """)
        
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(btn_import)
        header_layout.addWidget(btn_stimulate)
        layout.addLayout(header_layout)

        # Graph
        self.canvas = MplCanvas(self, width=5, height=6, dpi=100) # Taller for F-Wave stacks
        layout.addWidget(self.canvas)
        
        # F-Wave Slider (Under the graph, aligned with x-axis)
        self.fwave_group = QFrame()
        self.fwave_group.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; margin-top: 5px;")
        fwave_layout = QVBoxLayout(self.fwave_group)
        fwave_layout.setContentsMargins(0, 8, 0, 8)
        
        # Label row
        label_row = QHBoxLayout()
        label_row.setContentsMargins(15, 0, 15, 0)
        lbl_info = QLabel("🌊 Atur Batas Gain (F-Wave):")
        lbl_info.setStyleSheet("font-weight: bold; color: #495057;")
        label_row.addWidget(lbl_info)
        label_row.addStretch()
        self.fwave_cut_label = QLabel("20 ms")
        self.fwave_cut_label.setStyleSheet("font-weight: bold; color: #0d6efd; font-size: 16px; background-color: #e7f1ff; padding: 4px 12px; border-radius: 4px;")
        label_row.addWidget(self.fwave_cut_label)
        fwave_layout.addLayout(label_row)
        
        # Slider row with min/max labels
        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(15, 5, 15, 5)
        
        # Min label
        min_label = QLabel("0 ms")
        min_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        slider_row.addWidget(min_label)
        
        self.fwave_cut_slider = QSlider(Qt.Horizontal)
        self.fwave_cut_slider.setMinimum(5)
        self.fwave_cut_slider.setMaximum(80)
        self.fwave_cut_slider.setValue(20)
        self.fwave_cut_slider.setTickPosition(QSlider.TicksBelow)
        self.fwave_cut_slider.setTickInterval(10)
        self.fwave_cut_slider.setStyleSheet("""
            QSlider::groove:horizontal { 
                background: #dee2e6; height: 8px; border-radius: 4px; 
            }
            QSlider::handle:horizontal { 
                background: #0d6efd; width: 20px; height: 20px; 
                margin: -6px 0; border-radius: 10px;
                border: 2px solid white;
            }
            QSlider::handle:horizontal:hover {
                background: #0b5ed7;
            }
            QSlider::sub-page:horizontal { background: #0d6efd; border-radius: 4px; }
            QSlider::tick-mark:horizontal { background: #adb5bd; width: 1px; height: 5px; }
        """)
        self.fwave_cut_slider.valueChanged.connect(self.on_fwave_cut_change)
        slider_row.addWidget(self.fwave_cut_slider, stretch=1)
        
        # Max label
        max_label = QLabel("80 ms")
        max_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        slider_row.addWidget(max_label)
        
        fwave_layout.addLayout(slider_row)
        
        layout.addWidget(self.fwave_group)
        self.fwave_group.hide()

        # Footer info - Dynamic maybe?
        self.footer_layout = QHBoxLayout()
        self.footer_info_l = QLabel("Time Base: 5 ms/Div")
        self.footer_info_r = QLabel("Sensitivity: 2 mV/Div") # Dynamic
        self.footer_layout.addWidget(self.footer_info_l)
        self.footer_layout.addStretch()
        self.footer_layout.addWidget(self.footer_info_r)
        layout.addLayout(self.footer_layout)

    def create_right_panel(self):
        self.right_frame = QFrame()
        self.right_frame.setFixedWidth(300)
        self.right_frame.setStyleSheet("background-color: #f8f9fa; border-left: 1px solid #dee2e6;")
        layout = QVBoxLayout(self.right_frame)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(10)
        
        # File Selection Section (at top)
        file_header = QLabel("📁 Pilih File Data")
        file_header.setStyleSheet("font-weight: bold; font-size: 13px; color: #212529;")
        layout.addWidget(file_header)
        
        self.data_file_list = QListWidget()
        self.data_file_list.setMaximumHeight(180)
        self.data_file_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.data_file_list.setStyleSheet("""
            QListWidget { border: 1px solid #dee2e6; background: white; font-size: 12px; border-radius: 4px; }
            QListWidget::item { padding: 6px 10px; }
            QListWidget::item:selected { background-color: #e7f1ff; color: #0d6efd; font-weight: bold; }
            QListWidget::item:hover { background-color: #f8f9fa; }
        """)
        self.data_file_list.itemClicked.connect(self.on_file_selected)
        layout.addWidget(self.data_file_list)
        
        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #dee2e6;")
        layout.addWidget(line)
        
        # Parameter Section Header
        param_header = QLabel("📊 MOTOR SUMMARY TABLE")
        param_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #6c757d; margin-top: 5px;")
        layout.addWidget(param_header)

        # Motor Summary Table Container
        self.motor_summary_group = QFrame()
        self.motor_summary_group.setStyleSheet("background-color: white; border: 1px solid #dee2e6; border-radius: 8px;")
        table_layout = QGridLayout(self.motor_summary_group)
        table_layout.setContentsMargins(10, 10, 10, 10)
        table_layout.setSpacing(8)
        
        # Table Headers
        header_style = "font-weight: bold; font-size: 10px; color: #495057;"
        table_layout.addWidget(self._create_header("Stim Site", header_style), 0, 0)
        table_layout.addWidget(self._create_header("Lat (ms)", header_style), 0, 1)
        table_layout.addWidget(self._create_header("Amp (mV)", header_style), 0, 2)
        
        # Divider line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #dee2e6;")
        table_layout.addWidget(line, 1, 0, 1, 3)
        
        # Site 1 Row (Distal)
        self.site1_label = QLabel("Distal")
        self.site1_label.setStyleSheet("font-size: 12px; color: #212529;")
        table_layout.addWidget(self.site1_label, 2, 0)
        
        self.s1_lat = QLabel("0.00")
        self.s1_lat.setStyleSheet("font-size: 12px; font-weight: bold; color: #212529;")
        table_layout.addWidget(self.s1_lat, 2, 1)
        
        self.s1_amp = QLabel("0.00")
        self.s1_amp.setStyleSheet("font-size: 12px; font-weight: bold; color: #212529;")
        table_layout.addWidget(self.s1_amp, 2, 2)
        
        # Site 2 Row (Proximal)
        self.site2_label = QLabel("Proximal")
        self.site2_label.setStyleSheet("font-size: 12px; color: #212529;")
        table_layout.addWidget(self.site2_label, 3, 0)
        
        self.s2_lat = QLabel("0.00")
        self.s2_lat.setStyleSheet("font-size: 12px; font-weight: bold; color: #212529;")
        table_layout.addWidget(self.s2_lat, 3, 1)
        
        self.s2_amp = QLabel("0.00")
        self.s2_amp.setStyleSheet("font-size: 12px; font-weight: bold; color: #212529;")
        table_layout.addWidget(self.s2_amp, 3, 2)
        
        layout.addWidget(self.motor_summary_group)
        
        # Keep references for hiding (for other modes)
        self.site1_group = self.motor_summary_group
        self.site2_group = QFrame()  # Dummy for compatibility
        self.site2_group.hide()

        # Conduction Velocity Analysis Section - Clean Modern Design
        self.cv_group = QFrame()
        self.cv_group.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2196F3, stop:1 #42A5F5);
                border-radius: 12px; margin-top: 10px;
            }
        """)
        cv_main_layout = QVBoxLayout(self.cv_group)
        cv_main_layout.setContentsMargins(16, 14, 16, 14)
        cv_main_layout.setSpacing(12)
        
        # Header with icon
        header_row = QHBoxLayout()
        header_icon = QLabel("⚡")
        header_icon.setStyleSheet("font-size: 14px; background: transparent;")
        header_row.addWidget(header_icon)
        cv_header = QLabel("VELOCITY ANALYSIS")
        cv_header.setStyleSheet("font-weight: bold; font-size: 12px; color: white; background: transparent; letter-spacing: 1px;")
        header_row.addWidget(cv_header)
        header_row.addStretch()
        cv_main_layout.addLayout(header_row)
        
        # Values Grid - Delta and Dist side by side
        values_row = QHBoxLayout()
        values_row.setSpacing(15)
        
        # Delta box
        delta_box = QVBoxLayout()
        delta_box.setSpacing(4)
        delta_label = QLabel("Delta (ms)")
        delta_label.setStyleSheet("font-size: 10px; color: rgba(255,255,255,0.7); background: transparent;")
        delta_box.addWidget(delta_label)
        
        delta_val_row = QHBoxLayout()
        delta_val_row.setSpacing(0)
        delta_open = QLabel("(")
        delta_open.setStyleSheet("font-size: 24px; color: rgba(255,255,255,0.5); background: transparent;")
        delta_val_row.addWidget(delta_open)
        self.delta_val = QLabel("0.00")
        self.delta_val.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background: transparent;")
        delta_val_row.addWidget(self.delta_val)
        delta_val_row.addStretch()
        delta_box.addLayout(delta_val_row)
        values_row.addLayout(delta_box)
        
        # Dist box
        dist_box = QVBoxLayout()
        dist_box.setSpacing(4)
        dist_label = QLabel("Dist (mm)")
        dist_label.setStyleSheet("font-size: 10px; color: rgba(255,255,255,0.7); background: transparent;")
        dist_box.addWidget(dist_label)
        
        dist_val_row = QHBoxLayout()
        dist_val_row.setSpacing(0)
        dist_open = QLabel("(")
        dist_open.setStyleSheet("font-size: 24px; color: rgba(255,255,255,0.5); background: transparent;")
        dist_val_row.addWidget(dist_open)
        self.dist_input = QLabel("0")
        self.dist_input.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background: transparent;")
        dist_val_row.addWidget(self.dist_input)
        dist_val_row.addStretch()
        dist_box.addLayout(dist_val_row)
        values_row.addLayout(dist_box)
        
        values_row.addStretch()
        cv_main_layout.addLayout(values_row)
        
        # Separator
        separator = QFrame()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: rgba(255,255,255,0.2);")
        cv_main_layout.addWidget(separator)
        
        # Result Section
        result_col = QVBoxLayout()
        result_col.setSpacing(4)
        result_label = QLabel("Velocity Result")
        result_label.setStyleSheet("font-size: 10px; color: rgba(255,255,255,0.7); background: transparent;")
        result_col.addWidget(result_label)
        
        # Main velocity value with status
        result_row = QHBoxLayout()
        result_row.setSpacing(8)
        
        vel_val_row = QHBoxLayout()
        vel_val_row.setSpacing(0)
        vel_open = QLabel("(")
        vel_open.setStyleSheet("font-size: 32px; color: rgba(255,255,255,0.5); background: transparent;")
        vel_val_row.addWidget(vel_open)
        self.ncv_val = QLabel("0.0")
        self.ncv_val.setStyleSheet("font-size: 32px; font-weight: bold; color: white; background: transparent;")
        vel_val_row.addWidget(self.ncv_val)
        
        vel_unit = QLabel("M/S")
        vel_unit.setStyleSheet("font-size: 14px; font-weight: bold; color: rgba(255,255,255,0.8); background: transparent; margin-left: 4px;")
        vel_val_row.addWidget(vel_unit, alignment=Qt.AlignBottom)
        result_row.addLayout(vel_val_row)
        
        result_row.addStretch()
        
        # Status badge
        self.ncv_status = QLabel("NORMAL")
        self.ncv_status.setStyleSheet("""
            font-size: 10px; font-weight: bold; color: white; 
            background-color: #4CAF50;
            padding: 6px 12px; border-radius: 4px;
        """)
        result_row.addWidget(self.ncv_status, alignment=Qt.AlignVCenter)
        
        result_col.addLayout(result_row)
        cv_main_layout.addLayout(result_col)
        
        layout.addWidget(self.cv_group)
        
        # ncv_label reference for compatibility
        self.ncv_label = QLabel("NCV (m/s):")


        layout.addStretch()

    def _create_header(self, text, style):
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        lbl.setAlignment(Qt.AlignCenter)
        return lbl

    def on_fwave_cut_change(self, value):
        self.fwave_backend.manual_cut_time = float(value)
        self.fwave_cut_label.setText(f"{value:.1f} ms")
        if self.last_fwave_df is not None:
            result = self.fwave_backend.analyze(self.last_fwave_df)
            if result:
                self.update_ui_fwave(result)

    def open_patient_folder(self):
        import os
        folder = QFileDialog.getExistingDirectory(self, "Select Patient Folder")
        if not folder:
            return
        
        self.patient_folder = folder
        patient_name = os.path.basename(folder)
        self.name_label.setText(f"Pasien: {patient_name}")
        
        # Clear previous data
        self.motor_files = []
        self.fwave_files = []
        self.sensory_files = []
        
        # Scan folder structure
        fwave_folder = os.path.join(folder, "f wave")
        ncv_folder = os.path.join(folder, "ncv")
        
        # F-Wave files
        if os.path.exists(fwave_folder):
            for f in sorted(os.listdir(fwave_folder)):
                if f.endswith('.csv'):
                    filepath = os.path.join(fwave_folder, f)
                    self.fwave_files.append(filepath)
        
        # NCV files (1-8 motor, 9-14 sensory)
        if os.path.exists(ncv_folder):
            for f in sorted(os.listdir(ncv_folder)):
                if f.endswith('.csv'):
                    filepath = os.path.join(ncv_folder, f)
                    # Extract number from filename
                    try:
                        num = int(os.path.splitext(f)[0])
                        if 1 <= num <= 8:
                            self.motor_files.append(filepath)
                        elif 9 <= num <= 14:
                            self.sensory_files.append(filepath)
                    except ValueError:
                        pass
        
        # Update file list in right panel
        self.update_file_list()
        
        # Auto-load first file if available in current mode
        if self.active_mode == "MOTOR_NCS" and self.motor_files:
            self.process_motor_ncs(self.motor_files[0])
            self.data_file_list.setCurrentRow(0)
        elif self.active_mode == "F_WAVE" and self.fwave_files:
            self.process_f_wave(self.fwave_files[0])
            self.data_file_list.setCurrentRow(0)
        elif self.active_mode == "SENSORY_NCS" and self.sensory_files:
            self.process_sensory_ncs(self.sensory_files[0])
            self.data_file_list.setCurrentRow(0)

    def process_motor_ncs(self, filename):
        df, metadata = self.motor_backend.load_data(filename)
        if df is None:
             QMessageBox.critical(self, "Error", "Failed to load Motor NCS data")
             return
        
        self.update_patient_info(metadata)
        
        # Labels update
        if metadata and "labels" in metadata and len(metadata["labels"]) >= 2:
            self.current_labels = metadata["labels"]
            self.site1_label.setText(self.current_labels[0])
            self.site2_label.setText(self.current_labels[1])

        result = self.motor_backend.analyze(df)
        if result is None:
             QMessageBox.warning(self, "Warning", "Could not analyze Motor NCS data")
             return
        
        self.update_ui_motor(result)

    def process_f_wave(self, filename):
        df, metadata = self.fwave_backend.load_data(filename)
        if df is None:
             QMessageBox.critical(self, "Error", "Failed to load F-Wave data")
             return
             
        self.update_patient_info(metadata)
        self.last_fwave_df = df # Store for slider interaction
        
        result = self.fwave_backend.analyze(df)
        if result is None:
              QMessageBox.warning(self, "Warning", "Could not analyze F-Wave data")
              return
        
        self.update_ui_fwave(result)

    def process_sensory_ncs(self, filename):
        df, metadata = self.sensory_backend.load_data(filename)
        if df is None:
             QMessageBox.critical(self, "Error", "Failed to load Sensory NCS data")
             return
             
        self.update_patient_info(metadata)
        
        result = self.sensory_backend.analyze(df)
        if result is None:
              QMessageBox.warning(self, "Warning", "Could not analyze Sensory NCS data")
              return
        
        self.update_ui_sensory(result)

    def update_patient_info(self, metadata):
        if metadata:
            if "name" in metadata:
                self.name_label.setText(f"Pasien: {metadata['name']}")
            if "id" in metadata:
                self.id_label.setText(f"ID: {metadata['id']}")
            if "test_item" in metadata:
                self.title_label.setText(metadata['test_item'])


    def update_ui_motor(self, res):
        # Update Motor Summary Table
        self.s1_lat.setText(f"{res['p1_t']:.2f}")
        self.s1_amp.setText(f"{res['amp1_mV']:.2f}")
        self.s2_lat.setText(f"{res['p2_t']:.2f}")
        self.s2_amp.setText(f"{res['amp2_mV']:.2f}")
        
        # Update Conduction Velocity Analysis
        delta = res['p2_t'] - res['p1_t']
        self.delta_val.setText(f"{delta:.2f}")
        self.dist_input.setText(f"{res.get('distance_mm', 250)}")
        self.ncv_val.setText(f"{res['velocity']:.1f}")
        
        # Update status badge with solid colors
        if res['velocity'] >= 50:
            self.ncv_status.setText("NORMAL")
            self.ncv_status.setStyleSheet("font-size: 10px; font-weight: bold; color: white; background-color: #4CAF50; padding: 6px 12px; border-radius: 4px;")
        else:
            self.ncv_status.setText("ABNORMAL")
            self.ncv_status.setStyleSheet("font-size: 10px; font-weight: bold; color: white; background-color: #E53935; padding: 6px 12px; border-radius: 4px;")


        # Plot
        self.canvas.axes.clear()
        self.canvas.axes.set_facecolor('white')
        
        # Grid settings matching 5 ms/Div and 1 Div/Vertical
        # Horizontal grid every 5 ms
        self.canvas.axes.grid(True, which='major', axis='both', linestyle=':', color='lightgray', linewidth=0.5)
        
        # Plotting parameters
        # Notebook logic: wave / 2.0 (2mV/div)
        # Scale: 1 unit on Y axis = 1 Division
        mv_per_div = 2.0
        
        trace1_div = (res['trace1'] / 1000.0) / mv_per_div
        trace2_div = (res['trace2'] / 1000.0) / mv_per_div
        
        # Separation: Site 1 at Y=0, Site 2 at Y=-5 Divisions
        offset_div = 5.0 
        trace1_plot = trace1_div
        trace2_plot = trace2_div - offset_div

        label1 = self.current_labels[0]
        label2 = self.current_labels[1]

        # Professional black waveforms
        self.canvas.axes.plot(res['time'], trace1_plot, label=label1, color='#212529', linewidth=1.2)
        self.canvas.axes.plot(res['time'], trace2_plot, label=label2, color='#212529', linewidth=1.2)
        
        # Trace labels at the end
        self.canvas.axes.text(res['time'][-1] + 1, 0, f" {label1}", va='center', fontsize=9, fontweight='bold')
        self.canvas.axes.text(res['time'][-1] + 1, -offset_div, f" {label2}", va='center', fontsize=9, fontweight='bold')

        # Peak Markers
        if res['p1_t'] is not None:
             p1_div = (res['p1_val'] / 1000.0) / mv_per_div
             self.canvas.axes.plot(res['p1_t'], p1_div, 'o', color='red', markersize=6)
             self.canvas.axes.text(res['p1_t'], p1_div + 0.3, "P", ha='center', color='red', fontweight='bold', fontsize=8)
             
        if res['p2_t'] is not None:
             p2_div = ((res['p2_val'] / 1000.0) / mv_per_div) - offset_div
             self.canvas.axes.plot(res['p2_t'], p2_div, 'o', color='red', markersize=6)
             self.canvas.axes.text(res['p2_t'], p2_div + 0.3, "P", ha='center', color='red', fontweight='bold', fontsize=8)

        # Baseline markers (Green squares)
        z_idx = res['zero_idx']
        t_zero = res['time'][z_idx]
        b1_div = (res['b1_val'] / 1000.0) / mv_per_div
        b2_div = ((res['b2_val'] / 1000.0) / mv_per_div) - offset_div

        self.canvas.axes.plot(t_zero, b1_div, 's', color='green', markersize=5)
        self.canvas.axes.plot(t_zero, b2_div, 's', color='green', markersize=5)

        # Scale annotations
        # Setting Y ticks at every division to show grid lines, but hiding labels
        self.canvas.axes.set_yticks(np.arange(-offset_div - 2, 2, 1))
        self.canvas.axes.set_yticklabels([])
        
        ann_y = -offset_div - 2.5
        self.canvas.axes.text(0, ann_y, f"{int(mv_per_div)} mV/Div", va='top', ha='left', fontsize=10, fontweight='bold')
        self.canvas.axes.text(res['time'][-1], ann_y, "5 ms/Div", va='top', ha='right', fontsize=10, color='gray')

        # Setup Axes Limits
        self.canvas.axes.set_xlim(0, res['time'][-1] + 7)
        self.canvas.axes.set_xticks(np.arange(0, res['time'][-1] + 5, 5)) 
        self.canvas.axes.set_ylim(ann_y - 1, 3) # Room for both traces and annotations
        
        self.canvas.axes.spines['top'].set_visible(False)
        self.canvas.axes.spines['right'].set_visible(False)
        self.canvas.axes.spines['left'].set_visible(False) 
        self.canvas.axes.spines['bottom'].set_visible(True)
        self.canvas.axes.spines['bottom'].set_color('gray')
        
        self.canvas.axes.set_xlabel("Time (ms)")
        self.canvas.axes.set_ylabel("")
        
        self.canvas.draw()
        self.footer_info_r.setText(f"Sensitivity: {int(mv_per_div)} mV/Div")

    def update_ui_fwave(self, res):
        self.canvas.axes.clear()
        
        time = res['time']
        cut_time = res['cut_time']
        traces = res['traces']
        
        if len(time) == 0:
             return
             
        cut_idx = np.argmin(np.abs(time - cut_time))
        
        # Grid and Style setup
        self.canvas.axes.set_facecolor('white')
        self.canvas.axes.grid(True, which='major', axis='both', linestyle=':', color='lightgray', linewidth=0.5)
        
        # Horizontal grid every 5 ms
        self.canvas.axes.set_xticks(np.arange(0, time[-1] + 5, 5))
        
        self.canvas.axes.spines['top'].set_visible(False)
        self.canvas.axes.spines['right'].set_visible(False)
        self.canvas.axes.spines['left'].set_visible(False) 
        self.canvas.axes.spines['bottom'].set_visible(True)
        self.canvas.axes.spines['bottom'].set_color('gray')

        # Plotting parameters
        vertical_offset = 0
        spacing = 1.5 # Reduced from 5.0 to make signals look relatively larger
        
        # Professional colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, trace in enumerate(traces[:10]): # Show top 10 traces
            color = colors[i % len(colors)]
            
            # Normalize to mV (assuming input is uV)
            trace_mV = trace / 1000.0
            
            # Split
            t1 = time[:cut_idx+1]
            w1 = trace_mV[:cut_idx+1]
            t2 = time[cut_idx:]
            w2 = trace_mV[cut_idx:]
            
            # Apply Visual Gain (Normalization)
            # Notebook: wave1 / 5.0 (5mV/div), wave2 / 0.5 (500uV/div)
            w1_plot = w1 / 5.0
            w2_plot = w2 / 0.5
            
            # Notebook logic: y = wave_norm + vertical_offset
            # We add vertical_offset to the entire wave
            y1 = w1_plot + vertical_offset
            y2 = w2_plot + vertical_offset
            
            # Plot segments
            self.canvas.axes.plot(t1, y1, color=color, linewidth=1.2)
            self.canvas.axes.plot(t2, y2, color=color, linewidth=1.2)
            
            # Small vertical tick at cut point for this trace
            # Adjusted range to match smaller spacing
            self.canvas.axes.plot([cut_time, cut_time], [vertical_offset - 0.1, vertical_offset + 0.1], 
                                  color='gray', linewidth=1, alpha=0.8)
            
            # Trace Label at the end
            self.canvas.axes.text(time[-1] + 1, vertical_offset, f"Tr {i+1}", 
                                  va='center', ha='left', fontsize=9, color=color, fontweight='bold')

            vertical_offset += spacing # Stack upwards (increasing Y)
            
        # Draw main vertical separator at cut_time
        self.canvas.axes.axvline(x=cut_time, color='gray', linestyle='-', linewidth=1.5, alpha=0.4)

        # Scale annotations at the bottom
        # Setting Y ticks at every division to show grid lines, but hiding labels
        self.canvas.axes.set_yticks(np.arange(-1, vertical_offset + 1, 1))
        self.canvas.axes.set_yticklabels([])
        
        # Calculate annotation Y position (below the first trace)
        ann_y = -1.0 # Adjusted from -3.0
        self.canvas.axes.text(0, ann_y, "5 mV/Div", va='top', ha='left', fontsize=10, fontweight='bold')
        self.canvas.axes.text(cut_time + 1, ann_y, "500 \u03bcV/Div", va='top', ha='left', fontsize=10, fontweight='bold')
        self.canvas.axes.text(time[-1], ann_y, "5 ms/Div", va='top', ha='right', fontsize=10, color='gray')

        # Axis Limits
        self.canvas.axes.set_xlim(0, time[-1] + 8)
        self.canvas.axes.set_ylim(ann_y - 0.5, vertical_offset)
        
        self.canvas.axes.set_xlabel("Time (ms)")
        self.canvas.axes.set_ylabel("")
        
        self.canvas.draw()
        self.footer_info_r.setText("Scale: Split Gain (Upward Stacked)")

    def update_ui_sensory(self, res):
        # Update Inputs
        self.s1_lat.setText(f"{res['onset_lat']:.2f}")
        self.s1_amp.setText(f"{res['p2p_amp']:.1f}")
        self.ncv_val.setText(f"{res['velocity']:.1f}")

        # Plot
        self.canvas.axes.clear()
        
        # Grid settings
        self.canvas.axes.set_facecolor('white')
        self.canvas.axes.grid(True, which='major', axis='both', linestyle=':', color='lightgray', linewidth=0.5)
        
        self.canvas.axes.spines['top'].set_visible(False)
        self.canvas.axes.spines['right'].set_visible(False)
        self.canvas.axes.spines['left'].set_visible(False) 
        self.canvas.axes.spines['bottom'].set_visible(True)
        self.canvas.axes.spines['bottom'].set_color('gray')

        time = res['time']
        signal = res['signal']
        
        # Scaling: 20 uV/Div
        uv_per_div = 20.0
        signal_div = signal / uv_per_div

        self.canvas.axes.plot(time, signal_div, color='#212529', linewidth=1.2)
        
        # Peak & Trough markers
        p_div = signal[res['p_idx']] / uv_per_div
        n_div = signal[res['n_idx']] / uv_per_div
        o_div = signal[res['onset_idx']] / uv_per_div
        
        self.canvas.axes.plot(time[res['p_idx']], p_div, 'o', color='red', markersize=6)
        self.canvas.axes.plot(time[res['n_idx']], n_div, 'o', color='red', markersize=6)
        
        # Onset marker (Green)
        self.canvas.axes.plot(time[res['onset_idx']], o_div, 's', color='green', markersize=5)

        # Labels
        self.canvas.axes.text(time[res['p_idx']], p_div + 0.3, "P", ha='center', color='red', fontweight='bold', fontsize=8)
        self.canvas.axes.text(time[res['n_idx']], n_div - 0.3, "T", ha='center', va='top', color='red', fontweight='bold', fontsize=8)

        # Determine range for sensory
        v_ext = max(abs(np.max(signal_div)), abs(np.min(signal_div)))
        v_ext = max(v_ext, 2.0) # at least 2 divisions
        ann_y = -v_ext - 1.5
        
        # Scale annotations
        # Setting Y ticks at every division to show grid lines, but hiding labels
        self.canvas.axes.set_yticks(np.arange(int(-v_ext - 1), int(v_ext + 2), 1))
        self.canvas.axes.set_yticklabels([])
        
        self.canvas.axes.text(0, ann_y, f"{int(uv_per_div)} \u03bcV/Div", va='top', ha='left', fontsize=10, fontweight='bold')
        self.canvas.axes.text(time[-1], ann_y, "5 ms/Div", va='top', ha='right', fontsize=10, color='gray')

        # Setup Axes Limits
        self.canvas.axes.set_xlim(0, time[-1] + 5)
        self.canvas.axes.set_xticks(np.arange(0, time[-1] + 5, 5))
        self.canvas.axes.set_ylim(ann_y - 0.5, v_ext + 1)
        
        self.canvas.axes.set_xlabel("Time (ms)")
        self.canvas.axes.set_ylabel("")
        
        self.canvas.draw()
        self.footer_info_r.setText(f"Sensitivity: {int(uv_per_div)} \u03bcV/Div")

    def apply_styles(self):
        # Global Styles
        self.setStyleSheet("""
            QMainWindow { background-color: #f8f9fa; }
            QLabel { color: #212529; }
            QLineEdit { 
                border: 1px solid #ced4da; 
                border-radius: 4px; 
                padding: 4px; 
                background: white; 
                color: #495057;
            }
            QGroupBox {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                margin-top: 20px;
                font-weight: bold;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set default font
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    # Show Access Lock (Temporarily disabled for development)
    # lock = AccessLockDialog()
    # if lock.exec_() == QDialog.Accepted:
    #     window = NeuroDiagMainWindow()
    #     window.showMaximized()
    #     sys.exit(app.exec_())
    # else:
    #     sys.exit(0)

    # Direct launch
    window = NeuroDiagMainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
