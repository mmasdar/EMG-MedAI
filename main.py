import sys
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QListWidget, 
                             QFrame, QSplitter, QSizePolicy, QLineEdit, QGroupBox, QGridLayout, QFileDialog, QMessageBox)
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
        super(MplCanvas, self).__init__(self.fig)

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

            data = []
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
            
            # Create DataFrame
            MAX_COLS = len(data_lines[0])
            # Columns: Index, Tr 1, Tr 2, ... 
            # We assume column 0 is index/time? Or just data?
            # Based on previous CSVs, col 0 is often just an index 0,1,2... or time?
            # User snippet for Motor NCS used parts[1] and parts[2].
            # Meaning Column 0 was ignored.
            # We will follow that pattern: Column 0 is index, Traces start at Column 1.
            
            cols = ["Index"] + [f"Tr {k}" for k in range(1, MAX_COLS)]
            
            df = pd.DataFrame(data_lines, columns=cols)
            # Convert to float
            for c in cols:
                # Replace commas
                df[c] = df[c].astype(str).str.replace(',', '.')
                # Coerce to numeric
                df[c] = pd.to_numeric(df[c], errors='coerce')
                
            df = df.dropna()
            
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

            data = []
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

class NeuroDiagMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.motor_backend = MotorNCSBackend()
        self.fwave_backend = FWaveBackend()
        self.sensory_backend = SensoryBackend()
        
        self.setWindowTitle("NeuroDiag v0.1-alpha")
        self.setGeometry(100, 100, 1280, 720)
        self.current_labels = ["Site 1", "Site 2"] 
        self.active_mode = "MOTOR_NCS" # MOTOR_NCS, F_WAVE, SENSORY_NCS

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
        
        # Connect signals
        self.menu_list.currentRowChanged.connect(self.on_menu_change)

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
        
        if mode == "MOTOR_NCS":
            self.title_label.setText("Motor Nerve Conduction")
            self.right_frame.show() 
            self.site1_group.show()
            self.site2_group.show()
            self.ncv_label.setText("NCV (m/s):")
        elif mode == "F_WAVE":
            self.title_label.setText("F-Wave Analysis")
            self.right_frame.hide()
        elif mode == "SENSORY_NCS":
            self.title_label.setText("Sensory Nerve Conduction")
            self.right_frame.show()
            self.site1_group.setTitle("Sensory Parameters")
            self.site1_group.show()
            self.site2_group.hide()
            
            # Repurpose site1 inputs
            # Latency (Onset)
            # Amp (Peak-to-Peak)
            self.site1_group.layout().itemAtPosition(0, 0).widget().setText("Onset Lat (ms):")
            self.site1_group.layout().itemAtPosition(1, 0).widget().setText("Amp (uV):")
            self.ncv_label.setText("SNCV (m/s):")

    def create_sidebar(self):
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setFixedWidth(250)
        self.sidebar_frame.setStyleSheet("background-color: #f8f9fa; border-right: 1px solid #dee2e6;")
        layout = QVBoxLayout(self.sidebar_frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # User Profile
        user_layout = QHBoxLayout()
        icon_label = QLabel("👤") 
        icon_label.setStyleSheet("font-size: 24px; color: #0d6efd; border: 1px solid #dee2e6; border-radius: 20px; padding: 5px;")
        
        user_info = QVBoxLayout()
        self.name_label = QLabel("Pasien: -")
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.id_label = QLabel("ID: -")
        self.id_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        user_info.addWidget(self.name_label)
        user_info.addWidget(self.id_label)
        
        user_layout.addWidget(icon_label)
        user_layout.addLayout(user_info)
        layout.addLayout(user_layout)
        
        edit_link = QLabel("Edit Data Pasien")
        edit_link.setStyleSheet("color: #0d6efd; font-size: 12px; text-decoration: none;")
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
        self.menu_list.addItem("🌊 F-Wave Analysis") # Moved up or reordered? User said "tab F-wave"
        self.menu_list.addItem("👆 Sensory NCS")
        self.menu_list.setCurrentRow(0)
        self.menu_list.setStyleSheet("""
            QListWidget { border: none; background: transparent; font-size: 13px; }
            QListWidget::item { padding: 8px; border-radius: 4px; }
            QListWidget::item:selected { background-color: #e7f1ff; color: #0d6efd; font-weight: bold; }
            QListWidget::item:hover { background-color: #e9ecef; }
        """)
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
        btn_import = QPushButton("📂 Import CSV")
        btn_import.setStyleSheet("""
            QPushButton {
                background-color: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 4px; margin-right: 10px;
            }
            QPushButton:hover { background-color: #5a6268; }
        """)
        btn_import.clicked.connect(self.import_data)

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
        
        # Initial empty plot
        self.canvas.axes.text(0.5, 0.5, "Import CSV to View Data", ha='center')
        self.canvas.draw()

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
        layout.setSpacing(15)
        
        layout.addWidget(QLabel("Parameter Pengukuran"))

        # Site 1 Box
        self.site1_group = QGroupBox("Site 1: Ankle (Distal)")
        site1_layout = QGridLayout()
        site1_layout.addWidget(QLabel("Latency (ms):"), 0, 0)
        self.s1_lat = QLineEdit("0.0")
        site1_layout.addWidget(self.s1_lat, 0, 1)
        site1_layout.addWidget(QLabel("Amp (mV):"), 1, 0)
        self.s1_amp = QLineEdit("0.0")
        site1_layout.addWidget(self.s1_amp, 1, 1)
        self.site1_group.setLayout(site1_layout)
        layout.addWidget(self.site1_group)

        # Site 2 Box
        self.site2_group = QGroupBox("Site 2: Knee (Proximal)")
        self.site2_group.setStyleSheet("QGroupBox { color: #dc3545; font-weight: bold; }")
        site2_layout = QGridLayout()
        site2_layout.addWidget(QLabel("Latency (ms):"), 0, 0)
        self.s2_lat = QLineEdit("0.0")
        site2_layout.addWidget(self.s2_lat, 0, 1)
        site2_layout.addWidget(QLabel("Amp (mV):"), 1, 0)
        self.s2_amp = QLineEdit("0.0")
        site2_layout.addWidget(self.s2_amp, 1, 1)
        self.site2_group.setLayout(site2_layout)
        layout.addWidget(self.site2_group)

        # Calculation Box
        calc_group = QGroupBox("Conduction Velocity")
        calc_layout = QGridLayout()
        calc_layout.addWidget(QLabel("Jarak (mm):"), 0, 0)
        self.dist_input = QLineEdit("80") 
        calc_layout.addWidget(self.dist_input, 0, 1)
        
        ncv_label = QLabel("NCV (m/s):")
        ncv_label.setStyleSheet("font-weight: bold;")
        self.ncv_label = ncv_label # Store reference
        self.ncv_val = QLabel("0.0")
        self.ncv_val.setStyleSheet("font-weight: bold; color: #0d6efd; font-size: 16px;")
        
        calc_layout.addWidget(ncv_label, 1, 0)
        calc_layout.addWidget(self.ncv_val, 1, 1)
        calc_group.setLayout(calc_layout)
        layout.addWidget(calc_group)

        layout.addStretch()

    def import_data(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not filename:
             return
        
        if self.active_mode == "MOTOR_NCS":
             self.process_motor_ncs(filename)
        elif self.active_mode == "F_WAVE":
             self.process_f_wave(filename)
        elif self.active_mode == "SENSORY_NCS":
             self.process_sensory_ncs(filename)

    def process_motor_ncs(self, filename):
        df, metadata = self.motor_backend.load_data(filename)
        if df is None:
             QMessageBox.critical(self, "Error", "Failed to load Motor NCS data")
             return
        
        self.update_patient_info(metadata)
        
        # Labels update
        if metadata and "labels" in metadata and len(metadata["labels"]) >= 2:
            self.current_labels = metadata["labels"]
            self.site1_group.setTitle(f"Site 1: {self.current_labels[0]}")
            self.site2_group.setTitle(f"Site 2: {self.current_labels[1]}")

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
        # Update Inputs
        self.s1_lat.setText(f"{res['p1_t']:.2f}")
        self.s1_amp.setText(f"{res['amp1_mV']:.2f}")
        self.s2_lat.setText(f"{res['p2_t']:.2f}")
        self.s2_amp.setText(f"{res['amp2_mV']:.2f}")
        self.ncv_val.setText(f"{res['velocity']:.1f}")

        # Plot
        self.canvas.axes.clear()
        
        # Grid settings matching 5 ms/Div and 5 mV/Div
        self.canvas.axes.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        self.canvas.axes.spines['top'].set_visible(False)
        self.canvas.axes.spines['right'].set_visible(False)
        self.canvas.axes.spines['left'].set_visible(False) 
        self.canvas.axes.spines['bottom'].set_visible(False)

        trace1_mV = res['trace1'] / 1000.0
        trace2_mV = res['trace2'] / 1000.0
        offset = 10.0 
        trace2_shifted = trace2_mV - offset

        label1 = self.current_labels[0]
        label2 = self.current_labels[1]

        self.canvas.axes.plot(res['time'], trace1_mV, label=label1, color='#000000', linewidth=1.2)
        self.canvas.axes.plot(res['time'], trace2_shifted, label=label2, color='#000000', linewidth=1.2)
        
        self.canvas.axes.text(res['time'][-1], trace1_mV[-1], f" {label1}", verticalalignment='center')
        self.canvas.axes.text(res['time'][-1], trace2_shifted[-1], f" {label2}", verticalalignment='center')

        if res['p1_t'] is not None:
             p1_val_mV = res['p1_val'] / 1000.0
             self.canvas.axes.plot(res['p1_t'], p1_val_mV, 'o', color='red', markersize=6)
             self.canvas.axes.text(res['p1_t'], p1_val_mV + 0.5, "P", ha='center', color='red', fontweight='bold')
             
        if res['p2_t'] is not None:
             p2_val_mV = res['p2_val'] / 1000.0 - offset
             self.canvas.axes.plot(res['p2_t'], p2_val_mV, 'o', color='red', markersize=6)
             self.canvas.axes.text(res['p2_t'], p2_val_mV + 0.5, "P", ha='center', color='red', fontweight='bold')

        # Baseline markers (Green)
        z_idx = res['zero_idx']
        t_zero = res['time'][z_idx]
        b1_mV = res['b1_val'] / 1000.0
        b2_mV = res['b2_val'] / 1000.0 - offset

        self.canvas.axes.plot(t_zero, b1_mV, 's', color='green', markersize=5)
        self.canvas.axes.plot(t_zero, b2_mV, 's', color='green', markersize=5)

        # Setup Axes Limits
        self.canvas.axes.set_xlim(0, 50)
        self.canvas.axes.set_xticks(np.arange(0, 51, 5)) 
        self.canvas.axes.set_ylim(-25, 15)
        self.canvas.axes.set_yticks(np.arange(-25, 16, 5)) 
        self.canvas.axes.set_xticklabels([])
        self.canvas.axes.set_yticklabels([])
        self.canvas.axes.set_xlabel("")
        self.canvas.axes.set_ylabel("")
        
        self.canvas.draw()
        self.footer_info_r.setText("Sensitivity: 2 mV/Div (Approx)")

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
        self.canvas.axes.grid(True, which='major', axis='x', linestyle=':', color='lightgray', linewidth=0.5)
        # Vertical grid every 5 ms
        self.canvas.axes.set_xticks(np.arange(0, time[-1] + 5, 5))
        
        self.canvas.axes.spines['top'].set_visible(False)
        self.canvas.axes.spines['right'].set_visible(False)
        self.canvas.axes.spines['left'].set_visible(False) 
        self.canvas.axes.spines['bottom'].set_visible(True)
        self.canvas.axes.spines['bottom'].set_color('gray')

        # Plotting parameters
        vertical_offset = 0
        spacing = 5.0 # Division spacing
        
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
            self.canvas.axes.plot([cut_time, cut_time], [vertical_offset - 0.2, vertical_offset + 0.2], 
                                  color='gray', linewidth=1, alpha=0.8)
            
            # Trace Label at the end
            self.canvas.axes.text(time[-1] + 1, vertical_offset, f"Tr {i+1}", 
                                  va='center', ha='left', fontsize=9, color=color, fontweight='bold')

            vertical_offset += spacing # Stack upwards (increasing Y)
            
        # Draw main vertical separator at cut_time
        self.canvas.axes.axvline(x=cut_time, color='gray', linestyle='-', linewidth=1.5, alpha=0.4)

        # Scale annotations at the bottom
        # Clear Y ticks
        self.canvas.axes.set_yticks([])
        
        # Calculate annotation Y position (below the first trace)
        ann_y = -3.0
        self.canvas.axes.text(0, ann_y, "5 mV/Div", va='top', ha='left', fontsize=10, fontweight='bold')
        self.canvas.axes.text(cut_time + 2, ann_y, "500 \u03bcV/Div", va='top', ha='left', fontsize=10, fontweight='bold')
        self.canvas.axes.text(time[-1], ann_y, "5 ms/Div", va='top', ha='right', fontsize=10, color='gray')

        # Axis Limits
        self.canvas.axes.set_xlim(0, time[-1] + 8)
        self.canvas.axes.set_ylim(ann_y - 2, vertical_offset)
        
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
        self.canvas.axes.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        self.canvas.axes.spines['top'].set_visible(False)
        self.canvas.axes.spines['right'].set_visible(False)
        self.canvas.axes.spines['left'].set_visible(False) 
        self.canvas.axes.spines['bottom'].set_visible(False)

        time = res['time']
        signal = res['signal']

        self.canvas.axes.plot(time, signal, color='#000000', linewidth=1.2)
        
        # Peak & Trough markers
        self.canvas.axes.plot(time[res['p_idx']], signal[res['p_idx']], 'o', color='red', markersize=6)
        self.canvas.axes.plot(time[res['n_idx']], signal[res['n_idx']], 'o', color='red', markersize=6)
        
        # Onset marker (Green)
        self.canvas.axes.plot(time[res['onset_idx']], signal[res['onset_idx']], 's', color='green', markersize=5)

        # Labels
        self.canvas.axes.text(time[res['p_idx']], signal[res['p_idx']] + 5, "P", ha='center', color='red', fontweight='bold')
        self.canvas.axes.text(time[res['n_idx']], signal[res['n_idx']] - 5, "T", ha='center', verticalalignment='top', color='red', fontweight='bold')

        # Setup Axes Limits - Dynamic for Sensory
        self.canvas.axes.set_xlim(0, time[-1])
        # Centralized around signal
        y_max = max(abs(np.max(signal)), abs(np.min(signal))) + 20
        self.canvas.axes.set_ylim(-y_max, y_max)
        
        self.canvas.axes.set_xlabel("Time (ms)")
        self.canvas.axes.set_ylabel("Amplitude (uV)")
        
        self.canvas.draw()
        self.footer_info_r.setText("Sensitivity: 20 uV/Div (Approx)")

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

    window = NeuroDiagMainWindow()
    window.show()
    sys.exit(app.exec_())