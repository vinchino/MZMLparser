import sys
import os
import pandas as pd
import h5py
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QLabel, QFileDialog, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QLineEdit, QComboBox, QMessageBox, QGridLayout, QHeaderView
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from scipy.signal import find_peaks

# Import the parse_mzml function from parse_mzml.py
from parse_mzml import parse_mzml


class SpectrumPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')  # Set white background
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel('left', 'Intensity')
        self.setLabel('bottom', 'm/z')

        # Variables to store current spectrum data
        self.current_mz = None
        self.current_intensity = None

        # Initialize plot items
        self.spectrum_curve = self.plot(pen=pg.mkPen('b', width=1))
        
        # Remove or comment out the ScatterPlotItem to eliminate red circles
        # self.peak_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen('r'), brush=pg.mkBrush('r'))
        # self.addItem(self.peak_scatter)

        # Text items for annotations
        self.peak_texts = []

        # Connect view range change signal for dynamic peak updates
        self.sigXRangeChanged.connect(self.update_peaks)

    def plot_spectrum(self, mz, intensity, metadata):
        """
        Plots the spectrum and annotates metadata.

        Args:
            mz (array-like): m/z values.
            intensity (array-like): Intensity values.
            metadata (dict): Metadata for annotation.
        """
        self.current_mz = mz
        self.current_intensity = intensity

        # Update spectrum curve
        self.spectrum_curve.setData(mz, intensity)

        # Update title with metadata
        title = f"Spectrum Index: {metadata['Index Number']}"
        self.setTitle(title, color='#000000', size='12pt')

        # Remove previous annotations
        self.clear_peak_annotations()

        # Initial peak detection and annotation
        self.update_peaks()

    def update_peaks(self):
        """
        Detects peaks within the current x-axis view and annotates the top 30 most intense peaks.
        """
        if self.current_mz is None or self.current_intensity is None:
            return

        # Get current view range
        view_range = self.viewRange()
        x_min, x_max = view_range[0]

        # Filter data within current x-axis limits
        mask = (self.current_mz >= x_min) & (self.current_mz <= x_max)
        mz_filtered = self.current_mz[mask]
        intensity_filtered = self.current_intensity[mask]

        if len(mz_filtered) == 0:
            # Clear scatter and annotations if no data in view
            # self.peak_scatter.setData([])  # Already removed ScatterPlotItem
            self.clear_peak_annotations()
            return

        # Find peaks using scipy's find_peaks
        peaks, properties = find_peaks(intensity_filtered, height=0)

        if len(peaks) == 0:
            # Clear scatter and annotations if no peaks found
            # self.peak_scatter.setData([])  # Already removed ScatterPlotItem
            self.clear_peak_annotations()
            return

        peak_mz = mz_filtered[peaks]
        peak_intensity = intensity_filtered[peaks]

        # Select top 30 most intense peaks
        if len(peak_intensity) > 30:
            top_indices = np.argsort(peak_intensity)[-30:]
            top_peaks_mz = peak_mz[top_indices]
            top_peaks_intensity = peak_intensity[top_indices]
        else:
            top_peaks_mz = peak_mz
            top_peaks_intensity = peak_intensity

        # Update scatter plot with peaks
        # self.peak_scatter.setData(spots)  # Already removed ScatterPlotItem

        # Add text annotations for each peak
        self.clear_peak_annotations()
        for mz, intensity in zip(top_peaks_mz, top_peaks_intensity):
            text = pg.TextItem(f"{mz:.2f}", anchor=(0.5, 1.0), color='red')
            text.setPos(mz, intensity)
            self.addItem(text)
            self.peak_texts.append(text)

    def clear_peak_annotations(self):
        """
        Removes existing peak annotations from the plot.
        """
        for text in self.peak_texts:
            self.removeItem(text)
        self.peak_texts = []


class FilterWidget(QWidget):
    def __init__(self, parent=None):
        super(FilterWidget, self).__init__(parent)
        layout = QGridLayout()

        # Index Number Filter
        layout.addWidget(QLabel("Index Number:"), 0, 0)
        self.index_operator = QComboBox()
        self.index_operator.addItems(["==", "Between"])
        layout.addWidget(self.index_operator, 0, 1)
        self.index_value1 = QLineEdit()
        self.index_value1.setPlaceholderText("Value")
        layout.addWidget(self.index_value1, 0, 2)
        self.index_value2 = QLineEdit()
        self.index_value2.setPlaceholderText("Value")
        self.index_value2.setEnabled(False)
        layout.addWidget(self.index_value2, 0, 3)

        self.index_operator.currentTextChanged.connect(self.toggle_index_fields)

        # Cycle Number Filter
        layout.addWidget(QLabel("Cycle Number:"), 1, 0)
        self.cycle_operator = QComboBox()
        self.cycle_operator.addItems(["==", "Between"])
        layout.addWidget(self.cycle_operator, 1, 1)
        self.cycle_value1 = QLineEdit()
        self.cycle_value1.setPlaceholderText("Value")
        layout.addWidget(self.cycle_value1, 1, 2)
        self.cycle_value2 = QLineEdit()
        self.cycle_value2.setPlaceholderText("Value")
        self.cycle_value2.setEnabled(False)
        layout.addWidget(self.cycle_value2, 1, 3)

        self.cycle_operator.currentTextChanged.connect(self.toggle_cycle_fields)

        # Scan Start Time (RT) Filter
        layout.addWidget(QLabel("Scan Start Time (RT):"), 2, 0)
        self.rt_operator = QComboBox()
        self.rt_operator.addItems(["==", "Between", "±"])
        layout.addWidget(self.rt_operator, 2, 1)
        self.rt_value1 = QLineEdit()
        self.rt_value1.setPlaceholderText("Value")
        layout.addWidget(self.rt_value1, 2, 2)
        self.rt_value2 = QLineEdit()
        self.rt_value2.setPlaceholderText("Delta")
        self.rt_value2.setEnabled(False)
        layout.addWidget(self.rt_value2, 2, 3)

        self.rt_operator.currentTextChanged.connect(self.toggle_rt_fields)

        # Selected Ion m/z Filter
        layout.addWidget(QLabel("Selected Ion m/z:"), 3, 0)
        self.mz_operator = QComboBox()
        self.mz_operator.addItems(["==", "Between", "± ppm"])
        layout.addWidget(self.mz_operator, 3, 1)
        self.mz_value1 = QLineEdit()
        self.mz_value1.setPlaceholderText("Value")
        layout.addWidget(self.mz_value1, 3, 2)
        self.mz_value2 = QLineEdit()
        self.mz_value2.setPlaceholderText("ppm")
        self.mz_value2.setEnabled(False)
        layout.addWidget(self.mz_value2, 3, 3)

        self.mz_operator.currentTextChanged.connect(self.toggle_mz_fields)

        # MS Level Filter
        layout.addWidget(QLabel("MS Level:"), 4, 0)
        self.ms_level_operator = QComboBox()
        self.ms_level_operator.addItems(["=="])
        layout.addWidget(self.ms_level_operator, 4, 1)
        self.ms_level_value = QLineEdit()
        self.ms_level_value.setPlaceholderText("Value")
        layout.addWidget(self.ms_level_value, 4, 2, 1, 2)

        # Apply Filter Button
        self.apply_button = QPushButton("Apply Filters")
        layout.addWidget(self.apply_button, 5, 0, 1, 4)

        self.setLayout(layout)

    def toggle_index_fields(self, text):
        if text == "Between":
            self.index_value2.setEnabled(True)
        else:
            self.index_value2.setEnabled(False)
            self.index_value2.clear()

    def toggle_cycle_fields(self, text):
        if text == "Between":
            self.cycle_value2.setEnabled(True)
        else:
            self.cycle_value2.setEnabled(False)
            self.cycle_value2.clear()

    def toggle_rt_fields(self, text):
        if text in ["Between", "±"]:
            self.rt_value2.setEnabled(True)
            if text == "±":
                self.rt_value2.setPlaceholderText("Delta")
            else:
                self.rt_value2.setPlaceholderText("Value")
        else:
            self.rt_value2.setEnabled(False)
            self.rt_value2.setPlaceholderText("")
            self.rt_value2.clear()

    def toggle_mz_fields(self, text):
        if text in ["Between", "± ppm"]:
            self.mz_value2.setEnabled(True)
            if text == "± ppm":
                self.mz_value2.setPlaceholderText("ppm")
            else:
                self.mz_value2.setPlaceholderText("Value")
        else:
            self.mz_value2.setEnabled(False)
            self.mz_value2.setPlaceholderText("")
            self.mz_value2.clear()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mzML Viewer")
        self.setGeometry(100, 100, 1600, 900)

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Splitter to divide left and middle panes
        splitter = QSplitter(Qt.Horizontal)

        # Left pane: List of mzML files
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.setDragDropMode(QAbstractItemView.DropOnly)
        splitter.addWidget(self.file_list)

        # Middle pane: Spectrum list with filters
        middle_widget = QWidget()
        middle_layout = QVBoxLayout()

        # Spectrum list table
        self.spectrum_table = QTableWidget()
        self.spectrum_table.setColumnCount(7)
        self.spectrum_table.setHorizontalHeaderLabels([
            "Index", "Cycle", "Experiment", "MS Level",
            "TIC", "Scan Start Time (RT)", "Selected Ion m/z"
        ])
        self.spectrum_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.spectrum_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.spectrum_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.spectrum_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        middle_layout.addWidget(self.spectrum_table)

        # Filters
        self.filter_widget = FilterWidget()
        middle_layout.addWidget(self.filter_widget)

        # Apply filter connection
        self.filter_widget.apply_button.clicked.connect(self.apply_filters)

        middle_widget.setLayout(middle_layout)
        splitter.addWidget(middle_widget)

        # Right pane: Plot area
        self.plot_widget = SpectrumPlotWidget()
        splitter.addWidget(self.plot_widget)

        splitter.setStretchFactor(0, 1)  # File list
        splitter.setStretchFactor(1, 2)  # Spectrum list and filters
        splitter.setStretchFactor(2, 3)  # Plot area

        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Connections
        self.file_list.itemClicked.connect(self.load_spectra)
        self.spectrum_table.itemSelectionChanged.connect(self.auto_plot_selected_spectrum)

        # Data storage
        self.parsed_files = {}  # key: filename, value: {'csv': path, 'h5': path, 'metadata': DataFrame}

        # Enable drag and drop
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            file_path = url.toLocalFile()
            if file_path.lower().endswith('.mzml'):
                self.add_mzml_file(file_path)

    def add_mzml_file(self, file_path):
        """
        Parses the mzML file and updates the file list.

        Args:
            file_path (str): Path to the mzML file.
        """
        # Parse the mzML file
        output_dir = "parsed_data"
        os.makedirs(output_dir, exist_ok=True)

        # Parse mzML and get output paths
        try:
            metadata_csv, data_h5 = parse_mzml(file_path, output_dir)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse '{file_path}': {e}")
            return

        # Load metadata into pandas DataFrame
        try:
            metadata_df = pd.read_csv(metadata_csv)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read metadata CSV for '{file_path}': {e}")
            return

        # Store in parsed_files
        filename = os.path.basename(file_path)
        self.parsed_files[filename] = {
            'csv': metadata_csv,
            'h5': data_h5,
            'metadata': metadata_df
        }

        # Add to file list if not already present
        items = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if filename not in items:
            self.file_list.addItem(filename)

        QMessageBox.information(self, "Success", f"Successfully parsed '{filename}'.")

    def load_spectra(self, item):
        """
        Loads spectra metadata into the table when a file is selected.

        Args:
            item (QListWidgetItem): The selected file item.
        """
        filename = item.text()
        if filename not in self.parsed_files:
            QMessageBox.warning(self, "Error", f"File '{filename}' not found in parsed data.")
            return

        metadata_df = self.parsed_files[filename]['metadata']

        # Populate the spectrum table
        self.spectrum_table.setRowCount(0)
        for _, row in metadata_df.iterrows():
            row_position = self.spectrum_table.rowCount()
            self.spectrum_table.insertRow(row_position)

            # Index
            if not pd.isnull(row['Index Number']):
                try:
                    index_number = int(row['Index Number'])
                    index_item = QTableWidgetItem(str(index_number))
                except ValueError:
                    index_item = QTableWidgetItem("")
            else:
                index_item = QTableWidgetItem("")
            self.spectrum_table.setItem(row_position, 0, index_item)

            # Cycle
            if not pd.isnull(row['Cycle Number']):
                try:
                    cycle_number = int(row['Cycle Number'])
                    cycle_item = QTableWidgetItem(str(cycle_number))
                except ValueError:
                    cycle_item = QTableWidgetItem("")
            else:
                cycle_item = QTableWidgetItem("")
            self.spectrum_table.setItem(row_position, 1, cycle_item)

            # Experiment
            if not pd.isnull(row['Experiment Number']):
                try:
                    experiment_number = int(row['Experiment Number'])
                    experiment_item = QTableWidgetItem(str(experiment_number))
                except ValueError:
                    experiment_item = QTableWidgetItem("")
            else:
                experiment_item = QTableWidgetItem("")
            self.spectrum_table.setItem(row_position, 2, experiment_item)

            # MS Level
            if not pd.isnull(row['MS Level']):
                try:
                    ms_level = int(row['MS Level'])
                    ms_level_item = QTableWidgetItem(str(ms_level))
                except ValueError:
                    ms_level_item = QTableWidgetItem("")
            else:
                ms_level_item = QTableWidgetItem("")
            self.spectrum_table.setItem(row_position, 3, ms_level_item)

            # TIC
            if not pd.isnull(row['Total Ion Current']):
                tic = f"{row['Total Ion Current']:.2f}"
                tic_item = QTableWidgetItem(tic)
            else:
                tic_item = QTableWidgetItem("")
            self.spectrum_table.setItem(row_position, 4, tic_item)

            # Scan Start Time (RT)
            if not pd.isnull(row['Scan Start Time (min)']):
                rt = f"{row['Scan Start Time (min)']:.4f}"
                rt_item = QTableWidgetItem(rt)
            else:
                rt_item = QTableWidgetItem("")
            self.spectrum_table.setItem(row_position, 5, rt_item)

            # Selected Ion m/z
            if pd.notnull(row['Selected Ion m/z']):
                selected_ion_mz = f"{row['Selected Ion m/z']:.4f}"
            else:
                selected_ion_mz = "Full MS"
            selected_ion_mz_item = QTableWidgetItem(selected_ion_mz)
            self.spectrum_table.setItem(row_position, 6, selected_ion_mz_item)

        self.spectrum_table.resizeColumnsToContents()

    def auto_plot_selected_spectrum(self):
        """
        Automatically plots the currently selected spectrum in the table.
        This method is connected to the table's selectionChanged signal,
        allowing automatic plotting when navigating through rows using keyboard.
        """
        selected_items = self.spectrum_table.selectedItems()
        if selected_items:
            # Assuming the first selected item is in the "Index" column
            index_item = selected_items[0]
            row = index_item.row()
            self.plot_selected_spectrum(row)

    def plot_selected_spectrum(self, row=None):
        """
        Plots the spectrum corresponding to the selected row in the spectrum table.
        If 'row' is provided, it plots that row. Otherwise, it plots the currently selected row.

        Args:
            row (int, optional): The row number to plot. Defaults to None.
        """
        if row is None:
            selected_items = self.spectrum_table.selectedItems()
            if not selected_items:
                return
            row = selected_items[0].row()

        index_item = self.spectrum_table.item(row, 0)
        index_number = int(index_item.text()) if index_item and index_item.text().isdigit() else None

        if index_number is None:
            QMessageBox.warning(self, "Error", "Invalid Index Number selected.")
            return

        # Get selected file
        selected_file_item = self.file_list.currentItem()
        if not selected_file_item:
            QMessageBox.warning(self, "Error", "No mzML file selected.")
            return
        filename = selected_file_item.text()
        file_data = self.parsed_files.get(filename, {})
        h5_path = file_data.get('h5')
        metadata_df = file_data.get('metadata')

        if not h5_path or metadata_df is None:
            QMessageBox.warning(self, "Error", f"Data for file '{filename}' is incomplete.")
            return

        # Retrieve numerical data from HDF5
        try:
            with h5py.File(h5_path, 'r') as h5file:
                group_name = str(index_number)
                if group_name not in h5file:
                    QMessageBox.warning(self, "Error", f"No numerical data found for Index Number {index_number}.")
                    return
                mz = h5file[group_name]['m/z'][:]
                intensity = h5file[group_name]['intensity'][:]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to retrieve data for Index {index_number}: {e}")
            return

        # Retrieve metadata for annotation
        spectrum_meta = metadata_df[metadata_df['Index Number'] == index_number]
        if spectrum_meta.empty:
            QMessageBox.warning(self, "Error", f"No metadata found for Index Number {index_number}.")
            return
        spectrum_meta = spectrum_meta.iloc[0].to_dict()

        # Plot the spectrum
        self.plot_widget.plot_spectrum(mz, intensity, spectrum_meta)

    def keyPressEvent(self, event):
        """
        Overrides the keyPressEvent to allow opening files with Ctrl+O shortcut.

        Args:
            event (QKeyEvent): The key event.
        """
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_O:
            self.open_files_dialog()
        else:
            super().keyPressEvent(event)

    def open_files_dialog(self):
        """
        Opens a file dialog to select and load mzML files.
        """
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Open mzML Files", "",
                                                "mzML Files (*.mzml);;All Files (*)", options=options)
        if files:
            for file_path in files:
                if file_path.lower().endswith('.mzml'):
                    self.add_mzml_file(file_path)

    def closeEvent(self, event):
        """
        Prompts the user for confirmation before closing the application.

        Args:
            event (QCloseEvent): The close event.
        """
        reply = QMessageBox.question(self, 'Quit',
                                     'Are you sure you want to quit?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def apply_filters(self):
        """
        Applies the filters specified in the FilterWidget to the spectrum table.
        """
        # Get current selected file
        selected_file_item = self.file_list.currentItem()
        if not selected_file_item:
            QMessageBox.warning(self, "Error", "No mzML file selected.")
            return
        filename = selected_file_item.text()
        file_data = self.parsed_files.get(filename, {})
        metadata_df = file_data.get('metadata')

        if metadata_df is None:
            QMessageBox.warning(self, "Error", f"No metadata found for file '{filename}'.")
            return

        # Get filter criteria from FilterWidget
        filters = {}

        # Index Number
        index_op = self.filter_widget.index_operator.currentText()
        index_val1 = self.filter_widget.index_value1.text().strip()
        index_val2 = self.filter_widget.index_value2.text().strip()
        if index_val1:
            try:
                index_val1 = int(index_val1)
                if index_op == "==":
                    filters['Index Number'] = index_val1
                elif index_op == "Between" and index_val2:
                    index_val2 = int(index_val2)
                    filters['Index Number'] = (index_val1, index_val2)
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid Index Number filter value.")
                return

        # Cycle Number
        cycle_op = self.filter_widget.cycle_operator.currentText()
        cycle_val1 = self.filter_widget.cycle_value1.text().strip()
        cycle_val2 = self.filter_widget.cycle_value2.text().strip()
        if cycle_val1:
            try:
                cycle_val1 = int(cycle_val1)
                if cycle_op == "==":
                    filters['Cycle Number'] = cycle_val1
                elif cycle_op == "Between" and cycle_val2:
                    cycle_val2 = int(cycle_val2)
                    filters['Cycle Number'] = (cycle_val1, cycle_val2)
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid Cycle Number filter value.")
                return

        # Scan Start Time (RT)
        rt_op = self.filter_widget.rt_operator.currentText()
        rt_val1 = self.filter_widget.rt_value1.text().strip()
        rt_val2 = self.filter_widget.rt_value2.text().strip()
        if rt_val1:
            try:
                rt_val1 = float(rt_val1)
                if rt_op == "==":
                    filters['Scan Start Time (min)'] = rt_val1
                elif rt_op == "Between" and rt_val2:
                    rt_val2 = float(rt_val2)
                    filters['Scan Start Time (min)'] = (rt_val1, rt_val2)
                elif rt_op == "±" and rt_val2:
                    rt_val2 = float(rt_val2)
                    filters['Scan Start Time (min)'] = (rt_val1 - rt_val2, rt_val1 + rt_val2)
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid Scan Start Time filter value.")
                return

        # Selected Ion m/z
        mz_op = self.filter_widget.mz_operator.currentText()
        mz_val1 = self.filter_widget.mz_value1.text().strip()
        mz_val2 = self.filter_widget.mz_value2.text().strip()
        if mz_val1:
            try:
                mz_val1 = float(mz_val1)
                if mz_op == "==":
                    filters['Selected Ion m/z'] = mz_val1
                elif mz_op == "Between" and mz_val2:
                    mz_val2 = float(mz_val2)
                    filters['Selected Ion m/z'] = (mz_val1, mz_val2)
                elif mz_op == "± ppm" and mz_val2:
                    ppm = float(mz_val2)
                    delta = mz_val1 * ppm / 1e6
                    filters['Selected Ion m/z'] = (mz_val1 - delta, mz_val1 + delta)
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid Selected Ion m/z filter value.")
                return

        # MS Level
        ms_level_val = self.filter_widget.ms_level_value.text().strip()
        if ms_level_val:
            try:
                ms_level_val = int(ms_level_val)
                filters['MS Level'] = ms_level_val
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid MS Level filter value.")
                return

        # Apply filters to the DataFrame
        filtered_df = metadata_df.copy()

        # Apply MS Level filter
        if 'MS Level' in filters:
            filtered_df = filtered_df[filtered_df['MS Level'] == filters['MS Level']]

        # Apply Index Number filter
        if 'Index Number' in filters:
            if isinstance(filters['Index Number'], tuple):
                filtered_df = filtered_df[
                    (filtered_df['Index Number'] >= filters['Index Number'][0]) &
                    (filtered_df['Index Number'] <= filters['Index Number'][1])
                ]
            else:
                filtered_df = filtered_df[filtered_df['Index Number'] == filters['Index Number']]

        # Apply Cycle Number filter
        if 'Cycle Number' in filters:
            if isinstance(filters['Cycle Number'], tuple):
                filtered_df = filtered_df[
                    (filtered_df['Cycle Number'] >= filters['Cycle Number'][0]) &
                    (filtered_df['Cycle Number'] <= filters['Cycle Number'][1])
                ]
            else:
                filtered_df = filtered_df[filtered_df['Cycle Number'] == filters['Cycle Number']]

        # Apply Scan Start Time filter
        if 'Scan Start Time (min)' in filters:
            if isinstance(filters['Scan Start Time (min)'], tuple):
                filtered_df = filtered_df[
                    (filtered_df['Scan Start Time (min)'] >= filters['Scan Start Time (min)'][0]) &
                    (filtered_df['Scan Start Time (min)'] <= filters['Scan Start Time (min)'][1])
                ]
            else:
                filtered_df = filtered_df[filtered_df['Scan Start Time (min)'] == filters['Scan Start Time (min)']]

        # Apply Selected Ion m/z filter
        if 'Selected Ion m/z' in filters:
            if isinstance(filters['Selected Ion m/z'], tuple):
                filtered_df = filtered_df[
                    (filtered_df['Selected Ion m/z'] >= filters['Selected Ion m/z'][0]) &
                    (filtered_df['Selected Ion m/z'] <= filters['Selected Ion m/z'][1])
                ]
            else:
                filtered_df = filtered_df[filtered_df['Selected Ion m/z'] == filters['Selected Ion m/z']]

        # Update the spectrum table with filtered data
        self.spectrum_table.setRowCount(0)
        for _, row in filtered_df.iterrows():
            row_position = self.spectrum_table.rowCount()
            self.spectrum_table.insertRow(row_position)

            # Index
            if not pd.isnull(row['Index Number']):
                try:
                    index_number = int(row['Index Number'])
                    index_item = QTableWidgetItem(str(index_number))
                except ValueError:
                    index_item = QTableWidgetItem("")
            else:
                index_item = QTableWidgetItem("")
            self.spectrum_table.setItem(row_position, 0, index_item)

            # Cycle
            if not pd.isnull(row['Cycle Number']):
                try:
                    cycle_number = int(row['Cycle Number'])
                    cycle_item = QTableWidgetItem(str(cycle_number))
                except ValueError:
                    cycle_item = QTableWidgetItem("")
            else:
                cycle_item = QTableWidgetItem("")
            self.spectrum_table.setItem(row_position, 1, cycle_item)

            # Experiment
            if not pd.isnull(row['Experiment Number']):
                try:
                    experiment_number = int(row['Experiment Number'])
                    experiment_item = QTableWidgetItem(str(experiment_number))
                except ValueError:
                    experiment_item = QTableWidgetItem("")
            else:
                experiment_item = QTableWidgetItem("")
            self.spectrum_table.setItem(row_position, 2, experiment_item)

            # MS Level
            if not pd.isnull(row['MS Level']):
                try:
                    ms_level = int(row['MS Level'])
                    ms_level_item = QTableWidgetItem(str(ms_level))
                except ValueError:
                    ms_level_item = QTableWidgetItem("")
            else:
                ms_level_item = QTableWidgetItem("")
            self.spectrum_table.setItem(row_position, 3, ms_level_item)

            # TIC
            if not pd.isnull(row['Total Ion Current']):
                tic = f"{row['Total Ion Current']:.2f}"
                tic_item = QTableWidgetItem(tic)
            else:
                tic_item = QTableWidgetItem("")
            self.spectrum_table.setItem(row_position, 4, tic_item)

            # Scan Start Time (RT)
            if not pd.isnull(row['Scan Start Time (min)']):
                rt = f"{row['Scan Start Time (min)']:.4f}"
                rt_item = QTableWidgetItem(rt)
            else:
                rt_item = QTableWidgetItem("")
            self.spectrum_table.setItem(row_position, 5, rt_item)

            # Selected Ion m/z
            if pd.notnull(row['Selected Ion m/z']):
                selected_ion_mz = f"{row['Selected Ion m/z']:.4f}"
            else:
                selected_ion_mz = "Full MS"
            selected_ion_mz_item = QTableWidgetItem(selected_ion_mz)
            self.spectrum_table.setItem(row_position, 6, selected_ion_mz_item)

        self.spectrum_table.resizeColumnsToContents()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
