

from typing import TYPE_CHECKING
import os
import numpy as np
import tifffile
import pandas as pd
from skimage.io import imread as skimage_imread
from skimage.transform import resize
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QCheckBox,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QComboBox
)



if TYPE_CHECKING:
    import napari

base_dir = os.path.dirname(os.path.abspath(__file__))


class DataUploadWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.background_impedance_data = None

        # Coordinates for cropping wells from large microscopy image
        self.indices = [
            (277 - 10, 277 - 20), (1405, 277 - 10), (277 - 20, 1405), (1405, 1405 + 10),
            (277, 2308 + 1405 + 20 + 15), (1405 + 5, 2308 + 1405 + 23 + 20),
            (277 - 10, 2308 + 277), (1405 + 12, 2308 + 277 + 10),
            (2308 + 1405 + 20, 277 - 15), (2308 + 277, 277 - 15),
            (2308 + 1405 + 20, 1405 + 10 + 10), (2308 + 277, 1405 + 15),
            (2308 + 1405 + 20, 2308 + 1405 + 30 + 15), (2308 + 277, 2308 + 1405 + 25 + 15),
            (2308 + 1405 + 25, 2308 + 277 + 15), (2308 + 277, 2308 + 277 + 15)
        ]
        self.crop_sizes = [384] * 16

        self.full_microscopy_image = None
        self.current_heatmaps = None
        self.current_well_index = 0

        try:
            self.viewer.window._qt_window.showMaximized()
        except Exception as e:
            print(f"Failed to maximize viewer window: {e}")

        layout = QVBoxLayout()
        self.setLayout(layout)

         # === File Upload Group ===
        file_upload_group = QGroupBox("File Upload")
        file_upload_layout = QVBoxLayout()
        file_upload_group.setLayout(file_upload_layout)
        layout.addWidget(file_upload_group)

        self.btn_upload_microscopy = QPushButton("Upload Microscopy Image")
        self.btn_upload_microscopy.setToolTip("Load the main microscopy image.")
        self.btn_upload_microscopy.clicked.connect(self.load_microscopy_image)
        file_upload_layout.addWidget(self.btn_upload_microscopy)

        self.chk_subtract_background = QCheckBox("Subtract Background Impedance")
        self.chk_subtract_background.setToolTip("Enable this to remove background impedance from measurements using a reference CSV.")
        self.chk_subtract_background.setChecked(False)
        self.chk_subtract_background.stateChanged.connect(self.on_checkbox_toggled)
        file_upload_layout.addWidget(self.chk_subtract_background)

        self.btn_upload_background = QPushButton("Upload Background Impedance")
        self.btn_upload_background.setToolTip("Upload a CSV with background impedance values (e.g., media-only).")
        self.btn_upload_background.clicked.connect(self.load_background_data)
        file_upload_layout.addWidget(self.btn_upload_background)

        self.btn_upload_impedance = QPushButton("Upload Impedance Data (CSV)")
        self.btn_upload_impedance.setToolTip("Upload a CSV with impedance measurements for each electrode.")
        self.btn_upload_impedance.clicked.connect(self.load_impedance_data)
        file_upload_layout.addWidget(self.btn_upload_impedance)

        # === Well Selection & Cropping Parameters ===
        wells_group = QGroupBox("Well Selection and Cropping")
        wells_layout = QVBoxLayout()
        wells_group.setLayout(wells_layout)
        layout.addWidget(wells_group)

        self.chk_mea_chip = QCheckBox("MEA Chip?")
        self.chk_mea_chip.setToolTip("Check this if the image comes from a CMOS MEA chip to use predefined cropping.")
        self.chk_mea_chip.setChecked(False)
        self.chk_mea_chip.stateChanged.connect(self.on_mea_chip_toggled)
        wells_layout.insertWidget(0, self.chk_mea_chip)

        self.well_selector = QComboBox()
        self.well_selector.setToolTip("Select which well to view or crop from the microscopy image.")
        self.well_selector.addItems([f"Well {i+1}" for i in range(16)])
        self.well_selector.currentIndexChanged.connect(self.on_well_changed)
        wells_layout.addWidget(self.well_selector)

        self.input_top_left_x = QLineEdit()
        self.input_top_left_x.setFixedWidth(100)
        self.input_top_left_x.setToolTip("Manually set the X-coordinate of the well's top-left corner.")
        wells_layout.addWidget(QLabel("Top Left X:"))
        wells_layout.addWidget(self.input_top_left_x)

        self.input_top_left_y = QLineEdit()
        self.input_top_left_y.setFixedWidth(100)
        self.input_top_left_y.setToolTip("Manually set the Y-coordinate of the well's top-left corner.")
        wells_layout.addWidget(QLabel("Top Left Y:"))
        wells_layout.addWidget(self.input_top_left_y)

        self.input_crop_size = QLineEdit()
        self.input_crop_size.setFixedWidth(100)
        self.input_crop_size.setToolTip("Set the square crop size for extracting individual wells (in pixels).")
        wells_layout.addWidget(QLabel("Crop Size:"))
        wells_layout.addWidget(self.input_crop_size)

        self.input_top_left_x.editingFinished.connect(self.update_top_left_x)
        self.input_top_left_y.editingFinished.connect(self.update_top_left_y)
        self.input_crop_size.editingFinished.connect(self.update_crop_size)

        self.btn_update_crop = QPushButton("Update Crop and Show Well")
        self.btn_update_crop.setToolTip("Apply crop settings and display the selected well in Napari.")
        self.btn_update_crop.clicked.connect(self.update_crop_and_show)
        wells_layout.addWidget(self.btn_update_crop)

        self.on_mea_chip_toggled()

        # Pre-fill inputs with defaults for first well
        self.on_well_changed(0)

        self.microscopy_layer_name = "Microscopy Image"
        self.impedance_layer_name = "Impedance Image"

        self.update_button_states()

        self.viewer.layers.events.inserted.connect(lambda event: self.update_button_states())
        self.viewer.layers.events.removed.connect(self.on_layer_removed)


    def on_mea_chip_toggled(self):
        enabled = self.chk_mea_chip.isChecked()

        widgets_to_toggle = [
            self.well_selector,
            self.input_top_left_x,
            self.input_top_left_y,
            self.input_crop_size,
            self.btn_update_crop,
        ]

        for w in widgets_to_toggle:
            w.setEnabled(enabled)
            # Gray out if disabled
            if not enabled:
                w.setStyleSheet("color: gray; background-color: #444444;")
            else:
                w.setStyleSheet("")

    def on_well_changed(self, index):
        self.current_well_index = index
        x, y = self.indices[index]
        crop_size = self.crop_sizes[index]
        self.input_top_left_x.setText(str(x))
        self.input_top_left_y.setText(str(y))
        self.input_crop_size.setText(str(crop_size))

    def update_crop_and_show(self):
        if not self.chk_mea_chip.isChecked():
            print("MEA Chip not selected — skipping crop update.")
            return
        try:
            x = int(self.input_top_left_x.text())
            y = int(self.input_top_left_y.text())
            crop_size = int(self.input_crop_size.text())

            # Update your indices list or wherever you store coordinates
            self.indices[self.current_well_index] = (x, y)
            self.current_crop_size = crop_size  # store crop size if needed

            # Refresh views
            self.on_well_selected(self.current_well_index)

        except ValueError:
            print("Please enter valid integer coordinates and crop size")


    def on_checkbox_toggled(self):
        enable_bg = self.chk_subtract_background.isChecked() and (self.background_impedance_data is None)
        self.btn_upload_background.setEnabled(enable_bg)
        self.update_button_states()

    def update_button_states(self):
        dark_gray = "#444444"
        gray_font = "#888888"
        tooltip_style = """
            QToolTip {
                background-color: #333333;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
        """

        def style_button(button, disabled):
            if disabled:
                button.setEnabled(False)
                button.setStyleSheet(f"background-color: {dark_gray}; color: {gray_font};" + tooltip_style)
            else:
                button.setEnabled(True)
                button.setStyleSheet("")

        # Microscopy button disabled if layer present
        microscopy_disabled = self.microscopy_layer_name in self.viewer.layers
        style_button(self.btn_upload_microscopy, microscopy_disabled)
        self.btn_upload_microscopy.setToolTip("Microscopy image already uploaded." if microscopy_disabled else "")

        # Impedance button disabled if layer present
        impedance_disabled = self.impedance_layer_name in self.viewer.layers
        style_button(self.btn_upload_impedance, impedance_disabled)
        self.btn_upload_impedance.setToolTip("Impedance data already uploaded." if impedance_disabled else "")

        # Background button enabled only if checkbox is checked and no background loaded
        bg_enabled = self.chk_subtract_background.isChecked() and (self.background_impedance_data is None)
        style_button(self.btn_upload_background, not bg_enabled)

        if self.background_impedance_data is not None:
            style_button(self.btn_upload_background, True)
            self.btn_upload_background.setToolTip("Background impedance data loaded.")
        else:
            if self.chk_subtract_background.isChecked():
                self.btn_upload_background.setToolTip("Upload a background impedance file.")
            else:
                self.btn_upload_background.setToolTip("Background upload disabled (checkbox unchecked).")

    def on_layer_removed(self, event):
        removed_layer = event.value
        if removed_layer.name == self.impedance_layer_name:
            self.background_impedance_data = None
            print("Impedance layer removed. Background impedance data cleared.")
            self.update_button_states()
        else:
            self.update_button_states()

    def load_image(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.tif', '.tiff']:
            return tifffile.imread(file_path)
        elif ext in ['.png', '.jpg', '.jpeg']:
            return skimage_imread(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def resize_to_512(self, image):
        return resize(image, (512, 512), preserve_range=True, anti_aliasing=True).astype(image.dtype)

    def load_microscopy_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Microscopy Image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg)"
        )
        if file_path:
            data = self.load_image(file_path)
            self.full_microscopy_image = data  # keep original big image
            self.show_well_microscopy(0)       # show first well cropped image
            self.update_button_states()

    def show_well_microscopy(self, well_index):
        if self.full_microscopy_image is None:
            return
        if not self.chk_mea_chip.isChecked():
            # Show full image without cropping (or do nothing)
            if self.microscopy_layer_name in self.viewer.layers:
                self.viewer.layers.remove(self.microscopy_layer_name)
            self.viewer.add_image(self.full_microscopy_image, name=self.microscopy_layer_name)
            return

        x, y = self.indices[well_index]
        crop = self.full_microscopy_image[y:y + self.crop_sizes[well_index], x:x + self.crop_sizes[well_index]]
        crop_resized = self.resize_to_512(crop)

        if self.microscopy_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.microscopy_layer_name)
        self.viewer.add_image(crop_resized, name=self.microscopy_layer_name)

    def load_background_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Background Impedance CSV File", "", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                self.background_impedance_data = pd.read_csv(file_path)
                print("Background impedance data loaded.")
            except Exception as e:
                print(f"Failed to load background data: {e}")
            self.update_button_states()

    def load_impedance_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Impedance CSV File", "", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                df = pd.read_csv(file_path)
                heatmaps_16x16 = self.convert_impedance_to_heatmap(
                    df,
                    self.background_impedance_data if self.chk_subtract_background.isChecked() else None
                )
                self.current_heatmaps = heatmaps_16x16  # store all 16 wells heatmaps

                # Show impedance for current well
                self.show_well_impedance(self.current_well_index)
                self.update_button_states()
            except Exception as e:
                print(f"Failed to load and process impedance data: {e}")

    def show_well_impedance(self, well_index):
        if self.current_heatmaps is None:
            return

        heatmap = self.current_heatmaps[well_index]
        heatmap_512 = self.resize_to_512(heatmap)

        if self.impedance_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.impedance_layer_name)

        layer = self.viewer.add_image(heatmap_512, name=self.impedance_layer_name)
        layer.translate = (0, 576)
        self.adjust_view_for_impedance()

    def on_well_selected(self, well_index):
        self.current_well_index = well_index
        self.show_well_microscopy(well_index)
        self.show_well_impedance(well_index)

    def convert_impedance_to_heatmap(self, current_data, background_data=None):
        def sparrowHeatmap(chipid, data, mode, save, save_path='Default'):
            electrode_table = pd.read_csv(os.path.join(base_dir, 'Electrode Reference Table.csv'))
            well_table = pd.read_csv(os.path.join(base_dir, 'Well Reference Table.csv'))

            matrix_size = (16, 16)
            heatmap_data = np.empty((16,) + matrix_size)

            for index, value in data.items():
                n = electrode_table.iloc[index - 1, 1] - 1
                i = electrode_table.iloc[index - 1, 2] - 1
                j = electrode_table.iloc[index - 1, 3] - 1
                heatmap_data[n, i, j] = value
            return heatmap_data

        def extract_averages(df):
            subset = df.iloc[0:16385, 1:17]
            values = subset.apply(lambda row: row.dropna().values[0] if row.dropna().size > 0 else None, axis=1)
            values = values.reset_index(drop=True)
            return pd.Series([
                sum(float(values.iloc[i + j]) for j in range(4)) / 4.0
                for i in range(0, 16384, 4)
            ])

        main_avg = extract_averages(current_data)

        if background_data is not None:
            try:
                bg_avg = extract_averages(background_data)
                diff_avg = main_avg - bg_avg
                print("Background impedance subtracted.")
            except Exception as e:
                print(f"Failed background subtraction: {e}")
                diff_avg = main_avg
        else:
            diff_avg = main_avg

        heatmap = sparrowHeatmap("Caption", diff_avg, 'n', 0)
        return heatmap

    def adjust_view_for_impedance(self):
        self.viewer.camera.zoom = 0.5
        center = self.viewer.camera.center
        self.viewer.camera.center = (center[0] + 250, center[1] + 275)

    # Make sure to call your adjusted adjust_view_for_impedance() inside show_well_impedance:
    def show_well_impedance(self, well_index):
        if self.current_heatmaps is None:
            return

        heatmap = self.current_heatmaps[well_index]
        heatmap_512 = self.resize_to_512(heatmap)

        if self.impedance_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.impedance_layer_name)

        layer = self.viewer.add_image(heatmap_512, name=self.impedance_layer_name)
        layer.translate = (0, 576)
        self.adjust_view_for_impedance()

    def update_top_left_x(self):
        try:
            x = int(self.input_top_left_x.text())
            _, y = self.indices[self.current_well_index]
            self.indices[self.current_well_index] = (x, y)
        except ValueError:
            print("Invalid Top Left X")

    def update_top_left_y(self):
        try:
            y = int(self.input_top_left_y.text())
            x, _ = self.indices[self.current_well_index]
            self.indices[self.current_well_index] = (x, y)
        except ValueError:
            print("Invalid Top Left Y")

    def update_crop_size(self):
        try:
            size = int(self.input_crop_size.text())
            self.crop_sizes[self.current_well_index] = size
        except ValueError:
            print("Invalid Crop Size")



from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QCheckBox,
    QHBoxLayout, QSpinBox, QComboBox, QGroupBox
)
from qtpy.QtCore import Qt
import numpy as np
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import remove_small_objects

import csv
from pathlib import Path
from qtpy.QtWidgets import QFileDialog

import struct
import zipfile
import roifile

from skimage.draw import polygon as sk_polygon
from skimage.measure import find_contours

class ThresholdSegmentationWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.microscopy_layer_name = "Microscopy Image"
        self.mask_layer_name = "Segmentation Mask"
        self.label_layer_name = "Labeled Mask"
        self.roi_layer_name = "ROIs"

        layout = QVBoxLayout()
        self.setLayout(layout)

        # === Preprocessing Pane ===
        preprocessing_group = QGroupBox("Preprocessing")
        preprocessing_layout = QVBoxLayout()
        preprocessing_group.setLayout(preprocessing_layout)
        layout.addWidget(preprocessing_group)

        hist_layout = QHBoxLayout()
        hist_label = QLabel("Histogram Norm:")
        hist_label.setToolTip("Normalize contrast/brightness per channel")
        hist_layout.addWidget(hist_label)
        self.hist_btn_r = QPushButton("R")
        self.hist_btn_g = QPushButton("G")
        self.hist_btn_b = QPushButton("B")
        self.hist_btn_r.setToolTip("Normalize red channel using histogram equalization")
        self.hist_btn_g.setToolTip("Normalize green channel using histogram equalization")
        self.hist_btn_b.setToolTip("Normalize blue channel using histogram equalization")
        self.hist_btn_r.clicked.connect(lambda: self.normalize_channel_in_view('red', mode='hist'))
        self.hist_btn_g.clicked.connect(lambda: self.normalize_channel_in_view('green', mode='hist'))
        self.hist_btn_b.clicked.connect(lambda: self.normalize_channel_in_view('blue', mode='hist'))
        hist_layout.addWidget(self.hist_btn_r)
        hist_layout.addWidget(self.hist_btn_g)
        hist_layout.addWidget(self.hist_btn_b)
        preprocessing_layout.addLayout(hist_layout)

        # === Segmentation Pane ===
        segmentation_group = QGroupBox("Segmentation")
        segmentation_layout = QVBoxLayout()
        segmentation_group.setLayout(segmentation_layout)
        layout.addWidget(segmentation_group)

        title = QLabel("RGB Threshold-Based Segmentation")
        title.setToolTip("Segment by selecting color channel thresholds (0-255)")
        segmentation_layout.addWidget(title)

        self.r_check, self.r_thresh = self._make_channel_input("Red", segmentation_layout)
        self.g_check, self.g_thresh = self._make_channel_input("Green", segmentation_layout)
        self.b_check, self.b_thresh = self._make_channel_input("Blue", segmentation_layout)

        logic_layout = QHBoxLayout()
        logic_label = QLabel("Apply Logic:")
        logic_label.setToolTip("Choose 'AND' to require all thresholds, 'OR' for any")
        logic_layout.addWidget(logic_label)
        self.logic_box = QComboBox()
        self.logic_box.addItems(["AND", "OR"])
        logic_layout.addWidget(self.logic_box)
        segmentation_layout.addLayout(logic_layout)

        self.segment_btn = QPushButton("Apply Segmentation")
        self.segment_btn.setToolTip("Run thresholding based on selected channels and logic")
        self.segment_btn.clicked.connect(self.apply_segmentation)
        segmentation_layout.addWidget(self.segment_btn)

        # === Labeling & ROI Pane ===
        label_group = QGroupBox("Labeling & ROIs")
        label_layout = QVBoxLayout()
        label_group.setLayout(label_layout)
        layout.addWidget(label_group)

        self.label_btn = QPushButton("Label Instances")
        self.label_btn.setToolTip("Assign unique labels to connected regions")
        self.label_btn.clicked.connect(self.label_mask)
        label_layout.addWidget(self.label_btn)

        filter_layout = QHBoxLayout()
        filter_label = QLabel("Min Size (px):")
        filter_label.setToolTip("Exclude instances smaller than this value")
        filter_layout.addWidget(filter_label)
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 100000)
        self.min_size_spin.setValue(50)
        filter_layout.addWidget(self.min_size_spin)
        label_layout.addLayout(filter_layout)

        self.filter_btn = QPushButton("Remove Small Instances")
        self.filter_btn.setToolTip("Delete segments below min size threshold")
        self.filter_btn.clicked.connect(self.filter_labels)
        label_layout.addWidget(self.filter_btn)

        self.roi_btn = QPushButton("Convert Labels to ROIs")
        self.roi_btn.setToolTip("Convert labeled areas into editable polygon ROIs")
        self.roi_btn.clicked.connect(self.labels_to_rois)
        label_layout.addWidget(self.roi_btn)

        self.impedance_btn = QPushButton("Compute Impedance per ROI")
        self.impedance_btn.setToolTip("Extract impedance values within each ROI")
        self.impedance_btn.clicked.connect(self.calculate_impedance_per_roi)
        label_layout.addWidget(self.impedance_btn)

        self.recompute_btn = QPushButton("Recompute Based on ROI Corrections")
        self.recompute_btn.setToolTip("Re-run calculations using manually corrected ROIs")
        self.recompute_btn.clicked.connect(self.recompute_based_on_corrections)
        label_layout.addWidget(self.recompute_btn)

        self.btn_save_microscopy = QPushButton("Save Microscopy Image")
        self.btn_save_microscopy.setToolTip("Save the current microscopy layer to disk")
        self.btn_save_microscopy.clicked.connect(self.save_microscopy_image)
        layout.addWidget(self.btn_save_microscopy)

        self.save_rois_button = QPushButton("Save ROIs as ZIP")
        self.save_rois_button.setToolTip("Save drawn ROIs in a .zip format compatible with ImageJ")
        self.save_rois_button.clicked.connect(self.save_rois_as_zip)
        layout.addWidget(self.save_rois_button)

        self.export_csv_button = QPushButton("Export ROI Table to CSV")
        self.export_csv_button.setToolTip("Save per-ROI properties and impedance to CSV")
        self.export_csv_button.clicked.connect(self.export_roi_data_to_csv)
        layout.addWidget(self.export_csv_button)

    def save_microscopy_image(self):
        if self.microscopy_layer_name not in self.viewer.layers:
            print("No Microscopy Image layer to save.")
            return

        layer = self.viewer.layers[self.microscopy_layer_name]
        image_data = layer.data

        # Open file dialog for saving
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Microscopy Image",
            "",
            "TIFF Files (*.tif *.tiff);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)"
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in ['.tif', '.tiff']:
                tifffile.imwrite(file_path, image_data)
            elif ext in ['.png', '.jpg', '.jpeg']:
                from skimage.io import imsave
                imsave(file_path, image_data)
            else:
                print(f"Unsupported file extension '{ext}'. Saving as TIFF.")
                tifffile.imwrite(file_path, image_data)
            print(f"Microscopy image saved to {file_path}")
        except Exception as e:
            print(f"Failed to save image: {e}")


    def _make_channel_input(self, label, parent_layout):
        layout = QHBoxLayout()
        checkbox = QCheckBox(f"Use {label}")
        spinbox = QSpinBox()
        spinbox.setRange(0, 255)
        spinbox.setValue(175)
        spinbox.setEnabled(False)
        checkbox.stateChanged.connect(lambda state: spinbox.setEnabled(state == Qt.Checked))
        layout.addWidget(checkbox)
        layout.addWidget(spinbox)
        parent_layout.addLayout(layout)
        return checkbox, spinbox

    def apply_segmentation(self):
        if self.microscopy_layer_name not in self.viewer.layers:
            print("Microscopy image not found.")
            return

        img = self.viewer.layers[self.microscopy_layer_name].data
        if img.ndim != 3 or img.shape[-1] != 3:
            print("Image is not RGB.")
            return

        logic = self.logic_box.currentText()
        mask = None

        if self.r_check.isChecked():
            r_mask = img[..., 0] > self.r_thresh.value()
            mask = r_mask if mask is None else (mask & r_mask if logic == "AND" else mask | r_mask)
        if self.g_check.isChecked():
            g_mask = img[..., 1] > self.g_thresh.value()
            mask = g_mask if mask is None else (mask & g_mask if logic == "AND" else mask | g_mask)
        if self.b_check.isChecked():
            b_mask = img[..., 2] > self.b_thresh.value()
            mask = b_mask if mask is None else (mask & b_mask if logic == "AND" else mask | b_mask)

        if mask is None:
            print("No channels selected.")
            return

        mask = mask.astype(np.uint8)
        
        # Remove old mask layers if any
        if self.mask_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.mask_layer_name)
        if f"{self.mask_layer_name} (Impedance)" in self.viewer.layers:
            self.viewer.layers.remove(f"{self.mask_layer_name} (Impedance)")
        
        # Add mask layer aligned with microscopy image (no translation)
        self.viewer.add_labels(mask, name=self.mask_layer_name)
        
        # Add mask layer aligned with impedance image (translated)
        if "Impedance Image" in self.viewer.layers:
            impedance_mask_layer = self.viewer.add_labels(mask, name=f"{self.mask_layer_name} (Impedance)")
            impedance_mask_layer.translate = (0, 576)


    def label_mask(self):
        if self.mask_layer_name not in self.viewer.layers:
            print("No binary mask to label.")
            return
        mask = self.viewer.layers[self.mask_layer_name].data.astype(bool)
        labeled = label(mask)

        for layer_name in [
            self.mask_layer_name,
            f"{self.mask_layer_name} (Impedance)",
        ]:
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].visible = False
        
        # Remove old labeled layers if any
        if self.label_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.label_layer_name)
        if f"{self.label_layer_name} (Impedance)" in self.viewer.layers:
            self.viewer.layers.remove(f"{self.label_layer_name} (Impedance)")
        
        # Add labeled layer aligned with microscopy image (no translation)
        self.viewer.add_labels(labeled, name=self.label_layer_name)
        
        # Add labeled layer aligned with impedance image (translated)
        if "Impedance Image" in self.viewer.layers:
            impedance_label_layer = self.viewer.add_labels(labeled, name=f"{self.label_layer_name} (Impedance)")
            impedance_label_layer.translate = (0, 576)

    def filter_labels(self):
        if self.label_layer_name not in self.viewer.layers:
            print("No labeled layer to filter.")
            return
        labeled = self.viewer.layers[self.label_layer_name].data
        min_size = self.min_size_spin.value()
        filtered = remove_small_objects(labeled, min_size=min_size)
        self.viewer.layers[self.label_layer_name].data = filtered
        print(f"Filtered out objects smaller than {min_size} pixels.")

    def labels_to_rois(self):
        if self.label_layer_name not in self.viewer.layers:
            print("No labeled layer for ROI extraction.")
            return

        labeled_layer = self.viewer.layers[self.label_layer_name]
        labeled = labeled_layer.data
        roi_shapes = []
        edge_colors = []
        face_colors = []

        get_color = labeled_layer.get_color

        for region in regionprops(labeled):
            binary_mask = labeled == region.label
            contours = find_contours(binary_mask, 0.5)
            if not contours:
                continue
            contour = max(contours, key=len)

            contour_flipped = contour[:, [0, 1]]  # (row, col)

            roi_shapes.append(contour_flipped)

            rgba = get_color(region.label)
            rgba = (rgba[0], rgba[1], rgba[2], 0.75)
            edge_colors.append(rgba)
            face_colors.append(rgba)

        # Remove old ROI layers if any
        if self.roi_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.roi_layer_name)
        if f"{self.roi_layer_name} (Impedance)" in self.viewer.layers:
            self.viewer.layers.remove(f"{self.roi_layer_name} (Impedance)")

        # Add ROI layer aligned with microscopy image (no translation)
        roi_layer = self.viewer.add_shapes(
            data=roi_shapes,
            shape_type="polygon",
            # edge_color=edge_colors,
            face_color=face_colors,
            opacity=1.0,
            name=self.roi_layer_name,
        )
        roi_layer.editable = True

        if "Impedance Image" in self.viewer.layers:
            # Add ROI layer aligned with impedance image (translated)
            impedance_roi_layer = self.viewer.add_shapes(
                data=roi_shapes,
                shape_type="polygon",
                # edge_color=edge_colors,
                face_color=face_colors,
                opacity=1.0,
                name=f"{self.roi_layer_name} (Impedance)",
            )
            impedance_roi_layer.editable = True
            impedance_roi_layer.translate = (0, 576)

        for layer_name in [
            self.label_layer_name,
            f"{self.label_layer_name} (Impedance)",
        ]:
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].visible = False

        print(f"Extracted {len(roi_shapes)} editable ROIs with instance colors on both images.")


    def normalize_channel(self, channel_data, mode="hist", target_mean=127.5, target_std=40):
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std == 0:
            std = 1
        if mode == "hist":
            normalized = (channel_data - mean) / std
            normalized = normalized * target_std + target_mean
        elif mode == "zscore":
            normalized = (channel_data - mean) / std
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min()) * 255
        else:
            print(f"Unknown normalization mode: {mode}")
            return channel_data
        return np.clip(normalized, 0, 255).astype(np.uint8)

    def normalize_channel_in_view(self, color_channel, mode="hist"):
        if self.microscopy_layer_name not in self.viewer.layers:
            print("No microscopy image loaded to normalize.")
            return
        layer = self.viewer.layers[self.microscopy_layer_name]
        img = layer.data.copy()
        if img.ndim == 2:
            print("Microscopy image is grayscale, RGB normalization not applicable.")
            return
        channel_map = {'red': 0, 'green': 1, 'blue': 2}
        ch_idx = channel_map.get(color_channel.lower())
        if ch_idx is None:
            print(f"Unknown channel: {color_channel}")
            return
        img[..., ch_idx] = self.normalize_channel(img[..., ch_idx], mode=mode)
        layer.data = img
        print(f"{color_channel.upper()} channel normalized with {mode}.")

    def calculate_impedance_per_roi(self):
        if self.label_layer_name not in self.viewer.layers:
            print("No labeled layer found.")
            return

        if "Impedance Image" not in self.viewer.layers:
            print("No impedance image layer found.")
            return

        label_data = self.viewer.layers[self.label_layer_name].data
        impedance_data = self.viewer.layers["Impedance Image"].data

        if label_data.shape != impedance_data.shape:
            print("Shape mismatch between labels and impedance image.")
            return

        text_points = []
        impedance_strings = []
        index_strings = []

        for region in regionprops(label_data, intensity_image=impedance_data):
            if region.area == 0:
                continue
            mean_val = region.mean_intensity
            centroid = region.centroid  # (row, col)

            x, y = centroid[0], centroid[1] + 576
            text_points.append((x, y))
            impedance_strings.append(f"{mean_val:.2f}")
            index_strings.append(str(region.label))

        # Calculate background mask and mean impedance
        background_mask = label_data == 0
        mean_background_val = np.mean(impedance_data[background_mask])

        # Position for background text: below image center + offset
        y_bkg = impedance_data.shape[0] + 30  # 30 pixels below image bottom
        x_bkg = impedance_data.shape[1] // 2 + 576

        # Remove old text layers
        for layer_name in ["Mean Impedance Text", "Instance Index Text", "Background Impedance Text"]:
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)

        # Add impedance values for ROIs
        self.viewer.add_points(
            text_points,
            name="Mean Impedance Text",
            face_color="transparent",
            edge_color="white",
            size=1,
            text={
                "string": impedance_strings,
                "size": 12,
                "color": "white",
                "anchor": "center",
            },
        )

        # Add background impedance text below image
        self.viewer.add_points(
            [(y_bkg, x_bkg)],
            name="Background Impedance Text",
            face_color="transparent",
            edge_color="white",
            size=1,
            text={
                "string": [f"Background Impedance : {mean_background_val:.2f}"],
                "size": 12,
                "color": "yellow",
                "anchor": "center",
            },
        )

        print(f"Displayed mean impedance and index for {len(text_points)} instances plus background.")


    def recompute_based_on_corrections(self):
        self.correct_masks_from_rois()
        # Check if microscopy ROI layer exists
        if self.roi_layer_name not in self.viewer.layers:
            print("No microscopy ROIs found to recompute.")
            return

        # Remove the impedance ROI layer and impedance text layers if they exist
        impedance_roi_name = f"{self.roi_layer_name} (Impedance)"
        for layer_name in [impedance_roi_name, "Mean Impedance Text", "Instance Index Text"]:
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)

        # Copy microscopy ROIs and translate them to impedance coordinates
        microscopy_roi_layer = self.viewer.layers[self.roi_layer_name]

        # Copy shapes data and colors
        roi_shapes = microscopy_roi_layer.data.copy()
        edge_colors = microscopy_roi_layer.edge_color.copy()
        face_colors = microscopy_roi_layer.face_color.copy()

        if "Impedance Image" not in self.viewer.layers:
            print("No impedance image layer found.")
            return

        # Add new impedance ROI layer with translation (0, 576)
        impedance_roi_layer = self.viewer.add_shapes(
            data=roi_shapes,
            shape_type="polygon",
            # edge_color=edge_colors,
            face_color=face_colors,
            opacity=1.0,
            name=impedance_roi_name,
        )
        impedance_roi_layer.editable = True
        impedance_roi_layer.translate = (0, 576)

        # Now recompute impedance per ROI for the impedance layer
        self.calculate_impedance_per_roi_from_shapes(impedance_roi_layer)

        print("Recomputed impedance ROIs and values based on microscopy corrections.")

    def calculate_impedance_per_roi_from_shapes(self, roi_layer):
        if "Impedance Image" not in self.viewer.layers:
            print("No impedance image layer found.")
            return

        impedance_data = self.viewer.layers["Impedance Image"].data

        text_points = []
        impedance_strings = []

        # Process instance ROIs first
        for shape in roi_layer.data:
            mask = np.zeros(impedance_data.shape[:2], dtype=bool)
            polygon = shape.astype(np.int32)
            rr, cc = sk_polygon(polygon[:, 0], polygon[:, 1], mask.shape)
            mask[rr, cc] = True

            mean_val = np.mean(impedance_data[mask]) if np.any(mask) else 0.0
            centroid = np.mean(polygon, axis=0)
            y, x = centroid
            text_points.append((y, x + 576))
            impedance_strings.append(f"{mean_val:.2f}")

        # Remove old text layers
        if "Mean Impedance Text" in self.viewer.layers:
            self.viewer.layers.remove("Mean Impedance Text")

        # Add ROI impedance text points
        self.viewer.add_points(
            text_points,
            name="Mean Impedance Text",
            face_color="transparent",
            #edge_color="white",
            size=1,
            text={
                "string": impedance_strings,
                "size": 12,
                "color": "white",
                "anchor": "center",
            },
        )

        # --- Now handle background separately ---

        # Create combined mask of all ROIs
        full_mask = np.zeros(impedance_data.shape[:2], dtype=bool)
        for shape in roi_layer.data:
            polygon = shape.astype(np.int32)
            rr, cc = sk_polygon(polygon[:, 0], polygon[:, 1], full_mask.shape)
            full_mask[rr, cc] = True

        background_mask = ~full_mask

        mean_background_val = np.mean(impedance_data[background_mask])

        # Place background label **below the image**:
        # image height = impedance_data.shape[0]
        y_bkg = impedance_data.shape[0] + 30  # 30 pixels below bottom edge
        x_bkg = impedance_data.shape[1] // 2 + 576  # centered horizontally + offset

        # Remove old background label if any
        if "Background Impedance Text" in self.viewer.layers:
            self.viewer.layers.remove("Background Impedance Text")

        self.viewer.add_points(
            [(y_bkg, x_bkg)],
            name="Background Impedance Text",
            face_color="transparent",
            #edge_color="white",
            size=1,
            text={
                "string": [f"Background Impedance : {mean_background_val:.2f}"],
                "size": 12,
                "color": "yellow",
                "anchor": "center",
            },
        )

    def correct_masks_from_rois(self):
        from skimage.draw import polygon as sk_polygon

        if self.roi_layer_name not in self.viewer.layers:
            print("No ROIs found.")
            return

        roi_layer = self.viewer.layers[self.roi_layer_name]
        img_shape = self.viewer.layers[self.microscopy_layer_name].data.shape[:2]

        # 1. Create binary mask from polygons
        mask = np.zeros(img_shape, dtype=bool)
        for shape in roi_layer.data:
            poly = shape.astype(np.int32)
            rr, cc = sk_polygon(poly[:, 0], poly[:, 1], img_shape)
            mask[rr, cc] = True

        # 2. Replace Segmentation Mask layer
        if self.mask_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.mask_layer_name)
        if f"{self.mask_layer_name} (Impedance)" in self.viewer.layers:
            self.viewer.layers.remove(f"{self.mask_layer_name} (Impedance)")

        self.viewer.add_labels(mask.astype(np.uint8), name=self.mask_layer_name)
        imp_mask_layer = self.viewer.add_labels(mask.astype(np.uint8), name=f"{self.mask_layer_name} (Impedance)")
        imp_mask_layer.translate = (0, 576)

        # 3. Relabel the new mask
        self.label_mask()

        for layer_name in [
            "Labeled Mask",
            "Labeled Mask (Impedance)"
        ]:
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].visible = False

        print("Corrected masks and ROIs from edited ROI layer.")

    def save_rois_to_csv(self):
        if self.roi_layer_name not in self.viewer.layers:
            print("No ROIs to save.")
            return
        
        roi_layer = self.viewer.layers[self.roi_layer_name]
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save ROIs CSV", filter="CSV Files (*.csv);;All Files (*)"
        )
        if not output_path:
            return
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ROI_Index', 'Polygon_Vertices'])
            for i, polygon in enumerate(roi_layer.data, start=1):
                # Flatten polygon points into a string "x1,y1;x2,y2;..."
                vertices_str = ";".join([f"{x:.2f},{y:.2f}" for x, y in polygon])
                writer.writerow([i, vertices_str])
        
        print(f"Saved {len(roi_layer.data)} ROIs to {output_path}")

    def export_roi_data_to_csv(self):
        if self.roi_layer_name not in self.viewer.layers:
            print("No ROIs found for export.")
            return

        roi_layer = self.viewer.layers[self.roi_layer_name]
        impedance_exists = "Impedance Image" in self.viewer.layers
        if impedance_exists:
            impedance_data = self.viewer.layers["Impedance Image"].data
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Export ROI Table to CSV", filter="CSV Files (*.csv);;All Files (*)"
        )
        if not output_path:
            return
        
        rows = []
        for i, polygon in enumerate(roi_layer.data, start=1):
            polygon = np.array(polygon)
            # Compute centroid approx
            centroid = polygon.mean(axis=0)
            # Compute area with shoelace formula
            x, y = polygon[:, 0], polygon[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            mean_impedance = None
            if impedance_exists:
                # Create mask for polygon on impedance image
                mask = np.zeros(impedance_data.shape[:2], dtype=bool)
                rr, cc = sk_polygon(polygon[:, 0].astype(int), polygon[:, 1].astype(int), mask.shape)
                mask[rr, cc] = True
                if np.any(mask):
                    mean_impedance = np.mean(impedance_data[mask])
                else:
                    mean_impedance = np.nan
            
            row = {
                "ROI_Index": i,
                "Centroid_X": centroid[0],
                "Centroid_Y": centroid[1],
                "Area": area,
            }
            if mean_impedance is not None:
                row["Mean_Impedance"] = mean_impedance
            
            rows.append(row)
        
        # Write CSV
        fieldnames = ["ROI_Index", "Centroid_X", "Centroid_Y", "Area"]
        if impedance_exists:
            fieldnames.append("Mean_Impedance")
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        
        print(f"Exported ROI data for {len(rows)} ROIs to {output_path}")

    def write_polygon_roi(filename, points):
        """
        Write a polygon ROI in ImageJ .roi format.
        points: list of (x,y) tuples
        """
        # ImageJ .roi format specification: https://imagej.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
        
        # Constants
        header_size = 64
        version = 227
        roi_type = 2  # Polygon
        
        n_points = len(points)
        x_coords = [int(round(p[0])) for p in points]
        y_coords = [int(round(p[1])) for p in points]
        
        # Calculate bounding box
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        width = x_max - x_min
        height = y_max - y_min
        
        # Create binary data
        with open(filename, 'wb') as f:
            # Header (64 bytes)
            f.write(b'Iout')                          # 0-3   magic number
            f.write(struct.pack('>h', version))      # 4-5   version (big-endian short)
            f.write(struct.pack('B', roi_type))      # 6     roi type (polygon=2)
            f.write(b'\x00' * 1)                      # 7     unused
            f.write(struct.pack('>h', 0))             # 8-9   top (0 for now)
            f.write(struct.pack('>h', 0))             # 10-11 left (0 for now)
            f.write(struct.pack('>h', width))         # 12-13 width
            f.write(struct.pack('>h', height))        # 14-15 height
            f.write(struct.pack('>h', n_points))      # 16-17 nCoordinates
            f.write(b'\x00' * (64 - 18))              # rest zeros
            
            # Coordinates (each point relative to bounding box)
            for x in x_coords:
                f.write(struct.pack('>h', x - x_min))
            for y in y_coords:
                f.write(struct.pack('>h', y - y_min))
            
            # Footer (optional) - none for simple polygon


    def save_rois_as_zip(self):
        if self.roi_layer_name not in self.viewer.layers:
            print("No ROIs to save.")
            return

        roi_layer = self.viewer.layers[self.roi_layer_name]

        # Prompt user for zip file save location
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save ROIs ZIP", filter="ZIP Files (*.zip);;All Files (*)"
        )
        if not output_path:
            return

        # Ensure .zip extension
        if not output_path.lower().endswith('.zip'):
            output_path += '.zip'

        with zipfile.ZipFile(output_path, 'w') as zipf:
            for i, polygon in enumerate(roi_layer.data, start=1):
                # Flip coordinates: from (row, col) or (y, x) to (x, y)
                polygon_xy = np.flip(polygon, axis=1)

                # Create ImageJ ROI from polygon points
                roi = roifile.ImagejRoi.frompoints(polygon_xy)

                # Get ROI bytes
                roi_bytes = roi.tobytes()

                # Name inside the zip archive
                roi_name = f"roi_{i}.roi"

                # Write ROI binary data to zip
                zipf.writestr(roi_name, roi_bytes)

        print(f"Saved {len(roi_layer.data)} ROIs to {output_path}")



from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton,
    QMessageBox, QGroupBox
)
import numpy as np
from PIL import Image
from cellpose import models

class DeepLearningSegmentationWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        
        self.microscopy_layer_name = "Microscopy Image"
        self.label_layer_name = "Labeled Mask"
        self.roi_layer_name = "ROIs"

        layout = QVBoxLayout()
        self.setLayout(layout)

        # === Preprocessing Group ===
        preprocessing_group = QGroupBox("Preprocessing")
        preprocessing_layout = QVBoxLayout()
        preprocessing_group.setLayout(preprocessing_layout)
        layout.addWidget(preprocessing_group)

        hist_layout = QHBoxLayout()
        hist_label = QLabel("Histogram Norm:")
        hist_label.setToolTip("Adjusts image intensity so brightness is normalized across images.")
        hist_layout.addWidget(hist_label)

        self.hist_btn_r = QPushButton("R")
        self.hist_btn_g = QPushButton("G")
        self.hist_btn_b = QPushButton("B")

        self.hist_btn_r.setToolTip("Normalize Red channel")
        self.hist_btn_g.setToolTip("Normalize Green channel")
        self.hist_btn_b.setToolTip("Normalize Blue channel")

        self.hist_btn_r.clicked.connect(lambda: self.normalize_channel_in_view('red', mode='hist'))
        self.hist_btn_g.clicked.connect(lambda: self.normalize_channel_in_view('green', mode='hist'))
        self.hist_btn_b.clicked.connect(lambda: self.normalize_channel_in_view('blue', mode='hist'))

        hist_layout.addWidget(self.hist_btn_r)
        hist_layout.addWidget(self.hist_btn_g)
        hist_layout.addWidget(self.hist_btn_b)
        preprocessing_layout.addLayout(hist_layout)

        # === Deep Learning Segmentation ===
        dl_group = QGroupBox("Deep Learning Segmentation")
        dl_layout = QVBoxLayout()
        dl_group.setLayout(dl_layout)
        layout.addWidget(dl_group)

        model_label = QLabel("Select Model:")
        model_label.setToolTip("Choose pretrained segmentation model.")
        dl_layout.addWidget(model_label)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["Cellpose"])
        dl_layout.addWidget(self.model_selector)

        flow_label = QLabel("Flow Threshold:")
        flow_label.setToolTip("Confidence threshold for model's predicted flow fields. Default is 0.8.")
        dl_layout.addWidget(flow_label)

        self.input_flow_threshold = QLineEdit("0.8")
        self.input_flow_threshold.setToolTip("Enter a value between 0 and 1.")
        dl_layout.addWidget(self.input_flow_threshold)

        diameter_label = QLabel("Diameter:")
        diameter_label.setToolTip("Estimated average diameter of objects (cells).")
        dl_layout.addWidget(diameter_label)

        self.input_diameter = QLineEdit("65")
        self.input_diameter.setToolTip("Recommended to match average cell size in pixels.")
        dl_layout.addWidget(self.input_diameter)

        self.btn_run = QPushButton("Run Segmentation")
        self.btn_run.setToolTip("Run segmentation using the selected model and parameters.")
        self.btn_run.clicked.connect(self.run_segmentation)
        dl_layout.addWidget(self.btn_run)

        # === ROI Conversion & Processing ===
        roi_group = QGroupBox("ROI Conversion & Processing")
        roi_layout = QVBoxLayout()
        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        self.convert_labels_button = QPushButton("Convert Labels to ROIs")
        self.convert_labels_button.setToolTip("Convert labeled mask to editable ROIs.")
        self.convert_labels_button.clicked.connect(self.labels_to_rois)
        roi_layout.addWidget(self.convert_labels_button)

        self.recompute_button = QPushButton("Recompute Based on ROI Corrections")
        self.recompute_button.setToolTip("Update measurements based on manually corrected ROIs.")
        self.recompute_button.clicked.connect(self.recompute_based_on_corrections)
        roi_layout.addWidget(self.recompute_button)

        # === Impedance Analysis ===
        impedance_group = QGroupBox("Impedance Analysis")
        impedance_layout = QVBoxLayout()
        impedance_group.setLayout(impedance_layout)
        layout.addWidget(impedance_group)

        self.compute_impedance_button = QPushButton("Compute Impedance per ROI")
        self.compute_impedance_button.setToolTip("Measure impedance value inside each segmented ROI.")
        self.compute_impedance_button.clicked.connect(self.calculate_impedance_per_roi)
        impedance_layout.addWidget(self.compute_impedance_button)

        # === Saving & Export ===
        saving_group = QGroupBox("Saving & Export")
        saving_layout = QVBoxLayout()
        saving_group.setLayout(saving_layout)
        layout.addWidget(saving_group)

        self.save_microscopy_button = QPushButton("Save Microscopy Image")
        self.save_microscopy_button.setToolTip("Export the current microscopy image.")
        self.save_microscopy_button.clicked.connect(self.save_microscopy_image)
        saving_layout.addWidget(self.save_microscopy_button)

        export_buttons_layout = QHBoxLayout()

        self.save_rois_zip_button = QPushButton("Save ROIs as ZIP")
        self.save_rois_zip_button.setToolTip("Export all ROIs as a ZIP archive.")
        self.save_rois_zip_button.clicked.connect(self.save_rois_as_zip)
        export_buttons_layout.addWidget(self.save_rois_zip_button)

        self.export_roi_csv_button = QPushButton("Export ROI Table to CSV")
        self.export_roi_csv_button.setToolTip("Export measurements for each ROI as a CSV file.")
        self.export_roi_csv_button.clicked.connect(self.export_roi_data_to_csv)
        export_buttons_layout.addWidget(self.export_roi_csv_button)

        saving_layout.addLayout(export_buttons_layout)

                
    def run_segmentation(self):
        model_name = self.model_selector.currentText()
        
        try:
            flow_th = float(self.input_flow_threshold.text())
            diam = float(self.input_diameter.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numbers for Flow Threshold and Diameter.")
            return
        
        if "Microscopy Image" not in self.viewer.layers:
            QMessageBox.warning(self, "Layer Missing", "No 'Microscopy Image' layer found.")
            return
        
        image_data = self.viewer.layers["Microscopy Image"].data
        
        if image_data.ndim != 3 or image_data.shape[2] < 3:
            QMessageBox.warning(self, "Invalid Image", "Microscopy Image must have at least 3 channels (H, W, C).")
            return
        
        if model_name == "Cellpose":
            cyto = image_data[:, :, 1]
            nuclei = image_data[:, :, 2]
            input_img = np.stack([cyto, nuclei], axis=2)
            
            model = models.CellposeModel(gpu=False, model_type="cyto3")
            
            masks = model.eval(
                input_img,
                diameter=diam,
                flow_threshold=flow_th,
                channels=[0, 1],
                progress=False
            )[0]
            
            # Remove existing Labeled Mask layer if present
            if "Labeled Mask" in self.viewer.layers:
                self.viewer.layers.remove("Labeled Mask")
            
            self.viewer.add_labels(masks.astype(int), name="Labeled Mask")
            
        else:
            QMessageBox.warning(self, "Not Implemented", f"Model {model_name} not implemented.")

    def save_microscopy_image(self):
        if self.microscopy_layer_name not in self.viewer.layers:
            print("No Microscopy Image layer to save.")
            return

        layer = self.viewer.layers[self.microscopy_layer_name]
        image_data = layer.data

        # Open file dialog for saving
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Microscopy Image",
            "",
            "TIFF Files (*.tif *.tiff);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)"
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in ['.tif', '.tiff']:
                tifffile.imwrite(file_path, image_data)
            elif ext in ['.png', '.jpg', '.jpeg']:
                from skimage.io import imsave
                imsave(file_path, image_data)
            else:
                print(f"Unsupported file extension '{ext}'. Saving as TIFF.")
                tifffile.imwrite(file_path, image_data)
            print(f"Microscopy image saved to {file_path}")
        except Exception as e:
            print(f"Failed to save image: {e}")     
    def filter_labels(self):
        if self.label_layer_name not in self.viewer.layers:
            print("No labeled layer to filter.")
            return
        labeled = self.viewer.layers[self.label_layer_name].data
        min_size = self.min_size_spin.value()
        filtered = remove_small_objects(labeled, min_size=min_size)
        self.viewer.layers[self.label_layer_name].data = filtered
        print(f"Filtered out objects smaller than {min_size} pixels.")     
        
    def labels_to_rois(self):
        if self.label_layer_name not in self.viewer.layers:
            print("No labeled layer for ROI extraction.")
            return

        labeled_layer = self.viewer.layers[self.label_layer_name]
        labeled = labeled_layer.data
        roi_shapes = []
        edge_colors = []
        face_colors = []

        get_color = labeled_layer.get_color

        for region in regionprops(labeled):
            binary_mask = labeled == region.label
            contours = find_contours(binary_mask, 0.5)
            if not contours:
                continue
            contour = max(contours, key=len)

            contour_flipped = contour[:, [0, 1]]  # (row, col)

            roi_shapes.append(contour_flipped)

            rgba = get_color(region.label)
            rgba = (rgba[0], rgba[1], rgba[2], 0.75)
            edge_colors.append(rgba)
            face_colors.append(rgba)

        # Remove old ROI layers if any
        if self.roi_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.roi_layer_name)
        if f"{self.roi_layer_name} (Impedance)" in self.viewer.layers:
            self.viewer.layers.remove(f"{self.roi_layer_name} (Impedance)")

        # Add ROI layer aligned with microscopy image (no translation)
        roi_layer = self.viewer.add_shapes(
            data=roi_shapes,
            shape_type="polygon",
            # edge_color=edge_colors,
            face_color=face_colors,
            opacity=1.0,
            name=self.roi_layer_name,
        )
        roi_layer.editable = True

        if "Impedance Image" in self.viewer.layers:
            # Add ROI layer aligned with impedance image (translated)
            impedance_roi_layer = self.viewer.add_shapes(
                data=roi_shapes,
                shape_type="polygon",
                # edge_color=edge_colors,
                face_color=face_colors,
                opacity=1.0,
                name=f"{self.roi_layer_name} (Impedance)",
            )
            impedance_roi_layer.editable = True
            impedance_roi_layer.translate = (0, 576)

        for layer_name in [
            self.label_layer_name,
            f"{self.label_layer_name} (Impedance)",
        ]:
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].visible = False

        print(f"Extracted {len(roi_shapes)} editable ROIs with instance colors on both images.")

    def normalize_channel_in_view(self, color_channel, mode="hist"):
        if self.microscopy_layer_name not in self.viewer.layers:
            print("No microscopy image loaded to normalize.")
            return
        layer = self.viewer.layers[self.microscopy_layer_name]
        img = layer.data.copy()
        if img.ndim == 2:
            print("Microscopy image is grayscale, RGB normalization not applicable.")
            return
        channel_map = {'red': 0, 'green': 1, 'blue': 2}
        ch_idx = channel_map.get(color_channel.lower())
        if ch_idx is None:
            print(f"Unknown channel: {color_channel}")
            return
        img[..., ch_idx] = self.normalize_channel(img[..., ch_idx], mode=mode)
        layer.data = img
        print(f"{color_channel.upper()} channel normalized with {mode}.")


    def normalize_channel(self, channel_data, mode="hist", target_mean=127.5, target_std=40):
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std == 0:
            std = 1
        if mode == "hist":
            normalized = (channel_data - mean) / std
            normalized = normalized * target_std + target_mean
        elif mode == "zscore":
            normalized = (channel_data - mean) / std
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min()) * 255
        else:
            print(f"Unknown normalization mode: {mode}")
            return channel_data
        return np.clip(normalized, 0, 255).astype(np.uint8)    
        
    def calculate_impedance_per_roi(self):
        if self.label_layer_name not in self.viewer.layers:
            print("No labeled layer found.")
            return

        if "Impedance Image" not in self.viewer.layers:
            print("No impedance image layer found.")
            return

        label_data = self.viewer.layers[self.label_layer_name].data
        impedance_data = self.viewer.layers["Impedance Image"].data

        if label_data.shape != impedance_data.shape:
            print("Shape mismatch between labels and impedance image.")
            return

        text_points = []
        impedance_strings = []
        index_strings = []

        for region in regionprops(label_data, intensity_image=impedance_data):
            if region.area == 0:
                continue
            mean_val = region.mean_intensity
            centroid = region.centroid  # (row, col)

            x, y = centroid[0], centroid[1] + 576
            text_points.append((x, y))
            impedance_strings.append(f"{mean_val:.2f}")
            index_strings.append(str(region.label))

        # Calculate background mask and mean impedance
        background_mask = label_data == 0
        mean_background_val = np.mean(impedance_data[background_mask])

        # Position for background text: below image center + offset
        y_bkg = impedance_data.shape[0] + 30  # 30 pixels below image bottom
        x_bkg = impedance_data.shape[1] // 2 + 576

        # Remove old text layers
        for layer_name in ["Mean Impedance Text", "Instance Index Text", "Background Impedance Text"]:
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)

        # Add impedance values for ROIs
        self.viewer.add_points(
            text_points,
            name="Mean Impedance Text",
            face_color="transparent",
            #edge_color="white",
            size=1,
            text={
                "string": impedance_strings,
                "size": 12,
                "color": "white",
                "anchor": "center",
            },
        )

        # Add background impedance text below image
        self.viewer.add_points(
            [(y_bkg, x_bkg)],
            name="Background Impedance Text",
            face_color="transparent",
            #edge_color="white",
            size=1,
            text={
                "string": [f"Background Impedance : {mean_background_val:.2f}"],
                "size": 12,
                "color": "yellow",
                "anchor": "center",
            },
        )

        print(f"Displayed mean impedance and index for {len(text_points)} instances plus background.")


    def recompute_based_on_corrections(self):
        self.correct_masks_from_rois()
        # Check if microscopy ROI layer exists
        if self.roi_layer_name not in self.viewer.layers:
            print("No microscopy ROIs found to recompute.")
            return

        # Remove the impedance ROI layer and impedance text layers if they exist
        impedance_roi_name = f"{self.roi_layer_name} (Impedance)"
        for layer_name in [impedance_roi_name, "Mean Impedance Text", "Instance Index Text"]:
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)

        # Copy microscopy ROIs and translate them to impedance coordinates
        microscopy_roi_layer = self.viewer.layers[self.roi_layer_name]

        # Copy shapes data and colors
        roi_shapes = microscopy_roi_layer.data.copy()
        edge_colors = microscopy_roi_layer.edge_color.copy()
        face_colors = microscopy_roi_layer.face_color.copy()

        if "Impedance Image" not in self.viewer.layers:
            print("No impedance image layer found.")
            return

        # Add new impedance ROI layer with translation (0, 576)
        impedance_roi_layer = self.viewer.add_shapes(
            data=roi_shapes,
            shape_type="polygon",
            # edge_color=edge_colors,
            face_color=face_colors,
            opacity=1.0,
            name=impedance_roi_name,
        )
        impedance_roi_layer.editable = True
        impedance_roi_layer.translate = (0, 576)

        # Now recompute impedance per ROI for the impedance layer
        self.calculate_impedance_per_roi_from_shapes(impedance_roi_layer)

        print("Recomputed impedance ROIs and values based on microscopy corrections.")

    def calculate_impedance_per_roi_from_shapes(self, roi_layer):
        if "Impedance Image" not in self.viewer.layers:
            print("No impedance image layer found.")
            return

        impedance_data = self.viewer.layers["Impedance Image"].data

        text_points = []
        impedance_strings = []

        # Process instance ROIs first
        for shape in roi_layer.data:
            mask = np.zeros(impedance_data.shape[:2], dtype=bool)
            polygon = shape.astype(np.int32)
            rr, cc = sk_polygon(polygon[:, 0], polygon[:, 1], mask.shape)
            mask[rr, cc] = True

            mean_val = np.mean(impedance_data[mask]) if np.any(mask) else 0.0
            centroid = np.mean(polygon, axis=0)
            y, x = centroid
            text_points.append((y, x + 576))
            impedance_strings.append(f"{mean_val:.2f}")

        # Remove old text layers
        if "Mean Impedance Text" in self.viewer.layers:
            self.viewer.layers.remove("Mean Impedance Text")

        # Add ROI impedance text points
        self.viewer.add_points(
            text_points,
            name="Mean Impedance Text",
            face_color="transparent",
            #edge_color="white",
            size=1,
            text={
                "string": impedance_strings,
                "size": 12,
                "color": "white",
                "anchor": "center",
            },
        )

        # --- Now handle background separately ---

        # Create combined mask of all ROIs
        full_mask = np.zeros(impedance_data.shape[:2], dtype=bool)
        for shape in roi_layer.data:
            polygon = shape.astype(np.int32)
            rr, cc = sk_polygon(polygon[:, 0], polygon[:, 1], full_mask.shape)
            full_mask[rr, cc] = True

        background_mask = ~full_mask

        mean_background_val = np.mean(impedance_data[background_mask])

        # Place background label **below the image**:
        # image height = impedance_data.shape[0]
        y_bkg = impedance_data.shape[0] + 30  # 30 pixels below bottom edge
        x_bkg = impedance_data.shape[1] // 2 + 576  # centered horizontally + offset

        # Remove old background label if any
        if "Background Impedance Text" in self.viewer.layers:
            self.viewer.layers.remove("Background Impedance Text")

        self.viewer.add_points(
            [(y_bkg, x_bkg)],
            name="Background Impedance Text",
            face_color="transparent",
            #edge_color="white",
            size=1,
            text={
                "string": [f"Background Impedance : {mean_background_val:.2f}"],
                "size": 12,
                "color": "yellow",
                "anchor": "center",
            },
        )

    def correct_masks_from_rois(self):
        from skimage.draw import polygon as sk_polygon

        if self.roi_layer_name not in self.viewer.layers:
            print("No ROIs found.")
            return

        roi_layer = self.viewer.layers[self.roi_layer_name]
        img_shape = self.viewer.layers[self.microscopy_layer_name].data.shape[:2]

        # 1. Create binary mask from polygons
        mask = np.zeros(img_shape, dtype=bool)
        for shape in roi_layer.data:
            poly = shape.astype(np.int32)
            rr, cc = sk_polygon(poly[:, 0], poly[:, 1], img_shape)
            mask[rr, cc] = True

        # 2. Replace Segmentation Mask layer
        if self.mask_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.mask_layer_name)
        if f"{self.mask_layer_name} (Impedance)" in self.viewer.layers:
            self.viewer.layers.remove(f"{self.mask_layer_name} (Impedance)")

        self.viewer.add_labels(mask.astype(np.uint8), name=self.mask_layer_name)
        imp_mask_layer = self.viewer.add_labels(mask.astype(np.uint8), name=f"{self.mask_layer_name} (Impedance)")
        imp_mask_layer.translate = (0, 576)

        # 3. Relabel the new mask
        self.label_mask()

        for layer_name in [
            "Labeled Mask",
            "Labeled Mask (Impedance)"
        ]:
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].visible = False

        print("Corrected masks and ROIs from edited ROI layer.")

    def save_rois_to_csv(self):
        if self.roi_layer_name not in self.viewer.layers:
            print("No ROIs to save.")
            return
        
        roi_layer = self.viewer.layers[self.roi_layer_name]
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save ROIs CSV", filter="CSV Files (*.csv);;All Files (*)"
        )
        if not output_path:
            return
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ROI_Index', 'Polygon_Vertices'])
            for i, polygon in enumerate(roi_layer.data, start=1):
                # Flatten polygon points into a string "x1,y1;x2,y2;..."
                vertices_str = ";".join([f"{x:.2f},{y:.2f}" for x, y in polygon])
                writer.writerow([i, vertices_str])
        
        print(f"Saved {len(roi_layer.data)} ROIs to {output_path}")

    def export_roi_data_to_csv(self):
        if self.roi_layer_name not in self.viewer.layers:
            print("No ROIs found for export.")
            return

        roi_layer = self.viewer.layers[self.roi_layer_name]
        impedance_exists = "Impedance Image" in self.viewer.layers
        if impedance_exists:
            impedance_data = self.viewer.layers["Impedance Image"].data
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Export ROI Table to CSV", filter="CSV Files (*.csv);;All Files (*)"
        )
        if not output_path:
            return
        
        rows = []
        for i, polygon in enumerate(roi_layer.data, start=1):
            polygon = np.array(polygon)
            # Compute centroid approx
            centroid = polygon.mean(axis=0)
            # Compute area with shoelace formula
            x, y = polygon[:, 0], polygon[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            mean_impedance = None
            if impedance_exists:
                # Create mask for polygon on impedance image
                mask = np.zeros(impedance_data.shape[:2], dtype=bool)
                rr, cc = sk_polygon(polygon[:, 0].astype(int), polygon[:, 1].astype(int), mask.shape)
                mask[rr, cc] = True
                if np.any(mask):
                    mean_impedance = np.mean(impedance_data[mask])
                else:
                    mean_impedance = np.nan
            
            row = {
                "ROI_Index": i,
                "Centroid_X": centroid[0],
                "Centroid_Y": centroid[1],
                "Area": area,
            }
            if mean_impedance is not None:
                row["Mean_Impedance"] = mean_impedance
            
            rows.append(row)
        
        # Write CSV
        fieldnames = ["ROI_Index", "Centroid_X", "Centroid_Y", "Area"]
        if impedance_exists:
            fieldnames.append("Mean_Impedance")
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        
        print(f"Exported ROI data for {len(rows)} ROIs to {output_path}")

    def write_polygon_roi(filename, points):
        """
        Write a polygon ROI in ImageJ .roi format.
        points: list of (x,y) tuples
        """
        # ImageJ .roi format specification: https://imagej.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
        
        # Constants
        header_size = 64
        version = 227
        roi_type = 2  # Polygon
        
        n_points = len(points)
        x_coords = [int(round(p[0])) for p in points]
        y_coords = [int(round(p[1])) for p in points]
        
        # Calculate bounding box
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        width = x_max - x_min
        height = y_max - y_min
        
        # Create binary data
        with open(filename, 'wb') as f:
            # Header (64 bytes)
            f.write(b'Iout')                          # 0-3   magic number
            f.write(struct.pack('>h', version))      # 4-5   version (big-endian short)
            f.write(struct.pack('B', roi_type))      # 6     roi type (polygon=2)
            f.write(b'\x00' * 1)                      # 7     unused
            f.write(struct.pack('>h', 0))             # 8-9   top (0 for now)
            f.write(struct.pack('>h', 0))             # 10-11 left (0 for now)
            f.write(struct.pack('>h', width))         # 12-13 width
            f.write(struct.pack('>h', height))        # 14-15 height
            f.write(struct.pack('>h', n_points))      # 16-17 nCoordinates
            f.write(b'\x00' * (64 - 18))              # rest zeros
            
            # Coordinates (each point relative to bounding box)
            for x in x_coords:
                f.write(struct.pack('>h', x - x_min))
            for y in y_coords:
                f.write(struct.pack('>h', y - y_min))
            
            # Footer (optional) - none for simple polygon


    def save_rois_as_zip(self):
        if self.roi_layer_name not in self.viewer.layers:
            print("No ROIs to save.")
            return

        roi_layer = self.viewer.layers[self.roi_layer_name]

        # Prompt user for zip file save location
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save ROIs ZIP", filter="ZIP Files (*.zip);;All Files (*)"
        )
        if not output_path:
            return

        # Ensure .zip extension
        if not output_path.lower().endswith('.zip'):
            output_path += '.zip'

        with zipfile.ZipFile(output_path, 'w') as zipf:
            for i, polygon in enumerate(roi_layer.data, start=1):
                # Flip coordinates: from (row, col) or (y, x) to (x, y)
                polygon_xy = np.flip(polygon, axis=1)

                # Create ImageJ ROI from polygon points
                roi = roifile.ImagejRoi.frompoints(polygon_xy)

                # Get ROI bytes
                roi_bytes = roi.tobytes()

                # Name inside the zip archive
                roi_name = f"roi_{i}.roi"

                # Write ROI binary data to zip
                zipf.writestr(roi_name, roi_bytes)

        print(f"Saved {len(roi_layer.data)} ROIs to {output_path}")
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QGroupBox,
    QComboBox, QProgressBar, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal
import os


from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QGroupBox,
    QProgressBar, QCheckBox
)
from PyQt5.QtCore import Qt


class FinetuningSegmentationModelWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", dataset_size: int = 100):
        super().__init__()
        self.viewer = viewer
        self.dataset_size = dataset_size  # Number of samples in dataset for time estimate

        layout = QVBoxLayout()

        # === Hyperparameter Inputs ===
        hyperparams_box = QGroupBox("Training Parameters")
        hyperparams_layout = QHBoxLayout()

        self.epochs_input = QLineEdit("10")
        self.epochs_input.setToolTip(
            "Number of training passes through your dataset. Start with 50-100. "
            "Higher epochs increase training time linearly."
        )
        self.epochs_input.textChanged.connect(self.update_time_estimate)

        self.learning_rate_input = QLineEdit("0.01")
        self.learning_rate_input.setToolTip(
            "Controls how much the model updates per step. Lower values mean slower, more stable training."
        )

        label_epochs = QLabel("Epochs:")
        label_epochs.setToolTip(self.epochs_input.toolTip())

        label_lr = QLabel("Learning Rate:")
        label_lr.setToolTip(self.learning_rate_input.toolTip())

        hyperparams_layout.addWidget(label_epochs)
        hyperparams_layout.addWidget(self.epochs_input)
        hyperparams_layout.addWidget(label_lr)
        hyperparams_layout.addWidget(self.learning_rate_input)

        hyperparams_box.setLayout(hyperparams_layout)
        layout.addWidget(hyperparams_box)

        # === Training Time Estimate ===
        self.time_estimate_label = QLabel("Estimated Training Time: Unknown")
        self.time_estimate_label.setToolTip(
            "Estimated training time based on dataset size and epochs."
        )
        layout.addWidget(self.time_estimate_label)
        self.update_time_estimate()

        # === Preset Training Options ===
        preset_box = QGroupBox("Preset Training Options")
        preset_layout = QHBoxLayout()

        btn_fast = QPushButton("Fast")
        btn_fast.setToolTip("Quick training: fewer epochs, higher learning rate.")
        btn_fast.clicked.connect(lambda: self.apply_preset(epochs=10, lr=0.05))

        btn_balanced = QPushButton("Balanced")
        btn_balanced.setToolTip("Balanced training: moderate epochs and learning rate.")
        btn_balanced.clicked.connect(lambda: self.apply_preset(epochs=50, lr=0.01))

        btn_steady = QPushButton("Steady")
        btn_steady.setToolTip("Longer training: more epochs, lower learning rate for stability.")
        btn_steady.clicked.connect(lambda: self.apply_preset(epochs=100, lr=0.001))

        preset_layout.addWidget(btn_fast)
        preset_layout.addWidget(btn_balanced)
        preset_layout.addWidget(btn_steady)

        preset_box.setLayout(preset_layout)
        layout.addWidget(preset_box)

        # === Channel Selection ===
        channel_box = QGroupBox("Training Channels")
        channel_layout = QVBoxLayout()

        self.checkbox_green = QCheckBox("Green (e.g., CMFDA)")
        self.checkbox_green.setToolTip(
            "Use the green channel (commonly CellTracker Green CMFDA) as input for training."
        )

        self.checkbox_blue = QCheckBox("Blue (e.g., Hoechst)")
        self.checkbox_blue.setToolTip(
            "Use the blue channel (e.g., Hoechst for nuclei) for training."
        )

        self.checkbox_impedance = QCheckBox("Impedance (Red)")
        self.checkbox_impedance.setToolTip(
            "Use impedance imaging (mapped to red) as a structural or contextual input."
        )

        channel_layout.addWidget(self.checkbox_green)
        channel_layout.addWidget(self.checkbox_blue)
        channel_layout.addWidget(self.checkbox_impedance)

        channel_box.setLayout(channel_layout)
        layout.addWidget(channel_box)

        # === Progress Bar ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setToolTip("Shows training progress once started.")
        layout.addWidget(self.progress_bar)

        # === Upload to Drive ===
        self.uploader = GoogleDriveUploader()
        self.uploader.setToolTip("Upload your trained model to Google Drive for backup or sharing.")
        layout.addWidget(self.uploader)

        # === Download Trained Model ===
        self.download_button = QPushButton("Download Trained Model")
        self.download_button.setToolTip("Download your trained model from Google Drive.")
        self.download_button.clicked.connect(self.download_model_from_drive)
        layout.addWidget(self.download_button)

        self.setLayout(layout)

    def update_time_estimate(self):
        try:
            epochs = int(self.epochs_input.text())
            # Estimate time in seconds, for example 0.5 sec per sample per epoch
            seconds = self.dataset_size * epochs * 0.5
            minutes = seconds / 60
            if minutes < 1:
                time_str = f"{seconds:.1f} seconds"
            else:
                time_str = f"{minutes:.1f} minutes"
            self.time_estimate_label.setText(f"Estimated Training Time: {time_str}")
        except ValueError:
            self.time_estimate_label.setText("Estimated Training Time: Invalid epochs input")

    def apply_preset(self, epochs: int, lr: float):
        self.epochs_input.setText(str(epochs))
        self.learning_rate_input.setText(str(lr))
        self.update_time_estimate()


    def download_model_from_drive(self):
        """Dummy placeholder to download the trained model file from Google Drive."""
        try:
            model_file_name = "cellpose_model.pth"
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Trained Model", model_file_name)
            if save_path:
                threading.Thread(target=self._download_thread, args=(model_file_name, save_path), daemon=True).start()
        except Exception as e:
            QMessageBox.critical(self, "Download Failed", str(e))

    def _download_thread(self, file_name, save_path):
        try:
            drive_service = self.uploader.authenticate_drive()
            folder_id = self.uploader.drive_folder_id

            query = f"name='{file_name}' and '{folder_id}' in parents"
            result = drive_service.files().list(q=query, fields="files(id, name)").execute()
            files = result.get("files", [])
            if not files:
                self._show_message_box("Model Not Found", "No trained model found in Google Drive.", QMessageBox.Warning)
                return

            file_id = files[0]["id"]
            request = drive_service.files().get_media(fileId=file_id)
            with open(save_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()

            self._show_message_box("Download Complete", f"Model saved to:\n{save_path}", QMessageBox.Information)
        except Exception as e:
            self._show_message_box("Download Error", str(e), QMessageBox.Critical)

    def _show_message_box(self, title, message, icon):
        def show():
            QMessageBox(icon, title, message).exec_()
        QApplication.instance().postEvent(self, QEvent(QEvent.User))
        QMetaObject.invokeMethod(self, show, Qt.QueuedConnection)


class GoogleDriveUploader(QGroupBox):
    upload_finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('Finetuning Model')
        # self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        # self.setStyleSheet('QGroupBox {background-color: lightgray; border-radius: 10px;}')

        self.drive_folder_id = "1_ungUq9yOaKqKMxlRr-mfMC7f0I_018J"  # <- Update as needed

        self.upload_finished.connect(self.on_upload_finished)

        vbox = QVBoxLayout()
        self.setLayout(vbox)

        vbox.addWidget(QLabel('Dataset Folder Path:'))
        self.dataset_path_input = QLineEdit()
        vbox.addWidget(self.dataset_path_input)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_dataset_path)
        vbox.addWidget(browse_button)

        self.btn_upload = QPushButton('Finetune Model')
        self.btn_upload.clicked.connect(self.start_upload)
        vbox.addWidget(self.btn_upload)

    def browse_dataset_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if path:
            self.dataset_path_input.setText(path)

    def authenticate_drive(self):
        SCOPES = ['https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_file(
            "/Users/prateekgrover/Downloads/celldynamicsplatform-2de8aed90a79.json",
            scopes=SCOPES
        )
        return build('drive', 'v3', credentials=creds)

    def get_folder_id(self, parent_folder_id, folder_name):
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_folder_id}' in parents"
        response = self.drive_service.files().list(q=query, fields="files(id)").execute()

        if response.get('files'):
            return response['files'][0]['id']

        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }
        folder = self.drive_service.files().create(body=folder_metadata, fields='id').execute()
        return folder['id']

    def upload_file(self, file_path, parent_folder_id):
        file_metadata = {'name': os.path.basename(file_path), 'parents': [parent_folder_id]}
        media = MediaFileUpload(file_path, resumable=True)
        self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    def upload_folder(self, local_folder, drive_parent_folder_id):
        for root, dirs, files in os.walk(local_folder):
            relative_path = os.path.relpath(root, local_folder)
            current_drive_folder_id = (
                drive_parent_folder_id if relative_path == "." else self.get_folder_id(drive_parent_folder_id, relative_path)
            )

            for file in files:
                file_path = os.path.join(root, file)
                self.upload_file(file_path, current_drive_folder_id)

        # Upload the checkpoint file
        done_file_path = os.path.join(local_folder, "upload_complete.txt")
        with open(done_file_path, "w") as f:
            f.write("Upload complete.")
        self.upload_file(done_file_path, drive_parent_folder_id)

    def start_upload(self):
        dataset_path = self.dataset_path_input.text().strip()
        if not dataset_path or not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Error", "Invalid dataset path! Please enter a valid folder.")
            return

        self.btn_upload.setEnabled(False)
        threading.Thread(target=self.upload_process, args=(dataset_path,), daemon=True).start()

    def upload_process(self, dataset_path):
        try:
            self.drive_service = self.authenticate_drive()
            self.upload_folder(dataset_path, self.drive_folder_id)
            self.upload_finished.emit(True, "Dataset uploaded successfully!")
        except Exception as e:
            self.upload_finished.emit(False, f"Error: {str(e)}")

    def on_upload_finished(self, success, message):
        self.btn_upload.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Upload Failed", message)



from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QDoubleSpinBox, QComboBox
)
import numpy as np
from skimage.io import imread
from roifile import roiread
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import csv
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from skimage.draw import polygon2mask

class CellToCellAssociationWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.setMaximumWidth(360)
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        self.rois = {1: [], 2: []}
        self.centroids = {1: [], 2: []}
        self.eccentricities = {1: [], 2: []}
        self.areas = {1: [], 2: []}
        self.mappings = []  # (i1, i2)
        self.confidences = []  # Confidence scores per mapping

        self.layout().addWidget(QLabel("<h3>Compare ROIs at Two Timepoints</h3>"))
        self.layout().addWidget(QLabel("<b>Step 1: Load images and ROIs</b>"))

        self.layout().addWidget(self._file_load_row("Image @ T1", lambda: self.load_image(1)))
        self.layout().addWidget(self._file_load_row("ROI @ T1", lambda: self.load_rois(1)))
        self.layout().addWidget(self._file_load_row("Image @ T2", lambda: self.load_image(2)))
        self.layout().addWidget(self._file_load_row("ROI @ T2", lambda: self.load_rois(2)))

        self.layout().addSpacing(10)
        self.layout().addWidget(QLabel("<b>Step 2: Matching Strategy</b>"))

        # Use vertical layout to stack each spinbox row
        weight_layout = QVBoxLayout()

        # Centroid weight
        centroid_row = QHBoxLayout()
        self.w_centroid = self._make_spinbox("Centroid", centroid_row)
        weight_layout.addLayout(centroid_row)

        # Area weight
        area_row = QHBoxLayout()
        self.w_area = self._make_spinbox("Area", area_row)
        weight_layout.addLayout(area_row)

        # Eccentricity weight
        ecc_row = QHBoxLayout()
        self.w_ecc = self._make_spinbox("Eccentricity", ecc_row)
        weight_layout.addLayout(ecc_row)

        self.layout().addLayout(weight_layout)

        self.match_btn = QPushButton("Match ROIs")
        self.match_btn.clicked.connect(self.match_rois)
        self.layout().addWidget(self.match_btn)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.layout().addWidget(self.info_label)

        self.layout().addSpacing(10)
        self.layout().addWidget(QLabel("<b>Manual Matching</b>"))

        manual_row = QHBoxLayout()
        self.roi1_combo = QComboBox()
        self.roi2_combo = QComboBox()
        manual_row.addWidget(QLabel("T1:"))
        manual_row.addWidget(self.roi1_combo)
        manual_row.addWidget(QLabel("T2:"))
        manual_row.addWidget(self.roi2_combo)
        self.layout().addLayout(manual_row)

        manual_btns = QHBoxLayout()
        self.manual_match_btn = QPushButton("Match")
        self.manual_unmatch_btn = QPushButton("Unmatch")
        self.manual_match_btn.clicked.connect(self.manual_match)
        self.manual_unmatch_btn.clicked.connect(self.manual_unmatch)
        manual_btns.addWidget(self.manual_match_btn)
        manual_btns.addWidget(self.manual_unmatch_btn)
        self.layout().addLayout(manual_btns)

        self.save_btn = QPushButton("Save ROI Mappings")
        self.save_btn.clicked.connect(self.save_mappings)
        self.layout().addWidget(self.save_btn)

    def _make_spinbox(self, name, layout):
        label = QLabel(f"{name}:")
        spin = QDoubleSpinBox()
        spin.setRange(0, 10)
        spin.setSingleStep(0.1)
        spin.setValue(1.0 if name == "Centroid" else 0.0)
        layout.addWidget(label)
        layout.addWidget(spin)
        return spin

    def _file_load_row(self, text, fn):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove padding around the layout
        layout.setSpacing(5)  # Optional: reduce horizontal spacing between label and button
        layout.addWidget(QLabel(text))
        btn = QPushButton("Browse")
        btn.clicked.connect(fn)
        layout.addWidget(btn)
        return row

    def load_image(self, idx):
        path, _ = QFileDialog.getOpenFileName(self, f"Image {idx}", filter="Images (*.tif *.png *.jpg)")
        if not path:
            return
        img = imread(path)
        layer = self.viewer.add_image(img, name=f"Image {idx}")
        if idx == 2:
            layer.translate = (0, 576)

    def load_rois(self, idx):
        path, _ = QFileDialog.getOpenFileName(self, f"ROI Zip {idx}", filter="ZIP Files (*.zip)")
        if not path:
            return
        try:
            rois = roiread(path)
        except Exception as e:
            print(f"Error reading ROIs: {e}")
            return

        polys, cents, eccs, areas = [], [], [], []
        for roi in rois:
            coords = np.flip(roi.coordinates(), axis=1)
            if idx == 2:
                coords[:, 1] += 576
            polys.append(coords)
            cents.append(np.mean(coords, axis=0))
            eccs.append(self.get_eccentricity(coords))
            areas.append(self.polygon_area(coords))

        self.rois[idx] = polys
        self.centroids[idx] = cents
        self.eccentricities[idx] = eccs
        self.areas[idx] = areas

        # Remove old layers first if any
        for name in [f"ROIs {idx}", f"Centroids {idx}", f"ROI Indices {idx}"]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])

        self.viewer.add_shapes(polys, shape_type="polygon",
                               edge_color="red" if idx == 1 else "blue",
                               name=f"ROIs {idx}", opacity=0.4)
        self.viewer.add_points(cents, name=f"Centroids {idx}",
                               face_color="red" if idx == 1 else "blue",
                               size=6)

        self.draw_roi_indices(idx)
        self.update_roi_combos()

    def draw_roi_indices(self, idx):
        # Remove old ROI indices layer if exists
        name = f"ROI Indices {idx}"
        if name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[name])

        points = np.array(self.centroids[idx])
        texts = [str(i + 1) for i in range(len(points))]

        self.viewer.add_points(points, name=name,
                               face_color=[0, 0, 0, 0],
                               size=12, text = texts)

    def get_eccentricity(self, coords):
        mask = polygon2mask((1024, 1024), coords)  # Adjust to your image size if needed
        props = regionprops(mask.astype(int))
        if props:
            return props[0].eccentricity
        else:
            return 0

    def polygon_area(self, poly):
        x = poly[:, 1]
        y = poly[:, 0]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def update_roi_combos(self):
        self.roi1_combo.clear()
        self.roi2_combo.clear()
        self.roi1_combo.addItems([str(i + 1) for i in range(len(self.rois[1]))])
        self.roi2_combo.addItems([str(i + 1) for i in range(len(self.rois[2]))])

    def compute_confidences(self, cost, row_ind, col_ind):
        matched_costs = cost[row_ind, col_ind]
        epsilon = 1e-6
        confidences = 0.5 / (matched_costs + epsilon)
        return confidences

    def match_rois(self):
        w_cent = self.w_centroid.value()
        w_area = self.w_area.value()
        w_ecc = self.w_ecc.value()
        sum = w_cent + w_area + w_ecc
        w_cent = w_cent/sum
        w_area = w_area/sum
        w_ecc = w_ecc/sum

        c1 = np.array(self.centroids[1])
        c2 = np.array(self.centroids[2]) - (0, 576)
        a1 = np.array(self.areas[1])
        a2 = np.array(self.areas[2])
        e1 = np.array(self.eccentricities[1])
        e2 = np.array(self.eccentricities[2])

        cd = distance_matrix(c1, c2)
        ad = distance_matrix(a1[:, None], a2[:, None])
        ed = distance_matrix(e1[:, None], e2[:, None])

        cost = w_cent * cd/50.0 + w_area * ad/20.0 + w_ecc * ed

        self.mappings.clear()
        row_ind, col_ind = linear_sum_assignment(cost)
        self.mappings = list(zip(row_ind, col_ind))
        self.confidences = self.compute_confidences(cost, row_ind, col_ind)

        self.redraw_matches()
        self.update_info_label(cost, row_ind, col_ind)

    def update_info_label(self, cost, row_ind, col_ind):
        matched_1 = set(row_ind)
        matched_2 = set(col_ind)

        unmatched_1 = set(range(len(self.rois[1]))) - matched_1
        unmatched_2 = set(range(len(self.rois[2]))) - matched_2

        confidences = self.compute_confidences(cost, row_ind, col_ind)

        # Sort matches by confidence ascending (least confident first)
        sorted_indices = np.argsort(confidences)
        low_conf_count = min(3, len(confidences))
        low_conf_matches = []
        for idx in sorted_indices[:low_conf_count]:
            i1, i2 = row_ind[idx], col_ind[idx]
            conf = confidences[idx]
            low_conf_matches.append(f"ROI T1 #{i1 + 1} ⇔ ROI T2 #{i2 + 1} (Conf: {conf:.2f})")

        info_lines = []
        if unmatched_1:
            info_lines.append(f"Unmatched ROIs at T1: {sorted([i + 1 for i in unmatched_1])}")
        else:
            info_lines.append("All ROIs at T1 matched.")
        if unmatched_2:
            info_lines.append(f"Unmatched ROIs at T2: {sorted([i + 1 for i in unmatched_2])}")
        else:
            info_lines.append("All ROIs at T2 matched.")

        if low_conf_matches:
            info_lines.append("\nLeast confident matches:")
            info_lines.extend(low_conf_matches)

        self.info_label.setText("\n".join(info_lines))

    def redraw_matches(self):
        for name in ["Matched ROIs T1", "Matched ROIs T2"]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])

        polys1 = [self.rois[1][i] for i, _ in self.mappings]
        polys2 = [self.rois[2][j] for _, j in self.mappings]
        colors = [plt.cm.tab20(i % 20) for i in range(len(self.mappings))]  # RGBA floats

        colors_array = np.array(colors)  # shape (N,4)

        self.viewer.add_shapes(polys1, shape_type="polygon",
                            face_color=colors_array,
                            name="Matched ROIs T1", opacity=0.7)
        self.viewer.add_shapes(polys2, shape_type="polygon",
                            face_color=colors_array,
                            name="Matched ROIs T2", opacity=0.7)

        self.draw_roi_indices(1)
        self.draw_roi_indices(2)

    def rgb_to_hex(self, rgb):
        r, g, b, _ = [int(255 * c) for c in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"

    def manual_match(self):
        i1 = self.roi1_combo.currentIndex()  # T1
        i2 = self.roi2_combo.currentIndex()  # T2
        if i1 < 0 or i2 < 0:
            return


        # Add new mapping
        self.mappings.append((i1, i2))

        # Update confidences - setting 1.0 for manual match
        self.confidences = list(self.confidences)
        self.confidences.append(1.0)
        self.confidences = np.array(self.confidences)

        self.redraw_matches()
        self.update_info_label_after_manual()

    def manual_unmatch(self):
        i1 = self.roi1_combo.currentIndex()
        i2 = self.roi2_combo.currentIndex()
        if i1 < 0 or i2 < 0:
            return
        before = len(self.mappings)
        self.mappings = [(a, b) for (a, b) in self.mappings if not (a == i1 and b == i2)]
        after = len(self.mappings)
        if before == after:
            return  # Nothing removed

        # Remove corresponding confidence if possible
        # Since no direct index, just reset confidences completely from current matches
        self.recompute_confidences_from_mappings()

        self.redraw_matches()
        self.update_info_label_after_manual()

    def recompute_confidences_from_mappings(self):
        if not self.mappings:
            self.confidences = []
            return
        w_cent = self.w_centroid.value()
        w_area = self.w_area.value()
        w_ecc = self.w_ecc.value()

        c1 = np.array(self.centroids[1])
        c2 = np.array(self.centroids[2]) - (0, 576)
        a1 = np.array(self.areas[1])
        a2 = np.array(self.areas[2])
        e1 = np.array(self.eccentricities[1])
        e2 = np.array(self.eccentricities[2])

        cd = distance_matrix(c1, c2)
        ad = distance_matrix(a1[:, None], a2[:, None])
        ed = distance_matrix(e1[:, None], e2[:, None])
        cost = w_cent * cd/50.0 + w_area * ad/20.0 + w_ecc * ed

        rows, cols = zip(*self.mappings)
        self.confidences = self.compute_confidences(cost, np.array(rows), np.array(cols))

    def update_info_label_after_manual(self):
        if not self.mappings:
            self.info_label.setText("No matches currently.")
            return

        # Build a cost matrix from current ROIs and weights
        w_cent = self.w_centroid.value()
        w_area = self.w_area.value()
        w_ecc = self.w_ecc.value()
        sum = w_cent + w_area + w_ecc
        w_cent = w_cent/sum
        w_area = w_area/sum
        w_ecc = w_ecc/sum

        c1 = np.array(self.centroids[1])
        c2 = np.array(self.centroids[2])
        a1 = np.array(self.areas[1])
        a2 = np.array(self.areas[2])
        e1 = np.array(self.eccentricities[1])
        e2 = np.array(self.eccentricities[2])

        cd = distance_matrix(c1, c2)
        ad = distance_matrix(a1[:, None], a2[:, None])
        ed = distance_matrix(e1[:, None], e2[:, None])
        cost = w_cent * cd + w_area * ad + w_ecc * ed

        matched_1 = set([m[0] for m in self.mappings])
        matched_2 = set([m[1] for m in self.mappings])

        unmatched_1 = set(range(len(self.rois[1]))) - matched_1
        unmatched_2 = set(range(len(self.rois[2]))) - matched_2

        self.recompute_confidences_from_mappings()
        confidences = self.confidences

        # Sort matches by confidence ascending
        if len(confidences) > 0:
            sorted_indices = np.argsort(confidences)
            low_conf_count = min(3, len(confidences))
            low_conf_matches = []
            for idx in sorted_indices[:low_conf_count]:
                i1, i2 = self.mappings[idx]
                conf = confidences[idx]
                low_conf_matches.append(f"ROI T1 #{i1 + 1} ⇔ ROI T2 #{i2 + 1} (Conf: {conf:.2f})")
        else:
            low_conf_matches = []

        info_lines = []
        if unmatched_1:
            info_lines.append(f"Unmatched ROIs at T1: {sorted([i + 1 for i in unmatched_1])}")
        else:
            info_lines.append("All ROIs at T1 matched.")
        if unmatched_2:
            info_lines.append(f"Unmatched ROIs at T2: {sorted([i + 1 for i in unmatched_2])}")
        else:
            info_lines.append("All ROIs at T2 matched.")

        if low_conf_matches:
            info_lines.append("\nLeast confident matches:")
            info_lines.extend(low_conf_matches)

        self.info_label.setText("\n".join(info_lines))

    def save_mappings(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI Mapping CSV", filter="CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith('.csv'):
            path += '.csv'

        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'ROI 1 Index', 'ROI 2 Index',
                    'Centroid 1 (Y, X)', 'Centroid 2 (Y, X)',
                    'Area 1', 'Area 2'
                ])
                for i1, i2 in self.mappings:
                    if not isinstance(i2, list):
                        i2 = [i2]
                    for single_i2 in i2:
                        c1 = self.centroids[1][i1 - 1]
                        c2 = self.centroids[2][single_i2 - 1]
                        a1 = self.polygon_area(self.rois[1][i1 - 1])
                        a2 = self.polygon_area(self.rois[2][single_i2 - 1])
                        c2_corrected = c2 - np.array([0, 576])
                        writer.writerow([
                            i1, single_i2,
                            f"({c1[0]:.2f}, {c1[1]:.2f})",
                            f"({c2_corrected[0]:.2f}, {c2_corrected[1]:.2f})",
                            f"{a1:.2f}",
                            f"{a2:.2f}"
                        ])

            print(f"Saved mappings with centroids and areas to {os.path.basename(path)}")
        except Exception as e:
            print(f"Error saving mappings: {e}")





from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import csv

class LineageWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())
        
        self.load_btn = QPushButton("Load ROI Mappings CSV")
        self.load_btn.setToolTip("Load CSV file containing ROI mappings for lineage visualization.")
        self.load_btn.clicked.connect(self.load_csv)
        self.layout().addWidget(self.load_btn)

        self.fig, self.ax = plt.subplots(figsize=(24, 18))
        self.canvas = FigureCanvas(self.fig)
        self.graph = None

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load ROI Mapping CSV", filter="CSV Files (*.csv)")
        if not path:
            return
        
        self.graph = nx.Graph()
        
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                roi1 = f"TP1_{row['ROI 1 Index']}"
                roi2 = f"TP2_{row['ROI 2 Index']}"
                
                # Add nodes with extra attributes for TP1
                self.graph.add_node(roi1, bipartite=0, 
                                    roi_index=row['ROI 1 Index'], 
                                    centroid=row['Centroid 1 (Y, X)'],
                                    surface_area=float(row['Area 1']))
                
                # Add nodes with extra attributes for TP2
                self.graph.add_node(roi2, bipartite=1,
                                    roi_index=row['ROI 2 Index'],
                                    centroid=row['Centroid 2 (Y, X)'],
                                    surface_area=float(row['Area 2']))
                
                self.graph.add_edge(roi1, roi2)
        
        self.draw_graph()

    def draw_graph(self):
        self.ax.clear()
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')

        tp1_nodes = [n for n, d in self.graph.nodes(data=True) if d['bipartite'] == 0]
        tp2_nodes = [n for n, d in self.graph.nodes(data=True) if d['bipartite'] == 1]

        pos = {}
        pos.update({n: (0, i) for i, n in enumerate(tp1_nodes)})
        pos.update({n: (1, i) for i, n in enumerate(tp2_nodes)})

        # Node colors by bipartite set
        node_colors = ['skyblue' if self.graph.nodes[n]['bipartite'] == 0 else 'lightgreen' for n in self.graph.nodes()]
        node_edge_color = 'white'

        nx.draw_networkx_nodes(self.graph, pos, ax=self.ax,
                              node_color=node_colors,
                              edgecolors=node_edge_color,
                              linewidths=1.5)

        nx.draw_networkx_edges(self.graph, pos, ax=self.ax, edge_color='white')

        # Prepare multiline labels
        labels = {}
        for n, data in self.graph.nodes(data=True):
            roi_index = data['roi_index']
            centroid = data['centroid']
            area = data['surface_area']

            # Format centroid nicely with 1 decimal place
            centroid_str = centroid
            area_str = f"Area: {area:.1f}"

            # Combine into multiline label
            # Note: '\n' creates multiline labels in matplotlib
            label_text = f"{roi_index}\n{centroid_str}\n{area_str}"
            labels[n] = label_text

        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_color='white', font_size=14, ax=self.ax)

        self.ax.set_axis_off()

        # Draw canvas and convert to image for napari
        self.canvas.draw()
        w, h = self.canvas.get_width_height()
        buf = np.frombuffer(self.canvas.renderer.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(h, w, 4)[..., :3]

        # Remove old layer if present
        if "Lineage Graph" in [layer.name for layer in self.viewer.layers]:
            self.viewer.layers.remove(self.viewer.layers["Lineage Graph"])

        self.viewer.add_image(img, name="Lineage Graph", rgb=True)
