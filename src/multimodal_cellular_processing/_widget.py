from typing import TYPE_CHECKING
import itk
import napari
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_laplace
from aicssegmentation.core.vessel import vesselness2D
from skimage.filters import threshold_li, threshold_otsu, threshold_sauvola, threshold_triangle
from skimage.morphology import disk, erosion, medial_axis, white_tophat
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari


class IntensityNormalization(QGroupBox):
    # Same as before but renamed parameters to modality/channel aware names
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('Intensity normalization (per modality)')
        self.setVisible(False)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.viewer = parent.viewer
        self.parent = parent
        self.modality_name = ''          # renamed for multimodal context
        self.lower_percentage = 0.0
        self.upper_percentage = 95.0

        vbox = QVBoxLayout()
        self.setLayout(vbox)

        vbox.addWidget(QLabel('Select modality/channel'))
        self.cbx_modality = QComboBox()
        self.cbx_modality.addItems(parent.layer_names)
        self.cbx_modality.currentIndexChanged.connect(self.modality_changed)
        vbox.addWidget(self.cbx_modality)

        self.lbl_lower_percentage = QLabel('Lower percentile: 0.00')
        vbox.addWidget(self.lbl_lower_percentage)
        sld_lower_percentage = QSlider(Qt.Horizontal)
        sld_lower_percentage.setRange(0, 500)
        sld_lower_percentage.valueChanged.connect(self.lower_changed)
        vbox.addWidget(sld_lower_percentage)

        self.lbl_upper_percentage = QLabel('Upper percentile: 95.00')
        vbox.addWidget(self.lbl_upper_percentage)
        sld_upper_percentage = QSlider(Qt.Horizontal)
        sld_upper_percentage.setRange(9500, 10000)
        sld_upper_percentage.valueChanged.connect(self.upper_changed)
        vbox.addWidget(sld_upper_percentage)

        btn_run = QPushButton('Run normalization')
        btn_run.clicked.connect(self.run_intensity_normalization)
        vbox.addWidget(btn_run)

    def modality_changed(self, index: int):
        self.modality_name = self.parent.layer_names[index]

    def lower_changed(self, value: int):
        self.lower_percentage = float(value) / 100.0
        self.lbl_lower_percentage.setText(f'Lower percentile: {self.lower_percentage:.2f}')

    def upper_changed(self, value: int):
        self.upper_percentage = float(value) / 100.0
        self.lbl_upper_percentage.setText(f'Upper percentile: {self.upper_percentage:.2f}')

    def run_intensity_normalization(self):
        if self.modality_name == '':
            self.modality_changed(0)

        if any(layer.name == self.modality_name for layer in self.viewer.layers):
            layer = self.viewer.layers[self.modality_name]
            input_image = layer.data
        else:
            print(f'Error: The modality {self.modality_name} does not exist!')
            return

        lower_v = np.percentile(input_image, self.lower_percentage)
        upper_v = np.percentile(input_image, self.upper_percentage)
        clipped = np.clip(input_image, lower_v, upper_v)
        output = (clipped - lower_v) / (upper_v - lower_v)
        self.viewer.add_image(output, name=f'{self.modality_name}_normalized')


class multimodal_cellular_processing(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

       # Main horizontal layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Create Microscopy panel
        microscopy_group = QGroupBox("Microscopy Data")
        microscopy_layout = QVBoxLayout()
        microscopy_group.setLayout(microscopy_layout)
        microscopy_layout.addWidget(QLabel("Microscopy controls go here"))
        # Add your microscopy widgets here, e.g. normalization, filters etc.

        # Create Impedance panel
        impedance_group = QGroupBox("Impedance Data")
        impedance_layout = QVBoxLayout()
        impedance_group.setLayout(impedance_layout)
        impedance_layout.addWidget(QLabel("Impedance controls go here"))
        # Add your impedance widgets here

        # Use QSplitter for adjustable resizing between panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(microscopy_group)
        splitter.addWidget(impedance_group)

        main_layout.addWidget(splitter)