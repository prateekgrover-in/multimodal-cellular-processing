from qtpy.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QWidget,
    QSplitter,
)
from qtpy.QtCore import Qt

class multimodal_cellular_processing(QWidget):
    def __init__(self, viewer):
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
