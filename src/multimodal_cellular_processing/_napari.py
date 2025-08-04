from napari_plugin_engine import napari_hook_implementation
from ._widget import multimodal_cellular_processing, SegmentationWidget  # adjust imports

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [
        (multimodal_cellular_processing, {"name": "Preprocessing"}),
        (SegmentationWidget, {"name": "Segmentation"}),  # optional second panel
    ]
