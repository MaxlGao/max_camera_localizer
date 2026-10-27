"""Object meshes in the exact body frames used by recognition, in millimeters."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import numpy as np
import trimesh


DEFAULT_OBJECT_NAMES = ('wrench', 'allen_key', 'jenga_6')


def object_kind(name):
    """Map a recognized instance name to its geometry type."""
    if name in ('wrench', 'allen_key', 'jenga'):
        return name
    if name.startswith('jenga_') and name[6:].isdigit():
        return 'jenga'
    raise ValueError(f'No localizer geometry defined for {name!r}')


def load_object_meshes():
    """Load body-aligned meshes without calculating contours or opening a camera."""
    directory = Path(get_package_share_directory('risl_camera_localizer')) / 'STL'
    key = trimesh.load_mesh(directory / 'Allen Key.STL')
    key.apply_translation([-28.33, 0, -37])
    key.apply_transform(trimesh.transformations.rotation_matrix(
        angle=np.radians(90), direction=[1, 0, 0]))
    key.apply_translation([-50., -10., 0])

    wrench = trimesh.load_mesh(directory / 'Wrench.STL')
    wrench.apply_translation([-49, 0, -26])
    wrench.apply_transform(trimesh.transformations.rotation_matrix(
        angle=np.radians(90), direction=[1, 0, 0]))
    wrench.apply_transform(trimesh.transformations.rotation_matrix(
        angle=np.radians(-90), direction=[0, 0, -1]))

    jenga = trimesh.creation.box(extents=[72, 24, 14])
    jenga.apply_translation([0, 0, -7])
    return {'allen_key': key, 'wrench': wrench, 'jenga': jenga}
