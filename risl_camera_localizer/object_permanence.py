"""Visibility evidence for remembered, marker-centered Jenga block poses."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ObjectPermanence:
    """Expire contradicted locations quickly; retain unobservable locations.

    Poses use the marker center on the block top surface, with block +Z up.
    Protected tracks stay indefinitely by default; they are beliefs about the
    last observed world pose, not new measurements of a hidden object.
    """

    visible_miss_frames: int = 5
    protected_miss_frames: int = 0
    block_size: tuple = (.072, .024, .014)
    border_margin: int = 8
    min_marker_side: float = 12.
    occluded_fraction: float = .3
    depth_margin: float = .003

    def __post_init__(self):
        if self.visible_miss_frames < 1 or self.protected_miss_frames < 0:
            raise ValueError('visible misses must be positive; protected misses must be nonnegative')

    def classify(self, marker_id, pose, observations, camera_matrix, dist_coeffs,
                 image_shape, cam_pos, cam_quat, marker_size):
        """Return observability and occlusion evidence at the stored world pose."""
        position, quaternion = pose
        position = np.asarray(position, dtype=float)
        rotation = Rotation.from_quat(quaternion)
        camera = np.asarray(cam_pos, dtype=float)
        world_to_cam = Rotation.from_quat(cam_quat).inv()
        half = marker_size / 2
        corners = np.array([[-half, half, 0], [half, half, 0],
                            [half, -half, 0], [-half, -half, 0]])
        world_corners = rotation.apply(corners) + position
        camera_corners = world_to_cam.apply(world_corners - camera)
        if np.any(camera_corners[:, 2] <= .01):
            return 'out_of_frame', 0., []
        image, _ = cv2.projectPoints(camera_corners, np.zeros(3), np.zeros(3),
                                     camera_matrix, dist_coeffs)
        image = image.reshape(4, 2)
        height, width = image_shape[:2]
        margin = self.border_margin
        if (not np.isfinite(image).all() or np.any(image < margin)
                or np.any(image[:, 0] >= width - margin)
                or np.any(image[:, 1] >= height - margin)):
            return 'out_of_frame', 0., []
        view = camera - position
        facing = rotation.as_matrix()[:, 2] @ view / np.linalg.norm(view)
        if facing <= .15:
            return 'edge_on', 0., []
        if np.sqrt(abs(cv2.contourArea(image.astype(np.float32)))) < self.min_marker_side:
            return 'too_small', 0., []

        # Test rays to the marker area, not to the whole block: the tag must
        # be readable, whereas an exposed wooden end does not establish that.
        values = np.linspace(-.9 * half, .9 * half, 5)
        x, y = np.meshgrid(values, values)
        samples = rotation.apply(np.column_stack((x.ravel(), y.ravel(), np.zeros(25)))) + position
        blocked = np.zeros(len(samples), dtype=bool)
        occluders = []
        length, width, thickness = self.block_size
        lower = np.array([-length / 2, -width / 2, -thickness])
        upper = np.array([length / 2, width / 2, 0.])
        distances = np.linalg.norm(samples - camera, axis=1)
        for other_id, (other_pos, other_quat) in observations.items():
            if other_id == marker_id:
                continue
            inverse = Rotation.from_quat(other_quat).inv()
            origin = inverse.apply(camera - other_pos)
            direction = inverse.apply(samples - camera)
            parallel = np.abs(direction) < 1e-10
            outside = parallel & ((origin < lower) | (origin > upper))
            near_axis = np.full_like(direction, -np.inf)
            far_axis = np.full_like(direction, np.inf)
            a = np.divide(lower - origin, direction, out=np.zeros_like(direction), where=~parallel)
            b = np.divide(upper - origin, direction, out=np.zeros_like(direction), where=~parallel)
            near_axis[~parallel] = np.minimum(a, b)[~parallel]
            far_axis[~parallel] = np.maximum(a, b)[~parallel]
            near, far = near_axis.max(axis=1), far_axis.min(axis=1)
            hits = (~outside.any(axis=1) & (far >= np.maximum(near, 0.))
                    & (far > 0.) & (near < 1. - self.depth_margin / distances))
            if hits.any():
                occluders.append(int(other_id))
                blocked |= hits
        coverage = float(blocked.mean())
        return ('occluded' if coverage >= self.occluded_fraction else 'visible', coverage, occluders)

    def update(self, states, observations, detected_ids, current_frame,
               camera_matrix, dist_coeffs, image_shape, cam_pos, cam_quat, marker_size):
        """Build an observation-rooted occlusion chain before aging any tracks.

        Resolve one layer at a time, independently of dictionary order. A
        remembered block protected by a resolved layer becomes an occluder for
        the next layer, using its unchanged world pose. Rebuild from current
        observations every frame; old chains never seed a new one.
        """
        trusted = dict(observations)
        chains = {marker_id: [int(marker_id)] for marker_id in observations}
        roots = {marker_id: {int(marker_id)} for marker_id in observations}
        missing = {marker_id for marker_id, state in states.items()
                   if state['confirmed'] and marker_id not in detected_ids}
        unresolved = set(missing)
        visibility_results = {}
        while unresolved:
            promoted = {}
            for marker_id in sorted(unresolved):
                state = states[marker_id]
                pose = (state['world_position'], state['world_quaternion'])
                result = self.classify(
                    marker_id, pose, trusted, camera_matrix, dist_coeffs,
                    image_shape, cam_pos, cam_quat, marker_size)
                visibility_results[marker_id] = result
                cap_expires = (self.protected_miss_frames > 0 and
                               state.get('protected_misses', 0) + 1 >= self.protected_miss_frames)
                if result[0] == 'occluded' and not cap_expires:
                    promoted[marker_id] = pose
            if not promoted:
                break
            # Classify the whole layer before adding any of its geometry.
            # Parents therefore always belong to a previously resolved layer.
            for marker_id in promoted:
                parents = visibility_results[marker_id][2]
                paths = [chains[parent] + [int(marker_id)] for parent in parents]
                chains[marker_id] = min(paths, key=lambda path: (-len(path), path))
                roots[marker_id] = set().union(*(roots[parent] for parent in parents))
            trusted.update(promoted)
            unresolved.difference_update(promoted)

        dropped = []
        for marker_id, state in states.items():
            if not state['confirmed']:
                continue
            if marker_id in detected_ids:
                state.update(visibility=('observed' if state.get('measurement_frame') == current_frame
                                         else 'detected'), visible_misses=0, protected_misses=0,
                             occlusion_fraction=0., occluders=[],
                             occlusion_chain=[int(marker_id)], occlusion_roots=[int(marker_id)])
                continue
            visibility, coverage, occluders = visibility_results[marker_id]
            state['occlusion_chain'] = chains.get(marker_id, [])
            state['occlusion_roots'] = sorted(roots.get(marker_id, set()))
            state.update(visibility=visibility, occlusion_fraction=coverage, occluders=occluders)
            if visibility == 'visible':
                state['visible_misses'] = state.get('visible_misses', 0) + 1
                state['protected_misses'] = 0
                expired = state['visible_misses'] >= self.visible_miss_frames
            else:
                state['visible_misses'] = 0
                state['protected_misses'] = state.get('protected_misses', 0) + 1
                expired = (self.protected_miss_frames > 0
                           and state['protected_misses'] >= self.protected_miss_frames)
            if expired:
                state.update(confirmed=False, dropped=True, hold_counter=0, last_tvec=None)
                dropped.append(marker_id)
        return dropped
