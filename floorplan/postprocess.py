from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_propagation
from skimage.morphology import skeletonize


PRESETS = {
    "weak": {
        "strong": 0.50,
        "weak": 0.10,
        "close_radius": 8,
        "dilate_radius": 2,
    },
    "conservative": {
        "strong": 0.50,
        "weak": 0.18,
        "close_radius": 8,
        "dilate_radius": 1,
    },
}


@dataclass
class RoomRegion:
    label: int
    area: int
    polygon: np.ndarray


def room_count(walls, minimum_ratio=0.001):
    free = ~walls
    seed = np.zeros_like(free)
    seed[[0, -1]] = free[[0, -1]]
    seed[:, [0, -1]] |= free[:, [0, -1]]

    # Border-connected free space is exterior, not a room.
    enclosed = free & ~binary_propagation(seed, mask=free)
    count, _, stats, _ = cv2.connectedComponentsWithStats(enclosed.astype(np.uint8), 8)
    minimum = max(32, round(walls.size * minimum_ratio))
    return sum(stats[index, cv2.CC_STAT_AREA] >= minimum for index in range(1, count))


def endpoint_direction(skeleton, start, support):
    current, previous, path = start, None, [start]

    # Follow a non-branching skeleton path to estimate its outgoing tangent.
    for _ in range(2 * support):
        y, x = current
        neighbors = [
            (yy, xx)
            for yy in range(max(0, y - 1), min(skeleton.shape[0], y + 2))
            for xx in range(max(0, x - 1), min(skeleton.shape[1], x + 2))
            if (yy, xx) not in (current, previous) and skeleton[yy, xx]
        ]
        if len(neighbors) != 1:
            break
        previous, current = current, neighbors[0]
        path.append(current)
    vector = np.array((start[1] - path[-1][1], start[0] - path[-1][0]), np.float32)
    length = np.linalg.norm(vector)
    return vector / length if length >= support else None


def bridge_gaps(walls, max_gap=50, angle=12, support=15):
    skeleton = skeletonize(walls)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    points = [tuple(point) for point in np.argwhere(skeleton & (neighbors == 2))]
    directions = [endpoint_direction(skeleton, point, support) for point in points]
    cosine, candidates = np.cos(np.deg2rad(angle)), []

    # Pair only nearby endpoints whose tangents face each other.
    for i, (a, first) in enumerate(zip(points, directions)):
        if first is None:
            continue
        for j in range(i + 1, len(points)):
            second = directions[j]
            if second is None:
                continue
            delta = np.array((points[j][1] - a[1], points[j][0] - a[0]), np.float32)
            distance = np.linalg.norm(delta)
            direction = delta / max(distance, 1e-6)
            aligned = first @ direction >= cosine and second @ -direction >= cosine
            if 3 <= distance <= max_gap and aligned:
                candidates.append((distance, i, j))

    result, count, used = walls.copy(), room_count(walls), set()
    distance_map = cv2.distanceTransform(walls.astype(np.uint8), cv2.DIST_L2, 5)

    # Keep a bridge only when it creates a new enclosed room.
    for _, i, j in sorted(candidates):
        if i in used or j in used:
            continue
        a, b = points[i], points[j]
        thickness = int(np.clip(round(distance_map[a] + distance_map[b]), 2, 12))
        trial = result.astype(np.uint8)
        cv2.line(trial, (a[1], a[0]), (b[1], b[0]), 1, thickness, cv2.LINE_AA)
        new_count = room_count(trial.astype(bool))
        if new_count > count:
            result, count = trial.astype(bool), new_count
            used.update((i, j))
    return result


def wall_barrier(probability, preset):
    config = PRESETS[preset]
    probability = probability.copy()

    # Clear the frame so padding cannot become an artificial outer wall.
    probability[:3] = probability[-3:] = 0
    probability[:, :3] = probability[:, -3:] = 0

    # Hysteresis admits weak wall pixels only when linked to strong evidence.
    walls = binary_propagation(
        probability >= config["strong"], mask=probability >= config["weak"]
    )
    close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * config["close_radius"] + 1,) * 2
    )
    dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * config["dilate_radius"] + 1,) * 2
    )

    # Closing repairs local cracks; dilation makes the barrier watertight.
    walls = cv2.morphologyEx(walls.astype(np.uint8), cv2.MORPH_CLOSE, close)
    walls = cv2.dilate(walls, dilate).astype(bool)
    return bridge_gaps(walls)


def enclosed_rooms(walls, minimum_ratio=0.001, minimum_radius=2.0):
    free = ~walls
    seed = np.zeros_like(free)
    seed[[0, -1]] = free[[0, -1]]
    seed[:, [0, -1]] |= free[:, [0, -1]]

    # Remove all free space reachable from the image boundary.
    enclosed = free & ~binary_propagation(seed, mask=free)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        enclosed.astype(np.uint8), 8
    )
    output, rooms, next_label = np.zeros_like(labels), [], 1
    minimum = max(32, round(labels.size * minimum_ratio))

    # Reject tiny and narrow components before polygon approximation.
    for label in range(1, count):
        component, area = labels == label, int(stats[label, cv2.CC_STAT_AREA])
        radius = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3).max()
        if area < minimum or radius < minimum_radius:
            continue
        contours, _ = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(1, 0.005 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True)[:, 0]
        output[component] = next_label
        rooms.append(RoomRegion(next_label, area, polygon))
        next_label += 1
    return output, rooms


def extract_rooms(probability, preset="weak"):
    walls = wall_barrier(probability, preset)
    labels, rooms = enclosed_rooms(walls)
    return walls, labels, rooms
