import math
import os
import platform
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pygame
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_PROJECTION,
    GL_SRC_ALPHA,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor4f,
    glEnable,
    glEnd,
    glLoadIdentity,
    glMatrixMode,
    glPointSize,
    glRotatef,
    glTranslatef,
    glVertex3f,
)
from OpenGL.GLU import gluPerspective
from pygame.locals import (
    DOUBLEBUF,
    KEYDOWN,
    K_0,
    K_1,
    K_2,
    K_3,
    K_4,
    K_DOWN,
    K_ESCAPE,
    K_LEFT,
    K_RIGHT,
    K_UP,
    K_a,
    K_d,
    K_e,
    K_f,
    K_g,
    K_m,
    K_q,
    K_r,
    K_s,
    K_w,
    K_x,
    K_z,
    OPENGL,
)


WIDTH, HEIGHT = 1280, 720
NUM_PARTICLES = 4200
MORPH_SPEED = 0.12
COLOR_MORPH_SPEED = 0.14
POINT_SIZE = 4.1

MODE_RANDOM = 0
MODE_SOLAR = 1
MODE_PWN = 2
MODE_IMPRUP = 3
MODE_HEART = 4

GESTURE_STABLE_FRAMES = 6
NO_HAND_IDLE_FRAMES = 18
VIEW_ROTATE_STEP = 8.0
VIEW_MOVE_STEP = 0.45
VIEW_Z_STEP = 0.8

AUTO_SPIN_SPEEDS = {
    MODE_RANDOM: 0.24,
    MODE_SOLAR: 0.0,
    MODE_PWN: 0.0,
    MODE_IMPRUP: 0.0,
    MODE_HEART: 0.0,
}

MODE_LABELS = {
    MODE_RANDOM: "IDLE / ACAK",
    MODE_SOLAR: "SOLAR SYSTEM",
    MODE_PWN: "I LOVE PWN",
    MODE_IMPRUP: "PLS IMPRUP",
    MODE_HEART: "SARANGHAE / LOVE",
}

GESTURE_LABELS = {
    "open_hand": "LIMA JARI",
    "peace": "PEACE SIGN",
    "thumbs_up": "JEMPOL",
    "love_sign": "LOVE / FINGER HEART",
}

GESTURE_TO_MODE = {
    "open_hand": MODE_SOLAR,
    "peace": MODE_PWN,
    "thumbs_up": MODE_IMPRUP,
    "love_sign": MODE_HEART,
}

lock = threading.Lock()
shared_data = {
    "running": True,
    "mode": MODE_RANDOM,
    "gesture_label": "NO HAND",
    "camera_status": "Mencari kamera...",
    "gesture_control": True,
    "target_x": 0.0,
    "target_y": 0.0,
    "target_z": -18.0,
    "frame": None,
}


def running_in_wsl() -> bool:
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def build_camera_candidates() -> list[tuple[int, int | None, str]]:
    raw_index = os.environ.get("CAMERA_INDEX")
    if raw_index is None:
        raw_index = os.environ.get("NEBULA_CAMERA_INDEX", "0,1,2")
    indices: list[int] = []
    for item in raw_index.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            indices.append(int(item))
        except ValueError:
            continue
    if not indices:
        indices = [0]

    candidates: list[tuple[int, int | None, str]] = []
    if os.name == "nt":
        for index in indices:
            candidates.append((index, cv2.CAP_DSHOW, f"index {index} / DirectShow"))
            candidates.append((index, cv2.CAP_MSMF, f"index {index} / MSMF"))
            candidates.append((index, None, f"index {index} / default"))
    else:
        for index in indices:
            if hasattr(cv2, "CAP_V4L2"):
                candidates.append((index, cv2.CAP_V4L2, f"index {index} / V4L2"))
            candidates.append((index, None, f"index {index} / default"))
    return candidates


def open_camera() -> tuple[cv2.VideoCapture | None, str]:
    if running_in_wsl() and not any(os.path.exists(f"/dev/video{i}") for i in range(5)):
        return None, (
            "WSL2 tidak melihat webcam (`/dev/video*` tidak ada). "
            "Jalankan script ini dengan Python Windows, atau attach webcam ke WSL via usbipd."
        )

    errors: list[str] = []
    for index, backend, label in build_camera_candidates():
        cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap, f"Kamera aktif: {label}"
        errors.append(label)
        cap.release()

    return None, "Kamera gagal dibuka. Sudah coba: " + ", ".join(errors)


def repeat_color(color: tuple[float, float, float], count: int) -> np.ndarray:
    return np.tile(np.array(color, dtype=np.float32), (count, 1))


def mirror_x_points(points: np.ndarray) -> np.ndarray:
    mirrored = points.copy()
    mirrored[:, 0] *= -1.0
    return mirrored


def scale_and_offset_points(
    points: np.ndarray,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    transformed = points.copy()
    transformed *= np.array(scale, dtype=np.float32)
    transformed += np.array(offset, dtype=np.float32)
    return transformed


def rotate_points_z(points: np.ndarray, angle: float) -> np.ndarray:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    rotated = points.copy()
    x = rotated[:, 0]
    y = rotated[:, 1]
    rotated[:, 0] = (x * cos_angle) - (y * sin_angle)
    rotated[:, 1] = (x * sin_angle) + (y * cos_angle)
    return rotated


def sample_points(points: np.ndarray, total: int, jitter: float = 0.03) -> np.ndarray:
    replace = len(points) < total
    chosen = np.random.choice(len(points), total, replace=replace)
    sampled = points[chosen].astype(np.float32)
    sampled += np.random.normal(scale=jitter, size=sampled.shape).astype(np.float32)
    return sampled


def build_scatter(total: int) -> np.ndarray:
    return np.random.uniform(
        low=[-6.0, -3.8, -3.2],
        high=[6.0, 3.8, 3.2],
        size=(total, 3),
    ).astype(np.float32)


def build_scatter_colors(points: np.ndarray) -> np.ndarray:
    x_norm = (points[:, 0] + 6.0) / 12.0
    z_norm = (points[:, 2] + 3.2) / 6.4
    colors = np.column_stack(
        (
            0.12 + (0.18 * z_norm),
            0.45 + (0.35 * x_norm),
            0.78 + (0.2 * (1.0 - z_norm)),
        )
    )
    return colors.astype(np.float32)


def build_heart(total: int) -> np.ndarray:
    t = np.random.uniform(0.0, 2.0 * math.pi, total)
    fill_scale = np.sqrt(np.random.uniform(0.12, 1.0, total))

    x = 16.0 * (np.sin(t) ** 3)
    y = (
        13.0 * np.cos(t)
        - 5.0 * np.cos(2.0 * t)
        - 2.0 * np.cos(3.0 * t)
        - np.cos(4.0 * t)
    )

    x = (x * fill_scale) / 7.2
    y = ((y * fill_scale) / 7.2) + 0.15
    z = np.random.normal(0.0, 0.16, total)

    return np.column_stack((x, y, z)).astype(np.float32)


def build_heart_outline(total: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, total, endpoint=False, dtype=np.float32)
    t += np.random.normal(0.0, 0.02, total).astype(np.float32)

    x = 16.0 * (np.sin(t) ** 3)
    y = (
        13.0 * np.cos(t)
        - 5.0 * np.cos(2.0 * t)
        - 2.0 * np.cos(3.0 * t)
        - np.cos(4.0 * t)
    )

    x = x / 6.9
    y = (y / 6.9) + 0.18
    z = np.random.normal(0.0, 0.05, total)

    return np.column_stack((x, y, z)).astype(np.float32)


def build_text_points(text: str, font_size: int, total: int) -> np.ndarray:
    font = pygame.font.SysFont("dejavusans", font_size, bold=True)
    text_surface = font.render(text, True, (255, 255, 255))

    padding = 40
    canvas = pygame.Surface(
        (text_surface.get_width() + padding * 2, text_surface.get_height() + padding * 2),
        pygame.SRCALPHA,
    )
    canvas.blit(text_surface, (padding, padding))

    alpha = pygame.surfarray.array_alpha(canvas).T
    y_indices, x_indices = np.where(alpha > 10)

    x = (x_indices - (canvas.get_width() / 2.0)) / 48.0
    y = -((y_indices - (canvas.get_height() / 2.0)) / 48.0)
    z = np.random.normal(0.0, 0.05, len(x_indices))

    base_points = np.column_stack((x, y, z)).astype(np.float32)
    return sample_points(base_points, total, jitter=0.035)


def build_love_scene_builder(total: int):
    heart_count = 2200
    outline_count = 850
    line_one_count = 420
    line_two_count = 500
    sparkle_count = 230
    if heart_count + outline_count + line_one_count + line_two_count + sparkle_count != total:
        raise ValueError("Jumlah partikel love scene tidak pas")

    heart_core = scale_and_offset_points(build_heart(heart_count), scale=(1.16, 1.18, 1.0), offset=(0.0, 1.05, 0.0))
    heart_outline = scale_and_offset_points(
        build_heart_outline(outline_count),
        scale=(1.42, 1.42, 1.0),
        offset=(0.0, 1.1, 0.0),
    )
    line_one = scale_and_offset_points(build_text_points("WILL U", 102, line_one_count), scale=(0.92, 0.92, 1.0), offset=(0.0, -2.55, 0.0))
    line_two = scale_and_offset_points(build_text_points("BE MINE?", 112, line_two_count), scale=(0.92, 0.92, 1.0), offset=(0.0, -3.95, 0.0))
    line_one_mirror = mirror_x_points(line_one)
    line_two_mirror = mirror_x_points(line_two)

    sparkle_theta = np.random.uniform(0.0, 2.0 * math.pi, sparkle_count).astype(np.float32)
    sparkle_radius = np.random.uniform(3.2, 5.0, sparkle_count).astype(np.float32)
    sparkle_height = np.random.uniform(-3.6, 4.2, sparkle_count).astype(np.float32)
    sparkle_phase = np.random.uniform(0.0, 2.0 * math.pi, sparkle_count).astype(np.float32)
    sparkle_z = np.random.uniform(-1.4, 1.4, sparkle_count).astype(np.float32)

    heart_y_norm = np.clip((heart_core[:, 1] + 2.0) / 6.0, 0.0, 1.0)
    heart_colors = np.column_stack(
        (
            1.0 - (0.1 * (1.0 - heart_y_norm)),
            0.16 + (0.38 * heart_y_norm),
            0.38 + (0.33 * (1.0 - heart_y_norm)),
        )
    ).astype(np.float32)
    outline_colors = repeat_color((1.0, 0.78, 0.36), outline_count)
    line_one_colors = repeat_color((1.0, 0.88, 0.96), line_one_count)
    line_two_colors = repeat_color((1.0, 0.72, 0.85), line_two_count)

    sparkle_palette = np.array(
        [
            (1.0, 0.98, 0.92),
            (1.0, 0.86, 0.52),
            (1.0, 0.62, 0.82),
        ],
        dtype=np.float32,
    )
    sparkle_colors = sparkle_palette[np.arange(sparkle_count) % len(sparkle_palette)]

    def builder(elapsed: float, mirrored: bool = False) -> tuple[np.ndarray, np.ndarray]:
        heart_pulse = 1.0 + (0.055 * math.sin(elapsed * 2.35))
        outline_pulse = 1.0 + (0.11 * math.sin((elapsed * 3.2) + 0.7))
        outline_spin = rotate_points_z(heart_outline * outline_pulse, elapsed * 0.22)

        line_bob = 0.12 * math.sin(elapsed * 1.75)
        text_line_one = (line_one_mirror if mirrored else line_one).copy()
        text_line_two = (line_two_mirror if mirrored else line_two).copy()
        text_line_one[:, 1] += line_bob
        text_line_two[:, 1] += line_bob

        sparkle_angle = sparkle_theta + (elapsed * 0.55)
        sparkle_orbit = np.column_stack(
            (
                sparkle_radius * np.cos(sparkle_angle),
                sparkle_height + (0.18 * np.sin((elapsed * 2.2) + sparkle_phase)),
                sparkle_z + (sparkle_radius * 0.22 * np.sin(sparkle_angle * 0.7)),
            )
        ).astype(np.float32)

        positions = np.concatenate(
            (
                heart_core * heart_pulse,
                outline_spin,
                text_line_one,
                text_line_two,
                sparkle_orbit,
            ),
            axis=0,
        )
        colors = np.concatenate(
            (
                heart_colors,
                outline_colors,
                line_one_colors,
                line_two_colors,
                sparkle_colors,
            ),
            axis=0,
        )
        return positions, colors

    return builder


def build_love_confetti(total: int = 320):
    x_base = np.random.uniform(-7.2, 7.2, total).astype(np.float32)
    y_base = np.random.uniform(-1.0, 9.0, total).astype(np.float32)
    z_base = np.random.uniform(-1.8, 1.8, total).astype(np.float32)
    speed = np.random.uniform(0.9, 2.2, total).astype(np.float32)
    sway = np.random.uniform(0.2, 0.9, total).astype(np.float32)
    phase = np.random.uniform(0.0, 2.0 * math.pi, total).astype(np.float32)

    palette = np.array(
        [
            (1.0, 0.94, 0.58),
            (1.0, 0.68, 0.84),
            (1.0, 0.84, 0.96),
            (1.0, 0.52, 0.72),
            (0.98, 0.98, 1.0),
        ],
        dtype=np.float32,
    )
    colors = palette[np.arange(total) % len(palette)]

    def builder(elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        y = ((y_base - (elapsed * speed * 1.9)) % 12.0) - 5.0
        x = x_base + (0.42 * np.sin((elapsed * sway * 1.4) + phase))
        z = z_base + (0.3 * np.cos((elapsed * sway) + phase))
        points = np.column_stack((x, y, z)).astype(np.float32)
        return points, colors

    return builder


def build_background_stars(total: int = 1600) -> tuple[np.ndarray, np.ndarray]:
    points = np.random.uniform(
        low=[-16.0, -10.0, -22.0],
        high=[16.0, 10.0, -5.0],
        size=(total, 3),
    ).astype(np.float32)
    twinkle = np.random.uniform(0.65, 1.0, total)
    colors = np.column_stack(
        (
            0.1 + (0.12 * twinkle),
            0.22 + (0.25 * twinkle),
            0.55 + (0.35 * twinkle),
        )
    ).astype(np.float32)
    return points, colors


def unit_sphere_points(count: int, min_fill: float = 0.1) -> np.ndarray:
    theta = np.random.uniform(0.0, 2.0 * math.pi, count)
    u = np.random.uniform(-1.0, 1.0, count)
    radius = np.cbrt(np.random.uniform(min_fill, 1.0, count))
    radial = np.sqrt(1.0 - (u ** 2))

    x = radius * radial * np.cos(theta)
    y = radius * u
    z = radius * radial * np.sin(theta)
    return np.column_stack((x, y, z)).astype(np.float32)


def build_solar_system_builder(total: int):
    body_defs = [
        {"name": "Sun", "orbit": 0.0, "radius": 1.05, "count": 850, "color": (1.0, 0.76, 0.2), "speed": 0.0, "phase": 0.0},
        {"name": "Mercury", "orbit": 2.0, "radius": 0.13, "count": 70, "color": (0.78, 0.72, 0.66), "speed": 1.65, "phase": 0.4},
        {"name": "Venus", "orbit": 2.8, "radius": 0.22, "count": 110, "color": (0.95, 0.73, 0.42), "speed": 1.25, "phase": 1.1},
        {"name": "Earth", "orbit": 3.7, "radius": 0.24, "count": 120, "color": (0.24, 0.65, 1.0), "speed": 1.0, "phase": 2.2},
        {"name": "Mars", "orbit": 4.7, "radius": 0.18, "count": 90, "color": (0.93, 0.38, 0.24), "speed": 0.82, "phase": 0.2},
        {"name": "Jupiter", "orbit": 6.4, "radius": 0.47, "count": 260, "color": (0.93, 0.73, 0.52), "speed": 0.38, "phase": 1.7},
        {"name": "Saturn", "orbit": 8.3, "radius": 0.4, "count": 240, "color": (0.9, 0.84, 0.55), "speed": 0.28, "phase": 2.9},
        {"name": "Uranus", "orbit": 10.2, "radius": 0.32, "count": 170, "color": (0.53, 0.93, 0.96), "speed": 0.2, "phase": 4.1},
        {"name": "Neptune", "orbit": 12.0, "radius": 0.31, "count": 160, "color": (0.24, 0.44, 1.0), "speed": 0.15, "phase": 5.0},
    ]

    orbit_counts = {
        "Mercury": 130,
        "Venus": 150,
        "Earth": 160,
        "Mars": 170,
        "Jupiter": 190,
        "Saturn": 210,
        "Uranus": 220,
        "Neptune": 230,
    }
    saturn_ring_count = 200
    asteroid_belt_count = 240
    halo_count = 230

    expected_total = sum(item["count"] for item in body_defs)
    expected_total += sum(orbit_counts.values())
    expected_total += saturn_ring_count + asteroid_belt_count + halo_count
    if expected_total != total:
        raise ValueError(f"Jumlah partikel solar system tidak pas: {expected_total} != {total}")

    body_templates = {
        body["name"]: unit_sphere_points(body["count"], min_fill=0.18) * body["radius"]
        for body in body_defs
    }
    orbit_templates = {}
    for body in body_defs[1:]:
        count = orbit_counts[body["name"]]
        orbit_templates[body["name"]] = {
            "theta": np.random.uniform(0.0, 2.0 * math.pi, count).astype(np.float32),
            "radius_noise": np.random.normal(0.0, 0.016, count).astype(np.float32),
            "y_noise": np.random.normal(0.0, 0.006, count).astype(np.float32),
        }

    saturn_ring = {
        "theta": np.random.uniform(0.0, 2.0 * math.pi, saturn_ring_count).astype(np.float32),
        "radius": np.random.uniform(0.65, 1.15, saturn_ring_count).astype(np.float32),
        "y_noise": np.random.normal(0.0, 0.008, saturn_ring_count).astype(np.float32),
    }
    asteroid_belt = {
        "theta": np.random.uniform(0.0, 2.0 * math.pi, asteroid_belt_count).astype(np.float32),
        "radius": np.random.uniform(5.15, 5.85, asteroid_belt_count).astype(np.float32),
        "y_noise": np.random.normal(0.0, 0.02, asteroid_belt_count).astype(np.float32),
    }
    sun_halo = unit_sphere_points(halo_count, min_fill=0.82)

    def orbit_cloud(radius: float, template: dict[str, np.ndarray]) -> np.ndarray:
        orbital_radius = radius + template["radius_noise"]
        x = orbital_radius * np.cos(template["theta"])
        y = template["y_noise"]
        z = orbital_radius * np.sin(template["theta"])
        return np.column_stack((x, y, z)).astype(np.float32)

    def builder(elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        positions: list[np.ndarray] = []
        colors: list[np.ndarray] = []

        sun_pulse = 1.0 + (0.06 * math.sin(elapsed * 2.3))
        sun_core = body_templates["Sun"] * sun_pulse
        positions.append(sun_core)
        colors.append(repeat_color((1.0, 0.78, 0.18), len(sun_core)))

        halo_scale = 1.65 + (0.1 * math.sin(elapsed * 3.0))
        halo = sun_halo * halo_scale
        positions.append(halo.astype(np.float32))
        colors.append(repeat_color((1.0, 0.46, 0.12), len(halo)))

        saturn_center = None
        saturn_angle = 0.0

        for body in body_defs[1:]:
            orbit_points = orbit_cloud(body["orbit"], orbit_templates[body["name"]])
            positions.append(orbit_points)

            orbit_color = tuple(max(component * 0.38, 0.1) for component in body["color"])
            colors.append(repeat_color(orbit_color, len(orbit_points)))

            angle = body["phase"] + (elapsed * body["speed"])
            vertical = 0.035 * math.sin((elapsed * body["speed"] * 0.45) + body["phase"])
            center = np.array(
                [
                    body["orbit"] * math.cos(angle),
                    vertical,
                    body["orbit"] * math.sin(angle),
                ],
                dtype=np.float32,
            )

            planet_cloud = body_templates[body["name"]] + center
            positions.append(planet_cloud.astype(np.float32))
            colors.append(repeat_color(body["color"], len(planet_cloud)))

            if body["name"] == "Saturn":
                saturn_center = center
                saturn_angle = angle

        asteroid_angle = asteroid_belt["theta"] + (elapsed * 0.15)
        asteroid_points = np.column_stack(
            (
                asteroid_belt["radius"] * np.cos(asteroid_angle),
                asteroid_belt["y_noise"],
                asteroid_belt["radius"] * np.sin(asteroid_angle),
            )
        ).astype(np.float32)
        positions.append(asteroid_points)
        colors.append(repeat_color((0.65, 0.55, 0.46), len(asteroid_points)))

        if saturn_center is not None:
            ring_theta = saturn_ring["theta"] + (elapsed * 0.6)
            ring_radius = saturn_ring["radius"]
            ring_local = np.column_stack(
                (
                    ring_radius * np.cos(ring_theta),
                    saturn_ring["y_noise"],
                    0.55 * ring_radius * np.sin(ring_theta),
                )
            ).astype(np.float32)

            tilt = 0.75
            cos_tilt = math.cos(tilt)
            sin_tilt = math.sin(tilt)
            y_rot = (ring_local[:, 1] * cos_tilt) - (ring_local[:, 2] * sin_tilt)
            z_rot = (ring_local[:, 1] * sin_tilt) + (ring_local[:, 2] * cos_tilt)
            rotated_ring = np.column_stack((ring_local[:, 0], y_rot, z_rot)).astype(np.float32)

            spin = saturn_angle * 0.5
            cos_spin = math.cos(spin)
            sin_spin = math.sin(spin)
            x_spin = (rotated_ring[:, 0] * cos_spin) - (rotated_ring[:, 2] * sin_spin)
            z_spin = (rotated_ring[:, 0] * sin_spin) + (rotated_ring[:, 2] * cos_spin)
            saturn_points = np.column_stack((x_spin, rotated_ring[:, 1], z_spin)).astype(np.float32)
            saturn_points += saturn_center

            positions.append(saturn_points)
            colors.append(repeat_color((0.95, 0.88, 0.62), len(saturn_points)))

        return np.concatenate(positions, axis=0), np.concatenate(colors, axis=0)

    return builder


def setup_opengl() -> None:
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, WIDTH / HEIGHT, 0.1, 60.0)

    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPointSize(POINT_SIZE)


def draw_points(points: np.ndarray, colors: np.ndarray, alpha: float) -> None:
    glBegin(GL_POINTS)
    for point, color in zip(points, colors):
        depth_fade = max(0.3, 1.0 - abs(point[2]) * 0.08)
        glColor4f(color[0], color[1], color[2], alpha * depth_fade)
        glVertex3f(point[0], point[1], point[2])
    glEnd()


def landmarks_to_array(hand_landmarks) -> np.ndarray:
    return np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32)


def point_distance(points: np.ndarray, idx_a: int, idx_b: int) -> float:
    return float(np.linalg.norm(points[idx_a] - points[idx_b]))


def finger_extended(points: np.ndarray, tip: int, pip: int, mcp: int) -> bool:
    wrist_distance_tip = point_distance(points, 0, tip)
    wrist_distance_pip = point_distance(points, 0, pip)
    finger_span_tip = point_distance(points, mcp, tip)
    finger_span_pip = point_distance(points, mcp, pip)
    return (wrist_distance_tip > wrist_distance_pip * 1.12) and (finger_span_tip > finger_span_pip * 1.45)


def thumb_extended(points: np.ndarray) -> bool:
    wrist_to_tip = point_distance(points, 0, 4)
    wrist_to_ip = point_distance(points, 0, 3)
    tip_to_index = point_distance(points, 4, 5)
    ip_to_index = point_distance(points, 3, 5)
    return (wrist_to_tip > wrist_to_ip * 1.08) and (tip_to_index > ip_to_index * 1.1)


def finger_curled(points: np.ndarray, tip: int, pip: int, mcp: int) -> bool:
    wrist_distance_tip = point_distance(points, 0, tip)
    wrist_distance_pip = point_distance(points, 0, pip)
    finger_span_tip = point_distance(points, mcp, tip)
    finger_span_pip = point_distance(points, mcp, pip)
    return (wrist_distance_tip < wrist_distance_pip * 1.05) or (finger_span_tip < finger_span_pip * 1.25)


def finger_heart(points: np.ndarray) -> bool:
    palm_width = point_distance(points, 5, 17)
    thumb_index_close = point_distance(points, 4, 8) < palm_width * 0.34
    index_ready = point_distance(points, 0, 8) > point_distance(points, 0, 6) * 0.98
    thumb_ready = point_distance(points, 0, 4) > point_distance(points, 0, 3) * 0.98
    return (
        thumb_index_close
        and index_ready
        and thumb_ready
        and finger_curled(points, 12, 10, 9)
        and finger_curled(points, 16, 14, 13)
        and finger_curled(points, 20, 18, 17)
    )


def detect_gesture(hand_landmarks) -> str | None:
    points = landmarks_to_array(hand_landmarks)
    states = {
        "thumb": thumb_extended(points),
        "index": finger_extended(points, 8, 6, 5),
        "middle": finger_extended(points, 12, 10, 9),
        "ring": finger_extended(points, 16, 14, 13),
        "pinky": finger_extended(points, 20, 18, 17),
    }

    if all(states.values()):
        return "open_hand"
    if finger_heart(points):
        return "love_sign"
    if states["thumb"] and states["index"] and states["pinky"] and not states["middle"] and not states["ring"]:
        return "love_sign"
    if states["thumb"] and not states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
        return "thumbs_up"
    if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
        return "peace"
    return None


def camera_thread_func() -> None:
    cap, status = open_camera()
    if cap is None:
        with lock:
            shared_data["camera_status"] = status
        print(status, flush=True)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.solutions.hands
    drawing_utils = mp.solutions.drawing_utils
    last_candidate = None
    stable_frames = 0
    idle_frames = 0

    with lock:
        shared_data["camera_status"] = status

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            with lock:
                if not shared_data["running"]:
                    break
                gesture_control = shared_data["gesture_control"]

            ret, frame = cap.read()
            if not ret:
                with lock:
                    shared_data["camera_status"] = "Frame kamera gagal dibaca"
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            local_mode = None
            local_label = "NO HAND"
            local_status = status
            hand_x = 0.0
            hand_y = 0.0
            hand_z = -18.0

            if results.multi_hand_landmarks:
                idle_frames = 0
                hand_landmarks = results.multi_hand_landmarks[0]
                drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                gesture = detect_gesture(hand_landmarks)
                if gesture == last_candidate:
                    stable_frames += 1
                else:
                    last_candidate = gesture
                    stable_frames = 1

                if gesture is not None:
                    local_label = GESTURE_LABELS[gesture]
                    if gesture_control and stable_frames >= GESTURE_STABLE_FRAMES:
                        local_mode = GESTURE_TO_MODE[gesture]
                else:
                    local_label = "GESTURE TIDAK DIKENALI"

                points = landmarks_to_array(hand_landmarks)
                wrist = points[0]
                palm_span = point_distance(points, 0, 9)
                hand_x = (float(wrist[0]) - 0.5) * 8.0
                hand_y = -(float(wrist[1]) - 0.5) * 5.0
                hand_z = -17.5 + np.clip((0.22 - palm_span) * 18.0, -2.4, 2.8)
            else:
                idle_frames += 1
                stable_frames = 0
                last_candidate = None
                if gesture_control and idle_frames >= NO_HAND_IDLE_FRAMES:
                    local_mode = MODE_RANDOM
                    local_label = "NO HAND / IDLE"

            cv2.putText(
                frame,
                f"Gesture: {local_label}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 180),
                2,
            )
            cv2.putText(
                frame,
                "5 jari=Solar | Peace=PWN | Jempol=Imprup",
                (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 220, 160),
                1,
            )
            cv2.putText(
                frame,
                "Love: ILY sign atau finger heart (ibu jari + telunjuk)",
                (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 210),
                1,
            )
            control_text = "Gesture control: ON" if gesture_control else "Gesture control: OFF (manual)"
            cv2.putText(
                frame,
                control_text,
                (12, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (180, 220, 255),
                1,
            )

            with lock:
                if local_mode is not None:
                    shared_data["mode"] = local_mode
                shared_data["gesture_label"] = local_label
                shared_data["camera_status"] = local_status
                shared_data["target_x"] = hand_x
                shared_data["target_y"] = hand_y
                shared_data["target_z"] = hand_z
                shared_data["frame"] = frame

    cap.release()


def main() -> None:
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)

    setup_opengl()

    random_points = build_scatter(NUM_PARTICLES)
    random_colors = build_scatter_colors(random_points)
    pwn_points = build_text_points("I LOVE PWN", 165, NUM_PARTICLES)
    pwn_points_mirror = mirror_x_points(pwn_points)
    pwn_colors = repeat_color((1.0, 0.66, 0.18), NUM_PARTICLES)
    imprup_points = build_text_points("PLS IMPRUP", 155, NUM_PARTICLES)
    imprup_points_mirror = mirror_x_points(imprup_points)
    imprup_colors = repeat_color((0.46, 0.98, 0.7), NUM_PARTICLES)
    solar_builder = build_solar_system_builder(NUM_PARTICLES)
    love_builder = build_love_scene_builder(NUM_PARTICLES)
    love_confetti_builder = build_love_confetti()

    static_modes = {
        MODE_RANDOM: {"normal": (random_points, random_colors), "mirrored": (random_points, random_colors)},
        MODE_PWN: {"normal": (pwn_points, pwn_colors), "mirrored": (pwn_points_mirror, pwn_colors)},
        MODE_IMPRUP: {"normal": (imprup_points, imprup_colors), "mirrored": (imprup_points_mirror, imprup_colors)},
    }

    background_stars, background_colors = build_background_stars()
    current_points = build_scatter(NUM_PARTICLES)
    current_colors = build_scatter_colors(current_points)

    camera_thread = threading.Thread(target=camera_thread_func, daemon=True)
    camera_thread.start()
    time.sleep(0.4)

    clock = pygame.time.Clock()
    start_time = time.monotonic()
    auto_spin = 0.0
    hand_x = 0.0
    hand_y = 0.0
    hand_z = -18.0
    view_yaw = 0.0
    view_pitch = 0.0
    view_roll = 0.0
    view_offset_x = 0.0
    view_offset_y = 0.0
    view_offset_z = 0.0
    mirror_text = False
    current_mode = MODE_RANDOM
    previous_mode = current_mode
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_0:
                    with lock:
                        shared_data["mode"] = MODE_RANDOM
                        shared_data["gesture_control"] = False
                elif event.key == K_1:
                    with lock:
                        shared_data["mode"] = MODE_SOLAR
                        shared_data["gesture_control"] = False
                elif event.key == K_2:
                    with lock:
                        shared_data["mode"] = MODE_PWN
                        shared_data["gesture_control"] = False
                elif event.key == K_3:
                    with lock:
                        shared_data["mode"] = MODE_IMPRUP
                        shared_data["gesture_control"] = False
                elif event.key == K_4:
                    with lock:
                        shared_data["mode"] = MODE_HEART
                        shared_data["gesture_control"] = False
                elif event.key == K_g:
                    with lock:
                        shared_data["gesture_control"] = True
                elif event.key == K_LEFT:
                    view_yaw -= VIEW_ROTATE_STEP
                elif event.key == K_RIGHT:
                    view_yaw += VIEW_ROTATE_STEP
                elif event.key == K_UP:
                    view_pitch -= VIEW_ROTATE_STEP
                elif event.key == K_DOWN:
                    view_pitch += VIEW_ROTATE_STEP
                elif event.key == K_q:
                    view_roll -= VIEW_ROTATE_STEP
                elif event.key == K_e:
                    view_roll += VIEW_ROTATE_STEP
                elif event.key == K_a:
                    view_offset_x -= VIEW_MOVE_STEP
                elif event.key == K_d:
                    view_offset_x += VIEW_MOVE_STEP
                elif event.key == K_w:
                    view_offset_y += VIEW_MOVE_STEP
                elif event.key == K_s:
                    view_offset_y -= VIEW_MOVE_STEP
                elif event.key == K_z:
                    view_offset_z -= VIEW_Z_STEP
                elif event.key == K_x:
                    view_offset_z += VIEW_Z_STEP
                elif event.key == K_m:
                    mirror_text = not mirror_text
                elif event.key == K_f:
                    view_yaw = 0.0
                    view_pitch = 0.0
                    view_roll = 0.0
                elif event.key == K_r:
                    view_yaw = 0.0
                    view_pitch = 0.0
                    view_roll = 0.0
                    view_offset_x = 0.0
                    view_offset_y = 0.0
                    view_offset_z = 0.0
                    mirror_text = False

        with lock:
            current_mode = shared_data["mode"]
            target_hand_x = shared_data["target_x"]
            target_hand_y = shared_data["target_y"]
            target_hand_z = shared_data["target_z"]
            frame = shared_data["frame"]
            gesture_label = shared_data["gesture_label"]
            camera_status = shared_data["camera_status"]
            gesture_control = shared_data["gesture_control"]

        if current_mode != previous_mode:
            if current_mode in (MODE_SOLAR, MODE_PWN, MODE_IMPRUP, MODE_HEART):
                auto_spin = 0.0
            previous_mode = current_mode

        elapsed = time.monotonic() - start_time
        if current_mode == MODE_SOLAR:
            target_points, target_colors = solar_builder(elapsed)
        elif current_mode == MODE_HEART:
            target_points, target_colors = love_builder(elapsed, mirror_text)
        else:
            variant = "mirrored" if mirror_text and current_mode in (MODE_PWN, MODE_IMPRUP) else "normal"
            target_points, target_colors = static_modes[current_mode][variant]

        current_points += (target_points - current_points) * MORPH_SPEED
        current_colors += (target_colors - current_colors) * COLOR_MORPH_SPEED

        hand_x += (target_hand_x - hand_x) * 0.15
        hand_y += (target_hand_y - hand_y) * 0.15
        hand_z += (target_hand_z - hand_z) * 0.12

        auto_spin += AUTO_SPIN_SPEEDS[current_mode]

        pygame.display.set_caption(
            f"Nebula Love Pwn | {MODE_LABELS[current_mode]} | Gesture: {gesture_label} | "
            f"Control: {'Gesture' if gesture_control else 'Manual'} | Mirror: {'ON' if mirror_text else 'OFF'} | "
            f"Camera: {camera_status} | Arrows/WASD/ZX gerak | M mirror | R reset"
        )

        if frame is not None:
            cv2.putText(
                frame,
                f"Mode: {MODE_LABELS[current_mode]}",
                (12, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (120, 255, 120),
                2,
            )
            cv2.putText(
                frame,
                "Arrows/QE=rotasi | WASD/ZX=geser | M=mirror teks | R=reset",
                (12, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 220, 255),
                1,
            )
            cv2.putText(
                frame,
                camera_status,
                (12, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (170, 190, 255),
                1,
            )
            cv2.imshow("Camera Gesture Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False

        glClearColor(0.01, 0.02, 0.07, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glTranslatef(0.0, 0.0, -19.0)
        draw_points(background_stars, background_colors, 0.2)

        glTranslatef(hand_x + view_offset_x, hand_y + view_offset_y, hand_z + 19.0 + view_offset_z)
        glRotatef((20.0 if current_mode == MODE_SOLAR else 14.0) + view_pitch, 1.0, 0.0, 0.0)
        glRotatef(auto_spin + view_yaw, 0.0, 1.0, 0.0)
        glRotatef(view_roll, 0.0, 0.0, 1.0)

        if current_mode == MODE_RANDOM:
            glRotatef(math.sin(elapsed * 0.9) * 4.0, 0.0, 0.0, 1.0)
        elif current_mode == MODE_HEART:
            glRotatef(math.sin(elapsed * 0.8) * 3.6, 0.0, 0.0, 1.0)

        draw_points(current_points, current_colors, 0.96)

        if current_mode == MODE_HEART:
            confetti_points, confetti_colors = love_confetti_builder(elapsed)
            draw_points(confetti_points, confetti_colors, 0.88)

        pygame.display.flip()
        clock.tick(60)

    with lock:
        shared_data["running"] = False

    camera_thread.join(timeout=1.0)
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
