"""Wall background generation with realistic colors, textures, and perspective transforms."""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .worker_utils import get_worker_np_rng

# Research-backed wall color palettes for museum/gallery settings
WALL_PALETTES: List[dict] = [
    {"name": "museum_white", "base": (245, 245, 245), "weight": 0.25},
    {"name": "gallery_beige", "base": (235, 220, 200), "weight": 0.20},
    {"name": "cool_gray", "base": (195, 195, 200), "weight": 0.20},
    {"name": "warm_gray", "base": (205, 200, 190), "weight": 0.15},
    {"name": "sage_hint", "base": (210, 220, 215), "weight": 0.10},
    {"name": "rich_accent", "base": (110, 100, 95), "weight": 0.10},
]


@dataclass
class WallConfig:
    """Configuration for wall generation."""

    # Wall dimensions
    width: int = 1920
    height: int = 1080

    # Perspective transform bounds (in degrees)
    min_angle: float = 5.0
    max_angle: float = 30.0

    # Image placement on wall (as fraction of wall size)
    min_image_scale: float = 0.40  # Minimum 40% of wall area
    max_image_scale: float = 0.75  # Maximum 75% of wall area

    # Position jitter (as fraction of available space)
    position_jitter: float = 0.15

    # Minimum visibility of original image content (0.0-1.0)
    min_visibility: float = 0.70

    # Texture/effects (noise-based)
    add_texture: bool = True
    texture_strength: float = 0.03  # Subtle noise

    # Image-based wall textures
    texture_chance: float = 0.0  # Probability of using image texture (0-1)
    texture_opacity: float = 0.3  # How strongly to blend texture (0-1)

    # Shadow
    add_shadow: bool = True
    shadow_opacity: float = 0.15
    shadow_offset: Tuple[int, int] = (8, 12)
    shadow_blur: int = 15

    # Lighting gradient (top-to-bottom brightness variation)
    add_lighting_gradient: bool = True
    gradient_strength: float = 0.08

    # Oversize handling to avoid post-warp borders
    pre_warp_scale: float = 2.0  # Generate wall larger than final target
    post_warp_overscan: float = 1.0  # Extra scale before final crop

    # Depth shading to emphasize perspective
    add_depth_shading: bool = True
    depth_shading_strength: float = 0.18


@dataclass
class WallGenerator:
    """Generates realistic wall backgrounds and places framed images on them."""

    config: WallConfig = field(default_factory=WallConfig)
    rng: random.Random = field(default_factory=random.Random)
    texture_images: List[Tuple[str, Optional[Image.Image]]] = field(
        default_factory=list
    )
    last_texture_name: Optional[str] = None
    last_perspective: Optional[dict] = None

    def select_wall_color(self) -> Tuple[int, int, int]:
        """Select a wall color from the curated palette with weighted probability."""
        weights = [p["weight"] for p in WALL_PALETTES]
        selected = self.rng.choices(WALL_PALETTES, weights=weights, k=1)[0]

        # Add slight random variation to the base color
        base = selected["base"]
        variation = 10
        return tuple(
            max(0, min(255, c + self.rng.randint(-variation, variation))) for c in base
        )

    def create_wall_background(
        self,
        color: Optional[Tuple[int, int, int]] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Create a wall background with optional texture and lighting gradient.

        Args:
            color: RGB color tuple, or None to select randomly from palette

        Returns:
            RGBA wall background image
        """
        if color is None:
            color = self.select_wall_color()

        wall_size = (
            size if size is not None else (self.config.width, self.config.height)
        )
        wall = Image.new("RGBA", wall_size, (*color, 255))

        # Add subtle noise texture (Perlin-like approximation)
        if self.config.add_texture:
            wall = self._add_texture(wall, self.config.texture_strength)

        # Apply image-based texture with configured chance
        if (
            self.texture_images
            and self.config.texture_chance > 0
            and self.rng.random() < self.config.texture_chance
        ):
            texture_name, texture_image = self.rng.choice(self.texture_images)
            self.last_texture_name = texture_name
            if texture_image is None:
                texture_image = self._generate_random_texture(wall.size, texture_name)
            wall = self._apply_image_texture(
                wall, texture_image, self.config.texture_opacity
            )
        else:
            self.last_texture_name = None

        # Add top-to-bottom lighting gradient
        if self.config.add_lighting_gradient:
            wall = self._add_lighting_gradient(wall, self.config.gradient_strength)

        return wall

    def _apply_image_texture(
        self, wall: Image.Image, texture: Image.Image, opacity: float
    ) -> Image.Image:
        """Apply an image-based texture overlay to the wall.

        Args:
            wall: Base wall image
            texture: Texture image to apply
            opacity: How strongly to blend the texture (0-1)

        Returns:
            Wall with texture applied
        """
        import numpy as np

        # Randomize orientation for more variety
        rotation = self.rng.choice([0, 90, 180, 270])
        if rotation:
            texture = texture.rotate(rotation, expand=True)
        if self.rng.random() < 0.5:
            texture = ImageOps.mirror(texture)
        if self.rng.random() < 0.5:
            texture = ImageOps.flip(texture)

        # Resize texture to match wall size (tile or stretch)
        tex_w, tex_h = texture.size
        wall_w, wall_h = wall.size

        # Tile texture if smaller, or resize if larger
        if tex_w < wall_w or tex_h < wall_h:
            # Tile the texture
            tiled = Image.new("RGB", (wall_w, wall_h))
            rows = range(0, wall_h, tex_h)
            cols = range(0, wall_w, tex_w)
            for i, y in enumerate(rows):
                for j, x in enumerate(cols):
                    # Flip tile based on grid position for seamless variation
                    tile = texture.copy()
                    if i % 2 != 0:
                        tile = ImageOps.flip(tile)
                    if j % 2 != 0:
                        tile = ImageOps.mirror(tile)
                    tiled.paste(tile.convert("RGB"), (x, y))
            texture_resized = tiled
        else:
            # Resize to fit
            texture_resized = texture.convert("RGB").resize(
                (wall_w, wall_h), Image.Resampling.LANCZOS
            )

        # Convert to arrays for blending
        wall_arr = np.array(wall, dtype=np.float32)
        tex_arr = np.array(texture_resized, dtype=np.float32)

        # Use overlay blend mode for texture - better contrast than soft-light
        # Overlay: darker areas get darker, lighter areas get lighter (balanced)
        wall_rgb = wall_arr[:, :, :3] / 255.0
        tex_rgb = tex_arr / 255.0

        # Overlay blending formula
        result = np.where(
            wall_rgb <= 0.5,
            2 * wall_rgb * tex_rgb,
            1 - 2 * (1 - wall_rgb) * (1 - tex_rgb),
        )

        # Blend with original based on opacity
        blended = wall_rgb * (1 - opacity) + result * opacity
        wall_arr[:, :, :3] = np.clip(blended * 255, 0, 255)

        return Image.fromarray(wall_arr.astype(np.uint8))

    def _generate_random_texture(
        self, size: Tuple[int, int], texture_name: str
    ) -> Image.Image:
        """Generate a procedural texture based on the requested variant."""
        if texture_name.endswith("002"):
            variant = 2
        else:
            variant = 1

        base_color = self.select_wall_color()
        texture = Image.new("RGBA", size, (*base_color, 255))

        if variant == 1:
            texture = self._add_texture(texture, self.config.texture_strength * 1.3)
            texture = self._add_lighting_gradient(
                texture, self.config.gradient_strength * 0.7
            )
            texture = texture.filter(ImageFilter.GaussianBlur(radius=1))
        else:
            texture = self._add_texture(texture, self.config.texture_strength * 2.0)
            texture = self._add_side_gradient(
                texture, self.config.gradient_strength * 0.9
            )
            texture = texture.filter(ImageFilter.GaussianBlur(radius=2))

        return texture.convert("RGB")

    @staticmethod
    def _soft_light_d(x):
        """Helper function for soft light blend mode."""
        import numpy as np

        return np.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, np.sqrt(x))

    def _add_texture(self, image: Image.Image, strength: float) -> Image.Image:
        """Add subtle noise texture to simulate wall surface."""
        import numpy as np

        arr = np.array(image, dtype=np.float32)
        noise = get_worker_np_rng().normal(0, strength * 255, arr.shape[:2])

        # Apply noise to RGB channels
        for c in range(3):
            arr[:, :, c] = np.clip(arr[:, :, c] + noise, 0, 255)

        return Image.fromarray(arr.astype(np.uint8))

    def _add_lighting_gradient(
        self, image: Image.Image, strength: float
    ) -> Image.Image:
        """Add subtle top-to-bottom brightness gradient."""
        import numpy as np

        arr = np.array(image, dtype=np.float32)
        h = arr.shape[0]

        # Create gradient: slightly brighter at top, darker at bottom
        gradient = np.linspace(1 + strength, 1 - strength, h)
        gradient = gradient.reshape(-1, 1, 1)

        # Apply to RGB channels only
        arr[:, :, :3] = np.clip(arr[:, :, :3] * gradient, 0, 255)

        return Image.fromarray(arr.astype(np.uint8))

    def _add_side_gradient(self, image: Image.Image, strength: float) -> Image.Image:
        """Add subtle left-to-right brightness gradient."""
        import numpy as np

        arr = np.array(image, dtype=np.float32)
        w = arr.shape[1]

        gradient = np.linspace(1 + strength, 1 - strength, w)
        gradient = gradient.reshape(1, -1, 1)

        arr[:, :, :3] = np.clip(arr[:, :, :3] * gradient, 0, 255)

        return Image.fromarray(arr.astype(np.uint8))

    def _apply_depth_shading(
        self, image: Image.Image, angle_x: float, angle_y: float
    ) -> Image.Image:
        """Darken the farthest warped corner to emphasize depth."""
        if not self.config.add_depth_shading or self.config.depth_shading_strength <= 0:
            return image

        arr = np.array(image, dtype=np.float32)
        h, w = arr.shape[0], arr.shape[1]
        strength = self.config.depth_shading_strength

        # Heuristic: determine farthest corner from perspective angles
        far_x = 0 if angle_y > 0 else w - 1
        far_y = h - 1 if angle_x > 0 else 0

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - far_x) ** 2 + (y - far_y) ** 2)
        max_dist = np.sqrt((w - 1) ** 2 + (h - 1) ** 2)
        if max_dist <= 0:
            return image

        norm = dist / max_dist
        factor = 1.0 - (strength * (1.0 - norm))
        factor = factor.reshape(h, w, 1)
        arr[:, :, :3] = np.clip(arr[:, :, :3] * factor, 0, 255)

        return Image.fromarray(arr.astype(np.uint8))

    def calculate_perspective_transform(
        self,
        image_size: Tuple[int, int],
        target_size: Tuple[int, int],
        angle_x: Optional[float] = None,
        angle_y: Optional[float] = None,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Calculate source and destination points for perspective transform.

        Args:
            image_size: (width, height) of source image
            target_size: (width, height) of target area on wall
            angle_x: Pitch angle in radians (random if None)
            angle_y: Yaw angle in radians (random if None)

        Returns:
            Tuple of (source_points, dest_points) for perspective transform
        """
        min_rad = math.radians(self.config.min_angle)
        max_rad = math.radians(self.config.max_angle)

        if angle_x is None:
            # Random angle with sign
            angle_x = self.rng.uniform(min_rad, max_rad)
            if self.rng.random() < 0.5:
                angle_x = -angle_x

        if angle_y is None:
            angle_y = self.rng.uniform(min_rad, max_rad)
            if self.rng.random() < 0.5:
                angle_y = -angle_y

        w, h = target_size

        # Source corners (original rectangle)
        src = [(0, 0), (w, 0), (w, h), (0, h)]

        # Apply perspective distortion
        # Simulate viewing from an angle
        cx, cy = w / 2, h / 2
        distance = max(w, h) * 1.5  # Virtual camera distance

        dst = []
        for x, y in src:
            # Center coordinates
            xc, yc = x - cx, y - cy

            # Apply rotation
            cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
            cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)

            # Rotate around X (pitch)
            y1 = yc * cos_x
            z1 = yc * sin_x

            # Rotate around Y (yaw)
            x2 = xc * cos_y + z1 * sin_y
            z2 = -xc * sin_y + z1 * cos_y

            # Project back to 2D
            z_shifted = z2 + distance
            if abs(z_shifted) < 1e-6:
                z_shifted = 1e-6

            scale = distance / z_shifted
            proj_x = x2 * scale + cx
            proj_y = y1 * scale + cy

            dst.append((proj_x, proj_y))

        # Normalize destination points to start from (0, 0)
        min_x = min(p[0] for p in dst)
        min_y = min(p[1] for p in dst)
        dst = [(p[0] - min_x, p[1] - min_y) for p in dst]

        return src, dst

    def apply_perspective(
        self,
        image: Image.Image,
        src_points: List[Tuple[float, float]],
        dst_points: List[Tuple[float, float]],
        fillcolor: Optional[Tuple[int, int, int, int]] = None,
    ) -> Image.Image:
        """Apply perspective transform to an image.

        Args:
            image: Source image
            src_points: Four source corner points
            dst_points: Four destination corner points

        Returns:
            Transformed image with transparency for background
        """
        # Calculate output size from destination points
        max_x = max(p[0] for p in dst_points)
        max_y = max(p[1] for p in dst_points)
        out_size = (int(math.ceil(max_x)), int(math.ceil(max_y)))

        # Calculate transform coefficients
        coeffs = self._find_perspective_coeffs(dst_points, src_points)

        # Ensure image is RGBA for transparency
        image = image.convert("RGBA")

        # Apply transform
        transformed = image.transform(
            out_size,
            Image.PERSPECTIVE,
            coeffs,
            resample=Image.BICUBIC,
            fillcolor=fillcolor if fillcolor is not None else (0, 0, 0, 0),
        )

        return transformed

    def apply_perspective_standalone(self, image: Image.Image) -> Image.Image:
        """Perspective-warp a single RGBA painting without placing it on a wall.

        Uses the configured ``min_angle``/``max_angle`` and this generator's
        RNG so the tilt matches wall variants, but returns just the warped
        painting on a transparent background (for off-wall archetypes).
        """
        image = image.convert("RGBA")
        w, h = image.size
        src, dst = self.calculate_perspective_transform((w, h), (w, h))
        self.last_perspective = self._perspective_metadata(src, dst, (w, h))
        return self.apply_perspective(image, src, dst)

    @staticmethod
    def _perspective_metadata(
        src: List[Tuple[float, float]],
        dst: List[Tuple[float, float]],
        input_size: Tuple[int, int],
    ) -> dict:
        def area(points: List[Tuple[float, float]]) -> float:
            return abs(sum(
                points[index][0] * points[(index + 1) % len(points)][1]
                - points[(index + 1) % len(points)][0] * points[index][1]
                for index in range(len(points))
            )) / 2.0

        source_area = max(1.0, area(src))
        return {
            "applied": True,
            "input_size": list(input_size),
            "source_corners": [[round(x, 6), round(y, 6)] for x, y in src],
            "destination_corners": [[round(x, 6), round(y, 6)] for x, y in dst],
            "visible_fraction": round(min(1.0, area(dst) / source_area), 6),
        }

    def _find_perspective_coeffs(
        self,
        src_points: List[Tuple[float, float]],
        dst_points: List[Tuple[float, float]],
    ) -> List[float]:
        """Solve coefficients for PIL perspective transform."""
        import numpy as np

        matrix = []
        for (x_src, y_src), (x_dst, y_dst) in zip(src_points, dst_points):
            matrix.append([x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src])
            matrix.append([0, 0, 0, x_src, y_src, 1, -y_dst * x_src, -y_dst * y_src])

        A = np.array(matrix, dtype=np.float64)
        B = np.array(
            [
                p
                for pair in zip([p[0] for p in dst_points], [p[1] for p in dst_points])
                for p in pair
            ],
            dtype=np.float64,
        )

        res = np.linalg.solve(A, B)
        return res.tolist()

    def create_shadow(
        self,
        image: Image.Image,
        offset: Optional[Tuple[int, int]] = None,
        blur_radius: Optional[int] = None,
        opacity: Optional[float] = None,
    ) -> Image.Image:
        """Create a soft shadow for an image.

        Args:
            image: Source image (RGBA, uses alpha channel for shape)
            offset: (x, y) shadow offset
            blur_radius: Gaussian blur radius
            opacity: Shadow opacity (0.0-1.0)

        Returns:
            Shadow image (RGBA) same size as input with offset applied
        """
        if offset is None:
            offset = self.config.shadow_offset
        if blur_radius is None:
            blur_radius = self.config.shadow_blur
        if opacity is None:
            opacity = self.config.shadow_opacity

        # Extract alpha channel as shadow shape
        alpha = image.split()[-1]

        # Create shadow (black with alpha from image)
        shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_alpha = alpha.point(lambda a: int(a * opacity))
        shadow.putalpha(shadow_alpha)

        # Apply blur
        shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))

        # Create larger canvas to accommodate offset
        w, h = image.size
        ox, oy = offset
        canvas = Image.new("RGBA", (w + abs(ox) * 2, h + abs(oy) * 2), (0, 0, 0, 0))

        # Paste shadow with offset
        shadow_pos = (max(0, ox) + abs(ox), max(0, oy) + abs(oy))
        canvas.paste(shadow, shadow_pos)

        return canvas, (abs(ox), abs(oy))

    def calculate_image_scale(
        self,
        image_size: Tuple[int, int],
        wall_size: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Calculate scale factor for placing image on wall.

        Ensures image occupies between min_image_scale and max_image_scale
        of the wall area.
        """
        img_w, img_h = image_size
        wall_w, wall_h = (
            wall_size
            if wall_size is not None
            else (self.config.width, self.config.height)
        )

        # Calculate scale to fit within bounds
        scale_w = wall_w / img_w
        scale_h = wall_h / img_h

        # Use the smaller scale to ensure image fits
        base_scale = min(scale_w, scale_h)

        # Random scale within configured range
        min_scale = base_scale * self.config.min_image_scale
        max_scale = base_scale * self.config.max_image_scale

        return self.rng.uniform(min_scale, max_scale)

    def calculate_position(
        self,
        wall_size: Tuple[int, int],
        image_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Calculate position to place image on wall.

        Centers with random jitter.
        """
        wall_w, wall_h = wall_size
        img_w, img_h = image_size

        # Center position
        center_x = (wall_w - img_w) // 2
        center_y = (wall_h - img_h) // 2

        # Calculate jitter range
        jitter_x = int(center_x * self.config.position_jitter)
        jitter_y = int(center_y * self.config.position_jitter)

        # Apply random jitter
        x = (
            center_x + self.rng.randint(-jitter_x, jitter_x)
            if jitter_x > 0
            else center_x
        )
        y = (
            center_y + self.rng.randint(-jitter_y, jitter_y)
            if jitter_y > 0
            else center_y
        )

        return max(0, x), max(0, y)

    def place_on_wall(
        self,
        framed_image: Image.Image,
        wall: Optional[Image.Image] = None,
        light_pos: Optional[Tuple[float, float, float]] = None,
    ) -> Image.Image:
        """Place a framed image on a wall background with perspective transform.

        Both the wall and the painting are transformed together with the same
        perspective to create a cohesive 3D appearance.

        Args:
            framed_image: The framed image to place
            wall: Optional pre-generated wall, or None to create one

        Returns:
            Final composited image
        """
        # Store original target size
        target_w, target_h = self.config.width, self.config.height

        if wall is None:
            oversize_w = max(1, int(target_w * self.config.pre_warp_scale))
            oversize_h = max(1, int(target_h * self.config.pre_warp_scale))
            wall = self.create_wall_background(size=(oversize_w, oversize_h))

        # Ensure RGBA
        framed_image = framed_image.convert("RGBA")
        wall = wall.convert("RGBA")

        # Calculate scale for image based on oversized wall
        scale = self.calculate_image_scale(framed_image.size, wall.size) * 0.5

        # Resize painting to fit on wall (no perspective yet)
        new_size = (
            int(framed_image.width * scale),
            int(framed_image.height * scale),
        )
        scaled_image = framed_image.resize(new_size, Image.LANCZOS)

        # Calculate position on wall (center the painting with jitter)
        wall_w, wall_h = wall.size
        pos_x = (wall_w - new_size[0]) // 2
        pos_y = (wall_h - new_size[1]) // 2

        # Add jitter based on available space
        jitter_x = int((wall_w - new_size[0]) * self.config.position_jitter * 0.5)
        jitter_y = int((wall_h - new_size[1]) * self.config.position_jitter * 0.5)
        pos_x += self.rng.randint(-jitter_x, jitter_x) if jitter_x > 0 else 0
        pos_y += self.rng.randint(-jitter_y, jitter_y) if jitter_y > 0 else 0
        position = (max(0, pos_x), max(0, pos_y))

        # Build a transparent layer for painting + shadow (no wall texture)
        painting_layer = Image.new("RGBA", wall.size, (0, 0, 0, 0))
        if self.config.add_shadow:
            shadow_img = self._create_flat_shadow(scaled_image)
            shadow_pos = (
                position[0] + self.config.shadow_offset[0],
                position[1] + self.config.shadow_offset[1],
            )
            painting_layer = Image.alpha_composite(
                painting_layer, self._paste_on_canvas(shadow_img, wall.size, shadow_pos)
            )
        painting_layer = Image.alpha_composite(
            painting_layer, self._paste_on_canvas(scaled_image, wall.size, position)
        )

        # Generate random perspective angles
        min_rad = math.radians(self.config.min_angle)
        max_rad = math.radians(self.config.max_angle)

        angle_x = self.rng.uniform(min_rad, max_rad)
        if self.rng.random() < 0.5:
            angle_x = -angle_x

        angle_y = self.rng.uniform(min_rad, max_rad)
        if self.rng.random() < 0.5:
            angle_y = -angle_y

        # Apply perspective to wall and painting with the same transform
        wall_src_pts, wall_dst_pts = self.calculate_perspective_transform(
            wall.size,
            wall.size,
            angle_x=angle_x,
            angle_y=angle_y,
        )
        self.last_perspective = self._perspective_metadata(
            wall_src_pts, wall_dst_pts, wall.size
        )
        self.last_perspective["angle_x_deg"] = round(math.degrees(angle_x), 6)
        self.last_perspective["angle_y_deg"] = round(math.degrees(angle_y), 6)
        wall_transformed = self.apply_perspective(wall, wall_src_pts, wall_dst_pts)
        painting_transformed = self.apply_perspective(
            painting_layer, wall_src_pts, wall_dst_pts
        )

        # Find a crop window fully inside the warped wall (no transparent borders)
        def find_valid_crop(mask, crop_w, crop_h, required_bbox=None):
            import numpy as np

            h, w = mask.shape
            if crop_w > w or crop_h > h:
                return None

            padded = np.pad(mask.astype(np.uint32), ((1, 0), (1, 0)))
            integral = padded.cumsum(axis=0).cumsum(axis=1)

            y2 = np.arange(crop_h, h + 1)
            x2 = np.arange(crop_w, w + 1)
            y1 = y2 - crop_h
            x1 = x2 - crop_w

            sum_windows = (
                integral[np.ix_(y2, x2)]
                - integral[np.ix_(y1, x2)]
                - integral[np.ix_(y2, x1)]
                + integral[np.ix_(y1, x1)]
            )

            valid = sum_windows == (crop_w * crop_h)
            if required_bbox is not None:
                min_x = required_bbox[2] - crop_w
                max_x = required_bbox[0]
                min_y = required_bbox[3] - crop_h
                max_y = required_bbox[1]
                if min_x > max_x or min_y > max_y:
                    return None
                x_min = max(0, int(math.ceil(min_x)))
                x_max = min(int(math.floor(max_x)), w - crop_w)
                y_min = max(0, int(math.ceil(min_y)))
                y_max = min(int(math.floor(max_y)), h - crop_h)
                if x_min > x_max or y_min > y_max:
                    return None
                valid_x = (x1 >= x_min) & (x1 <= x_max)
                valid_y = (y1 >= y_min) & (y1 <= y_max)
                valid = valid & valid_y[:, None] & valid_x[None, :]
            if not valid.any():
                return None

            ys, xs = np.where(valid)
            center_x = (w - crop_w) / 2.0
            center_y = (h - crop_h) / 2.0
            dx = xs - center_x
            dy = ys - center_y
            idx = (dx * dx + dy * dy).argmin()
            return int(xs[idx]), int(ys[idx])

        wall_alpha = np.array(wall_transformed.split()[-1]) > 0
        painting_bbox = painting_transformed.split()[-1].getbbox()
        crop_pos = find_valid_crop(wall_alpha, target_w, target_h, painting_bbox)

        wall_scaled = wall_transformed
        painting_scaled = painting_transformed
        if crop_pos is None:
            content_bbox = wall_transformed.split()[-1].getbbox()
            if content_bbox is None:
                content_bbox = (0, 0, wall_transformed.width, wall_transformed.height)
            bbox_w = max(1, content_bbox[2] - content_bbox[0])
            bbox_h = max(1, content_bbox[3] - content_bbox[1])

            scale_x = target_w / bbox_w
            scale_y = target_h / bbox_h
            cover_scale = max(scale_x, scale_y) * self.config.post_warp_overscan

            scaled_w = int(wall_transformed.width * cover_scale)
            scaled_h = int(wall_transformed.height * cover_scale)
            wall_scaled = wall_transformed.resize((scaled_w, scaled_h), Image.LANCZOS)
            painting_scaled = painting_transformed.resize(
                (scaled_w, scaled_h), Image.LANCZOS
            )

            wall_alpha_scaled = np.array(wall_scaled.split()[-1]) > 0
            if painting_bbox is not None:
                scale_x = scaled_w / wall_transformed.width
                scale_y = scaled_h / wall_transformed.height
                painting_bbox_scaled = (
                    int(round(painting_bbox[0] * scale_x)),
                    int(round(painting_bbox[1] * scale_y)),
                    int(round(painting_bbox[2] * scale_x)),
                    int(round(painting_bbox[3] * scale_y)),
                )
            else:
                painting_bbox_scaled = None
            crop_pos = find_valid_crop(
                wall_alpha_scaled, target_w, target_h, painting_bbox_scaled
            )
            if crop_pos is None:
                crop_pos = ((scaled_w - target_w) // 2, (scaled_h - target_h) // 2)

        crop_x, crop_y = crop_pos
        wall_cropped = wall_scaled.crop(
            (crop_x, crop_y, crop_x + target_w, crop_y + target_h)
        )
        painting_cropped = painting_scaled.crop(
            (crop_x, crop_y, crop_x + target_w, crop_y + target_h)
        )

        # Apply depth shading to wall only
        wall_shaded = self._apply_depth_shading(wall_cropped, angle_x, angle_y)

        # Apply frame shadow on wall using painting alpha (post-warp)
        wall_with_shadow = self._apply_frame_shadow(
            wall_shaded, painting_cropped, light_pos=light_pos
        )

        # Apply matching shadow onto the painting itself (subtle edge darkening)
        painting_with_shadow = self._apply_shadow_to_painting(
            painting_cropped, light_pos=light_pos
        )

        # Composite painting+frame over the transformed wall
        result = Image.alpha_composite(wall_with_shadow, painting_with_shadow)

        return result

    def _create_flat_shadow(self, image: Image.Image) -> Image.Image:
        """Create a simple drop shadow for a flat image."""
        # Create shadow from alpha channel
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        alpha = image.split()[-1]

        # Create black shadow with the image's shape
        shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_alpha = alpha.point(lambda x: int(x * self.config.shadow_opacity))
        shadow.putalpha(shadow_alpha)

        # Apply blur
        shadow = shadow.filter(ImageFilter.GaussianBlur(self.config.shadow_blur))

        return shadow

    def _apply_frame_shadow(
        self,
        wall: Image.Image,
        painting: Image.Image,
        light_pos: Optional[Tuple[float, float, float]] = None,
    ) -> Image.Image:
        """Apply a shadow from the painting alpha onto the wall."""
        if not self.config.add_shadow:
            return wall

        if wall.mode != "RGBA":
            wall = wall.convert("RGBA")
        if painting.mode != "RGBA":
            painting = painting.convert("RGBA")

        alpha = painting.split()[-1]
        shadow = Image.new("RGBA", painting.size, (0, 0, 0, 0))
        shadow_opacity, shadow_blur = self._shadow_settings(light_pos)
        shadow_alpha = alpha.point(lambda x: int(x * shadow_opacity))
        shadow.putalpha(shadow_alpha)
        shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))

        ox, oy = self._calculate_shadow_offset(light_pos)
        shadow_layer = Image.new("RGBA", wall.size, (0, 0, 0, 0))
        shadow_layer.paste(shadow, (ox, oy), shadow)

        return Image.alpha_composite(wall, shadow_layer)

    def _apply_shadow_to_painting(
        self,
        painting: Image.Image,
        light_pos: Optional[Tuple[float, float, float]] = None,
    ) -> Image.Image:
        """Apply a subtle shadow from the painting alpha onto itself."""
        if not self.config.add_shadow:
            return painting

        if painting.mode != "RGBA":
            painting = painting.convert("RGBA")

        alpha = painting.split()[-1]
        shadow = Image.new("RGBA", painting.size, (0, 0, 0, 0))
        shadow_opacity, shadow_blur = self._shadow_settings(light_pos)
        shadow_alpha = alpha.point(lambda x: int(x * shadow_opacity))
        shadow.putalpha(shadow_alpha)
        shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))

        ox, oy = self._calculate_shadow_offset(light_pos)
        shadow_layer = Image.new("RGBA", painting.size, (0, 0, 0, 0))
        shadow_layer.paste(shadow, (ox, oy), shadow)

        return Image.alpha_composite(painting, shadow_layer)

    def _calculate_shadow_offset(
        self, light_pos: Optional[Tuple[float, float, float]]
    ) -> Tuple[int, int]:
        """Compute shadow offset aligned opposite to the light source."""
        base_ox, base_oy = self.config.shadow_offset
        if light_pos is None:
            return base_ox, base_oy

        light_x, light_y, _ = light_pos
        dir_x = 0.5 - light_x
        dir_y = 0.5 - light_y
        norm = math.hypot(dir_x, dir_y)
        if norm < 1e-6:
            return base_ox, base_oy

        dir_x /= norm
        dir_y /= norm

        magnitude = math.hypot(base_ox, base_oy)
        ox = int(round(dir_x * magnitude))
        oy = int(round(dir_y * magnitude))
        return ox, oy

    def _shadow_settings(
        self, light_pos: Optional[Tuple[float, float, float]]
    ) -> Tuple[float, int]:
        """Return stronger, harder shadow settings when light is defined."""
        base_opacity = self.config.shadow_opacity
        base_blur = self.config.shadow_blur

        if light_pos is None:
            return base_opacity, base_blur

        boosted_opacity = min(0.45, base_opacity * 2.0)
        reduced_blur = max(1, int(round(base_blur * 0.55)))
        return boosted_opacity, reduced_blur

    def _get_average_color(self, image: Image.Image) -> Tuple[int, int, int, int]:
        """Get the average color of an image for background fill."""
        import numpy as np

        arr = np.array(image.convert("RGBA"))
        # Sample from corners to get wall color (avoid painting area)
        corners = [
            arr[0:50, 0:50],
            arr[0:50, -50:],
            arr[-50:, 0:50],
            arr[-50:, -50:],
        ]
        samples = np.concatenate([c.reshape(-1, 4) for c in corners])
        avg = samples.mean(axis=0).astype(int)
        return tuple(avg)

    def _paste_on_canvas(
        self,
        image: Image.Image,
        canvas_size: Tuple[int, int],
        position: Tuple[int, int],
    ) -> Image.Image:
        """Paste an image onto a transparent canvas at the given position."""
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

        # Handle case where image extends beyond canvas
        x, y = position
        if (
            x < 0
            or y < 0
            or x + image.width > canvas_size[0]
            or y + image.height > canvas_size[1]
        ):
            # Crop image to fit
            crop_left = max(0, -x)
            crop_top = max(0, -y)
            crop_right = min(image.width, canvas_size[0] - x)
            crop_bottom = min(image.height, canvas_size[1] - y)

            if crop_right <= crop_left or crop_bottom <= crop_top:
                return canvas  # Image completely outside canvas

            image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
            x = max(0, x)
            y = max(0, y)

        canvas.paste(image, (x, y))
        return canvas
