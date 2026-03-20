import config
import numpy as np
from PIL import Image


def normalize_density(layout):
    land_image = layout.land_image_display.get_image()
    if land_image is None:
        return

    w, h = land_image.size
    density = Image.new("L", (w, h), config.DEFAULT_DENSITY_GREY)
    layout.density_image = density

    layout.density_image_display.set_image(density.convert("RGBA"))
    layout.check_territory_ready()


def equator_density(layout):
    land_image = layout.land_image_display.get_image()
    if land_image is None:
        return

    w, h = land_image.size
    # Equirectangular correction: area per pixel ∝ cos(latitude).
    # Pixel weight = (256 - value), so dark = more provinces.
    # Set value = 255 * (1 - cos(lat)) so the equator (cos=1) is black
    # and the poles (cos=0) are white, scaled by actual area distortion.
    lats = np.linspace(np.pi / 2, -np.pi / 2, h)
    pixel_values = (255.0 * (1.0 - np.cos(lats))).astype(np.uint8)
    arr = np.tile(pixel_values[:, np.newaxis], (1, w))

    density = Image.fromarray(arr, mode="L")
    layout.density_image = density
    layout.density_image_display.set_image(density.convert("RGBA"))
    layout.check_territory_ready()
