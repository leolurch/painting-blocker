import numpy as np
from PIL import Image

frame = Image.open("frame3.png").convert("RGBA")
photo = Image.open("braunes_pferd.jpg").convert("RGBA")

frame_w, frame_h = frame.size

# Detection of the irregular frame opening (alpha area < 255)
alpha = np.array(frame.split()[-1])
transparent_mask = alpha < 255

ys, xs = np.where(transparent_mask)

min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()

inner_w = max_x - min_x
inner_h = max_y - min_y

# Fit the photo with minimal cropping (cover)
photo_w, photo_h = photo.size

# key change -> minimal crop
ratio = max(inner_w / photo_w, inner_h / photo_h)

new_w = int(photo_w * ratio)
new_h = int(photo_h * ratio)

photo_resized = photo.resize((new_w, new_h), Image.LANCZOS)

# Crop the photo to the size of the window
# Centering the crop
crop_x = (new_w - inner_w) // 2
crop_y = (new_h - inner_h) // 2

photo_cropped = photo_resized.crop((crop_x, crop_y, crop_x + inner_w, crop_y + inner_h))

# Insert the photo into the window
background = Image.new("RGBA", frame.size, (0, 0, 0, 0))
background.paste(photo_cropped, (min_x, min_y))

# Composition
composite = Image.alpha_composite(background, frame)
composite.save("output.png")
