from PIL import Image


def create_image_thumbnail(input_file, output_file, dimension=256):
    with Image.open(input_file) as img:
        img.thumbnail((dimension, dimension))
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            img = background

        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(output_file)
