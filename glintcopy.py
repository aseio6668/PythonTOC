# glyphify - image to ascii art
# depends pip install click pillow

import click
import numpy as np
from PIL import Image

# Density-ordered ASCII characters from darkest to lightest
ASCII_DENSITY = "@%#*+=-:. "
ASCII_CHARS_LEVEL_1 = "@#S%?*+;:,. "  # 10 chars, high contrast
ASCII_CHARS_LEVEL_2 = "@%#*+=-:. "  # 10 chars, classic
ASCII_CHARS_LEVEL_3 = "@%#*+=-:.^`'. "  # 15 chars, more gradation
ASCII_CHARS_LEVEL_4 = "@%#S&$B8WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|()1{}[]?-_+~<>i!lI;:,^`'. "  # 50+ chars, high detail
ASCII_CHARS_LEVEL_5 = "@%#S&$B8WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|()1{}[]?-_+~<>i!lI;:,\"^`'. "  # 70+ chars, most accurate

@click.command()
@click.argument("image_path")
@click.option("--width", default=100, help="Set ASCII art width")
@click.option("--output", default=None, help="Save ASCII art to a file")
@click.option("--level", default=1, help="Level of detail (1-3 or custom)")
@click.option("--charset", default=None, help="Custom ASCII character set (overrides level)")
@click.option("--brightness", default=1.0, type=float, help="Adjust brightness (e.g. 1.2 for brighter, 0.8 for darker)")
@click.option("--dither/--no-dither", default=False, help="Enable Floyd–Steinberg dithering")
@click.option("--edge/--no-edge", default=False, help="Enable edge detection (Sobel filter)")
def image_to_ascii(image_path, width, level, output, charset, brightness, dither, edge):
    img=Image.open(image_path).convert("L") # convert to greyscale
    
    # Adjust brightness
    if brightness != 1.0:
        img = Image.fromarray(np.clip(np.array(img) * brightness, 0, 255).astype(np.uint8))
    
    # Edge detection (Sobel)
    if edge:
        from scipy.ndimage import sobel
        arr = np.array(img, dtype=np.float32)
        sx = sobel(arr, axis=0, mode='constant')
        sy = sobel(arr, axis=1, mode='constant')
        arr = np.hypot(sx, sy)
        arr = np.clip(arr / arr.max() * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    
    # resize image and maintain aspect ratio
    aspect_ratio=img.height/img.width
    new_height=int(width*aspect_ratio*0.55) # adjust for proportions
    img=img.resize((width,new_height))
    
    # Set ASCII character set
    if charset:
        ASCII_CHARS = charset
    else:
        if level==1:
            ASCII_CHARS=ASCII_CHARS_LEVEL_1
        elif level==2:
            ASCII_CHARS=ASCII_CHARS_LEVEL_2
        elif level==3:
            ASCII_CHARS=ASCII_CHARS_LEVEL_3
        elif level==4:
            ASCII_CHARS=ASCII_CHARS_LEVEL_4
        elif level==5:
            ASCII_CHARS=ASCII_CHARS_LEVEL_5
        else:
            ASCII_CHARS=ASCII_CHARS_LEVEL_1
    
    pixels = np.array(img, dtype=np.float32)
    if dither:
        # Floyd–Steinberg dithering
        for y in range(pixels.shape[0]):
            for x in range(pixels.shape[1]):
                old_pixel = pixels[y, x]
                new_pixel = round((old_pixel / 255) * (len(ASCII_CHARS)-1)) * (255/(len(ASCII_CHARS)-1))
                pixels[y, x] = new_pixel
                quant_error = old_pixel - new_pixel
                if x+1 < pixels.shape[1]:
                    pixels[y, x+1] += quant_error * 7/16
                if y+1 < pixels.shape[1]:
                    if x > 0:
                        pixels[y+1, x-1] += quant_error * 3/16
                    pixels[y+1, x] += quant_error * 5/16
                    if x+1 < pixels.shape[1]:
                        pixels[y+1, x+1] += quant_error * 1/16
        pixels = np.clip(pixels, 0, 255)
    
    ascii_image = "\n".join(
        "".join(ASCII_CHARS[min(int(pixel//(255//(len(ASCII_CHARS)-1))),len(ASCII_CHARS)-1)] for pixel in row)
        for row in pixels
    )
    # print or save the output
    if output:
        with open(output, "w") as f:
            f.write(ascii_image)
        print(f"ASCII art saved to {output}")
    else:
        print(ascii_image)

if __name__=="__main__":
    image_to_ascii()