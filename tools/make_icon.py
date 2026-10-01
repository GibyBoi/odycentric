"""Draws assets/odycentric.ico and .png: a person cut out against a transparency checkerboard."""

from pathlib import Path

from PIL import Image, ImageDraw

S = 1024  # draw big, scale down for each icon size
OUT = Path(__file__).resolve().parent.parent / "assets"

icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)
draw.rounded_rectangle((32, 32, S - 32, S - 32), radius=220, fill=(24, 28, 48, 255))

# checkerboard disc: the "O", and the universal sign for "transparent"
disc = Image.new("RGBA", (S, S), (0, 0, 0, 0))
cells = ImageDraw.Draw(disc)
cell = 80
for y in range(0, S, cell):
    for x in range(0, S, cell):
        shade = 235 if (x // cell + y // cell) % 2 else 185
        cells.rectangle((x, y, x + cell, y + cell), fill=(shade, shade, shade, 255))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).ellipse((150, 150, S - 150, S - 150), fill=255)
icon.paste(disc, (0, 0), mask)

# the person, clipped to the disc
person = Image.new("L", (S, S), 0)
p = ImageDraw.Draw(person)
p.ellipse((392, 270, 632, 510), fill=255)  # head
p.ellipse((262, 560, 762, 1060), fill=255)  # shoulders
person = Image.composite(person, Image.new("L", (S, S), 0), mask)
icon.paste(Image.new("RGBA", (S, S), (255, 122, 48, 255)), (0, 0), person)

# ring
draw = ImageDraw.Draw(icon)
draw.ellipse((150, 150, S - 150, S - 150), outline=(255, 122, 48, 255), width=34)

OUT.mkdir(exist_ok=True)
icon.resize((256, 256), Image.LANCZOS).save(OUT / "odycentric.png")
icon.save(OUT / "odycentric.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("wrote", OUT)
