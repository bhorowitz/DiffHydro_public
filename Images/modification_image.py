from PIL import Image, ImageDraw, ImageFont
import os

img = Image.open("Images/output/big_bang_modified.png").convert("RGBA")
W, H = img.size  # 1430 x 820

# ── 1. Effacer la bande du bas ────────────────────────────────────────────
STRIP_TOP = H - 80
black_strip = Image.new("RGBA", (W, H - STRIP_TOP), (0, 0, 0, 255))
img.paste(black_strip, (0, STRIP_TOP))

# ── 2. Charger fontes ─────────────────────────────────────────────────────
def tryFont(path, size):
    try:
        return ImageFont.truetype(path, size)
    except:
        return ImageFont.load_default()

fBold  = tryFont("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
fMono  = tryFont("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
fSmall = tryFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)

draw = ImageDraw.Draw(img)

# ── 2b. Fonction pour centrer du texte multi-ligne ───────────────────────
def draw_multiline_centered(draw, x_center, y_top, text, font, fill, spacing=2):
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    text_w = bbox[2] - bbox[0]
    draw.multiline_text(
        (x_center - text_w / 2, y_top),
        text,
        fill=fill,
        font=font,
        spacing=spacing,
        align="center"
    )

# ── 3. Définition des périodes ───────────────────────────────────────────
periods = [
    # x0, x1, statut, bg_rgba, border_color, label_bas, equation
    (0, 250,
     "IONIZED",
     (20, 50, 110, 190),
     (120, 170, 255),
     "Pre-stellar era",
     "H⁺ + e⁻ → H + γ"),

    (250, 450,
     "NEUTRAL",
     (20, 50, 110, 190),
     (120, 170, 255),
     "Cosmic Wave background",
     ""),

    (450, 770,
     "RE-IONIZED",
     (130, 35, 10, 200),
     (255, 120, 50),
     "Epoch of Reionization",
     "H → H⁺ + e⁻"),

    (770, 1410,
     "LATE UNIVERSE",
     (130, 35, 10, 200),
     (255, 120, 50),
     "Modern Universe",
     "IGM ionized"),
]

# ── 4. Coordonnées Y de la bande ──────────────────────────────────────────
BAR_Y1  = STRIP_TOP + 4
BAR_Y2  = STRIP_TOP + 38
EQ_Y    = STRIP_TOP + 44
LINE_Y  = STRIP_TOP + 62
NAME_Y  = STRIP_TOP + 66

# ── 5. Dessiner chaque période ────────────────────────────────────────────
for x0, x1, statut, bg, fg, name, eq in periods:
    bloc = Image.new("RGBA", (x1 - x0, BAR_Y2 - BAR_Y1), bg)
    img.alpha_composite(bloc, (x0, BAR_Y1))

    draw = ImageDraw.Draw(img)
    draw.rectangle([(x0, BAR_Y1), (x1 - 1, BAR_Y2 - 1)], outline=fg + (255,), width=2)

    cx = (x0 + x1) // 2

    # statut = mono-ligne, anchor OK
    draw.text((cx, (BAR_Y1 + BAR_Y2) // 2), statut, fill=fg + (255,), font=fBold, anchor="mm")

    # équation = potentiellement multi-ligne
    if eq.strip():
        draw_multiline_centered(
            draw, cx, EQ_Y, eq,
            font=fMono,
            fill=(200, 220, 255, 255),
            spacing=1
        )

draw = ImageDraw.Draw(img)

# Ligne de séparation en bas
draw.line([(0, LINE_Y), (W, LINE_Y)], fill=(70, 100, 150, 255), width=1)

# Noms d'époques = multi-ligne
for x0, x1, statut, bg, fg, name, eq in periods:
    cx = (x0 + x1) // 2
    draw_multiline_centered(
        draw, cx, NAME_Y, name,
        font=fSmall,
        fill=(160, 185, 225, 255),
        spacing=1
    )

# Séparateurs verticaux alignés sur les zones définies
draw.line([(250, STRIP_TOP), (250, H)], fill=(80, 110, 160, 140), width=1)
draw.line([(450, STRIP_TOP), (450, H)], fill=(200, 160, 80, 200), width=2)
draw.line([(450, STRIP_TOP), (450, H)], fill=(80, 110, 160, 140), width=1)
draw.line([(770, STRIP_TOP), (770, H)], fill=(80, 110, 160, 140), width=1)

# Trait supérieur
draw.line([(0, STRIP_TOP), (W, STRIP_TOP)], fill=(70, 120, 10, 255), width=1)

# ── 6. Sauvegarder ────────────────────────────────────────────────────────
os.makedirs("Images/output", exist_ok=True)
out = "Images/output/big_bang_edited.png"
img.convert("RGB").save(out, quality=95)
print("Done:", out, img.size)