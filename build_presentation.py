"""
HELIOS-AI Final Presentation Builder
Builds a 10-slide professional PPTX using the team15.pptx template.
"""
import os, copy, math
from pptx import Presentation
from pptx.util import Inches, Pt, Emu, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ──── Paths ────
BASE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(BASE, "team15.pptx")
OUTPUT = os.path.join(BASE, "presentation_output", "SolarPanelsDefectDetector_Final_Presentation.pptx")
RESULTS = os.path.join(BASE, "Results")
DOCS = os.path.join(BASE, "docs", "images")

# ──── Design System (matches template) ────
NAVY = RGBColor(0x0F, 0x17, 0x2A)        # primary text
SLATE = RGBColor(0x64, 0x74, 0x8B)        # secondary text  
MED_SLATE = RGBColor(0x47, 0x55, 0x69)    # body text
LIGHT_SLATE = RGBColor(0x94, 0xA3, 0xB8)  # captions
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CARD_BG = RGBColor(0xF8, 0xFA, 0xFC)      # light card background
ACCENT_ORANGE = RGBColor(0xF9, 0x73, 0x16) # solar/energy accent
ACCENT_BLUE = RGBColor(0x38, 0x7E, 0xF5)  # tech accent
ACCENT_RED = RGBColor(0xEF, 0x44, 0x44)   # critical severity
ACCENT_AMBER = RGBColor(0xF5, 0x9E, 0x0B) # medium severity
ACCENT_GREEN = RGBColor(0x22, 0xC5, 0x5E) # positive/good
ACCENT_TEAL = RGBColor(0x14, 0xB8, 0xA6)  # supporting accent
DARK_BG = RGBColor(0x1E, 0x29, 0x3B)      # dark card background

FONT_HEADING = "Space Grotesk"
FONT_MONO = "IBM Plex Mono"
FONT_BODY = "Space Grotesk"

# Slide dimensions (from template)
SLIDE_W = 9144000  # 10 inches
SLIDE_H = 5143500  # 5.625 inches
MARGIN_L = Emu(407194)
MARGIN_T = Emu(321469)
CONTENT_W = Emu(8329800)

def emu(inches):
    return Emu(int(inches * 914400))

# ──── Helpers ────
def set_shape_fill(shape, color):
    """Set solid fill on a shape."""
    fill = shape.fill
    fill.solid()
    fill.fore_color.rgb = color

def set_shape_no_fill(shape):
    """Remove fill from shape."""
    shape.fill.background()

def set_shape_border(shape, color, width_pt=1):
    """Set border/outline on a shape."""
    line = shape.line
    line.color.rgb = color
    line.width = Pt(width_pt)

def set_no_border(shape):
    """Remove border from shape."""
    shape.line.fill.background()

def add_textbox(slide, left, top, width, height, text="", font_name=FONT_BODY,
                font_size=Pt(12), bold=False, color=NAVY, alignment=PP_ALIGN.LEFT,
                word_wrap=True):
    """Add a text box with a single paragraph."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = word_wrap
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = font_name
    p.font.size = font_size
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = alignment
    return txBox

def add_run(paragraph, text, font_name=FONT_BODY, font_size=Pt(12), bold=False, color=NAVY):
    """Add a run to an existing paragraph."""
    run = paragraph.add_run()
    run.text = text
    run.font.name = font_name
    run.font.size = font_size
    run.font.bold = bold
    run.font.color.rgb = color
    return run

def add_multiline_textbox(slide, left, top, width, height, lines, 
                          font_name=FONT_BODY, font_size=Pt(11), color=MED_SLATE,
                          spacing_before=Pt(2), spacing_after=Pt(2), bullet=False,
                          bold_first_word=False):
    """Add a text box with multiple paragraphs."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        
        if bullet:
            p.text = "• " + line
        else:
            p.text = line
        
        p.font.name = font_name
        p.font.size = font_size
        p.font.color.rgb = color
        p.space_before = spacing_before
        p.space_after = spacing_after
    return txBox

def add_card(slide, left, top, width, height, fill_color=CARD_BG, border_color=None, 
             corner_radius=None):
    """Add a rounded rectangle card."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    set_shape_fill(shape, fill_color)
    if border_color:
        set_shape_border(shape, border_color, 1)
    else:
        set_no_border(shape)
    # Adjust corner radius via XML if needed
    if corner_radius is not None:
        # corner_radius in EMU
        sp = shape._element
        prstGeom = sp.find(qn('a:prstGeom'), sp.nsmap) if hasattr(sp, 'nsmap') else None
        # Use default radius
    return shape

def add_accent_line(slide, left, top, width, color=ACCENT_ORANGE, height=Emu(21300)):
    """Add a horizontal accent line."""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    set_shape_fill(shape, color)
    set_no_border(shape)
    return shape

def add_footer_bar(slide, text_left="HELIOS-AI | Solar Panel Defect Detection", text_right="Team 15"):
    """Add a consistent footer bar."""
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN_L, Emu(4443413), CONTENT_W, Emu(378600))
    set_shape_fill(bar, CARD_BG)
    set_no_border(bar)
    
    add_textbox(slide, Emu(585788), Emu(4520000), Emu(5400000), Emu(153900),
                text_left, FONT_MONO, Pt(8), False, SLATE)
    add_textbox(slide, Emu(7200000), Emu(4520000), Emu(1500000), Emu(153900),
                text_right, FONT_MONO, Pt(8), False, SLATE, PP_ALIGN.RIGHT)

def add_slide_title(slide, title_text, subtitle_text=None):
    """Add consistent title at top of slide."""
    add_textbox(slide, MARGIN_L, MARGIN_T, CONTENT_W, Emu(400000),
                title_text, FONT_HEADING, Pt(24), True, NAVY)
    add_accent_line(slide, MARGIN_L, Emu(680000), Emu(571500))
    if subtitle_text:
        add_textbox(slide, MARGIN_L, Emu(720000), CONTENT_W, Emu(200000),
                    subtitle_text, FONT_BODY, Pt(11), False, SLATE)

def add_pipeline_arrow(slide, x, y, width=Emu(180000), color=ACCENT_BLUE):
    """Add a right-pointing arrow."""
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, width, Emu(120000))
    set_shape_fill(arrow, color)
    set_no_border(arrow)
    return arrow

def add_pipeline_box(slide, x, y, w, h, title, subtitle="", fill=ACCENT_BLUE, text_color=WHITE):
    """Add a pipeline stage box with title and optional subtitle."""
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    set_shape_fill(box, fill)
    set_no_border(box)
    tf = box.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    run.font.name = FONT_HEADING
    run.font.size = Pt(8)
    run.font.bold = True
    run.font.color.rgb = text_color
    
    if subtitle:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        run2 = p2.add_run()
        run2.text = subtitle
        run2.font.name = FONT_BODY
        run2.font.size = Pt(6.5)
        run2.font.bold = False
        run2.font.color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
    
    # Center vertically
    tf.paragraphs[0].space_before = Pt(4)
    return box

def add_stat_card(slide, x, y, w, h, value, label, value_color=NAVY, bg_color=CARD_BG):
    """Add a stat card with a large number and label."""
    card = add_card(slide, x, y, w, h, bg_color)
    
    add_textbox(slide, x + Emu(60000), y + Emu(40000), w - Emu(120000), Emu(200000),
                value, FONT_MONO, Pt(16), True, value_color, PP_ALIGN.CENTER)
    add_textbox(slide, x + Emu(60000), y + Emu(230000), w - Emu(120000), Emu(120000),
                label, FONT_BODY, Pt(7), False, SLATE, PP_ALIGN.CENTER)
    return card


# ════════════════════════════════════════════════════════════
# SLIDE BUILDERS
# ════════════════════════════════════════════════════════════

def build_slide_1_intro(prs):
    """Slide 1: Project Introduction"""
    layout = prs.slide_layouts[0]  # BLANK
    slide = prs.slides.add_slide(layout)
    
    # Main content card
    main_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, 
        MARGIN_L, Emu(685800), CONTENT_W, Emu(3571800))
    set_shape_fill(main_card, CARD_BG)
    set_no_border(main_card)
    
    # Footer bar
    footer_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
        MARGIN_L, Emu(4443413), CONTENT_W, Emu(378600))
    set_shape_fill(footer_bar, CARD_BG)
    set_no_border(footer_bar)
    
    # Orange accent line
    add_accent_line(slide, MARGIN_L, Emu(571500), Emu(571500))
    
    # Department/course info
    add_textbox(slide, MARGIN_L, MARGIN_T, Emu(5000000), Emu(123000),
                "B.Tech Computer Science | IPCV Project | MNNIT Allahabad", 
                FONT_MONO, Pt(8), True, SLATE)
    
    # Project title
    title_box = add_textbox(slide, Emu(580000), Emu(1100000), Emu(7800000), Emu(600000),
                "HELIOS-AI", FONT_HEADING, Pt(36), True, NAVY)
    
    # Subtitle
    add_textbox(slide, Emu(580000), Emu(1680000), Emu(7800000), Emu(350000),
                "Autonomous Aerial Thermography &\nPhotovoltaic Anomaly Intelligence",
                FONT_HEADING, Pt(16), False, MED_SLATE)
    
    # Brief description
    desc_box = slide.shapes.add_textbox(Emu(580000), Emu(2150000), Emu(4800000), Emu(400000))
    tf = desc_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "AI-powered thermal inspection system that detects 9 types of solar panel faults using YOLOv9, applies classical IPCV analysis, grades defects per IEC TS 62446-3 standards, and estimates yield loss — all from a single drone-captured thermal image."
    p.font.name = FONT_BODY
    p.font.size = Pt(9)
    p.font.color.rgb = MED_SLATE
    
    # Tech stack badges row
    badges = ["YOLOv9", "OpenCV", "FastAPI", "IEC 62446-3", "Python"]
    badge_x = Emu(580000)
    for badge_text in badges:
        bw = Emu(len(badge_text) * 75000 + 200000)
        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, badge_x, Emu(2680000), bw, Emu(180000))
        set_shape_fill(badge, DARK_BG)
        set_no_border(badge)
        tf = badge.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = badge_text
        r.font.name = FONT_MONO
        r.font.size = Pt(7)
        r.font.bold = True
        r.font.color.rgb = WHITE
        badge_x += bw + Emu(80000)
    
    # Right side — cockpit screenshot
    cockpit_img = os.path.join(DOCS, "cockpit.jpg")
    if os.path.exists(cockpit_img):
        slide.shapes.add_picture(cockpit_img, Emu(5600000), Emu(2900000), Emu(3000000), Emu(1250000))
    
    # At-a-glance stats row
    stats = [
        ("9", "Fault Classes"),
        ("25.5M", "Parameters"),
        ("~0.42s", "Full Pipeline"),
        ("IEC", "Compliant")
    ]
    stat_x = Emu(580000)
    stat_w = Emu(1200000)
    for val, label in stats:
        # small stat box
        sbox = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, stat_x, Emu(3100000), stat_w, Emu(400000))
        set_shape_fill(sbox, WHITE)
        set_shape_border(sbox, RGBColor(0xE2, 0xE8, 0xF0), 1)
        
        add_textbox(slide, stat_x + Emu(40000), Emu(3120000), stat_w - Emu(80000), Emu(180000),
                    val, FONT_MONO, Pt(14), True, ACCENT_ORANGE, PP_ALIGN.CENTER)
        add_textbox(slide, stat_x + Emu(40000), Emu(3320000), stat_w - Emu(80000), Emu(100000),
                    label, FONT_BODY, Pt(7), False, SLATE, PP_ALIGN.CENTER)
        stat_x += stat_w + Emu(100000)
    
    # Team info
    add_textbox(slide, Emu(585788), Emu(4550569), Emu(6072300), Emu(153900),
                "Teammates: Arpit Shukla, Ashish Gautam, Amit Kumar Bhorayat, Alok Kumar Shukla",
                FONT_BODY, Pt(8), True, NAVY)
    add_textbox(slide, Emu(6822281), Emu(4550569), Emu(2169900), Emu(153900),
                "Mentor: Dr. Divyansh Thakur", FONT_BODY, Pt(8), True, NAVY, PP_ALIGN.RIGHT)


def build_slide_2_problem(prs):
    """Slide 2: Problem Statement & Motivation"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    
    add_slide_title(slide, "Problem Statement & Motivation",
                    "Why automated thermal defect detection matters")
    
    # Left column — The Problem
    col1_x = MARGIN_L
    col1_w = Emu(3800000)
    
    add_textbox(slide, col1_x, Emu(900000), col1_w, Emu(200000),
                "The Problem", FONT_HEADING, Pt(14), True, ACCENT_RED)
    
    problems = [
        "Solar farms lose energy and revenue from invisible faults — failed bypass diodes, cracked cells, reversed polarity",
        "These faults are invisible in the visible spectrum; only infrared thermography reveals them",
        "A single drone flight produces thousands of thermal frames that must be manually interpreted",
        "Manual inspection is slow, subjective, and cannot scale to utility-size solar plants"
    ]
    add_multiline_textbox(slide, col1_x, Emu(1100000), col1_w, Emu(1800000),
                         problems, FONT_BODY, Pt(9), MED_SLATE, bullet=True)
    
    # Right column — Why It Matters
    col2_x = Emu(4800000)
    col2_w = Emu(3900000)
    
    add_textbox(slide, col2_x, Emu(900000), col2_w, Emu(200000),
                "Why It Matters", FONT_HEADING, Pt(14), True, ACCENT_GREEN)
    
    motivations = [
        "Undetected defects cause progressive degradation — and in worst cases, fire hazards (Class 3 faults)",
        "Early detection saves maintenance cost and prevents catastrophic damage",
        "IEC TS 62446-3 standard exists for outdoor PV thermography, but compliance requires expert thermographers",
        "Automated AI-based detection can make inspection faster, consistent, and accessible"
    ]
    add_multiline_textbox(slide, col2_x, Emu(1100000), col2_w, Emu(1800000),
                         motivations, FONT_BODY, Pt(9), MED_SLATE, bullet=True)
    
    # Bottom impact cards
    impacts = [
        ("$425/yr", "Revenue loss from\n9 defects on one frame"),
        ("2.16 kW", "Power dissipated by\ndetected anomalies"),
        ("9 Classes", "Distinct PV fault types\ndetectable by the system"),
        ("< 0.5s", "Complete analysis\nper thermal frame")
    ]
    card_x = MARGIN_L
    card_w = Emu(2000000)
    card_gap = Emu(110000)
    for val, desc in impacts:
        card = add_card(slide, card_x, Emu(3200000), card_w, Emu(700000), DARK_BG)
        add_textbox(slide, card_x + Emu(100000), Emu(3250000), card_w - Emu(200000), Emu(250000),
                    val, FONT_MONO, Pt(16), True, ACCENT_ORANGE, PP_ALIGN.LEFT)
        add_textbox(slide, card_x + Emu(100000), Emu(3520000), card_w - Emu(200000), Emu(300000),
                    desc, FONT_BODY, Pt(7.5), False, LIGHT_SLATE, PP_ALIGN.LEFT)
        card_x += card_w + card_gap
    
    add_footer_bar(slide)


def build_slide_3_solution(prs):
    """Slide 3: Our Proposed Solution"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    
    add_slide_title(slide, "Our Proposed Solution",
                    "HELIOS-AI: From a single thermal image to an auditable maintenance decision")
    
    # High-level flow: Input → Processing → Output
    # Three column cards
    col_w = Emu(2600000)
    col_gap = Emu(200000)
    col_y = Emu(950000)
    col_h = Emu(2600000)
    
    # Column 1: Input
    x1 = MARGIN_L
    c1 = add_card(slide, x1, col_y, col_w, col_h, CARD_BG, RGBColor(0xE2, 0xE8, 0xF0))
    
    add_textbox(slide, x1 + Emu(100000), col_y + Emu(80000), col_w - Emu(200000), Emu(200000),
                "📷  INPUT", FONT_HEADING, Pt(12), True, ACCENT_BLUE, PP_ALIGN.CENTER)
    
    input_items = [
        "Aerial thermal image from drone (JPEG, PNG, 16-bit TIFF, HEIC)",
        "Single LWIR frame (e.g. 640×512)",
        "Configurable parameters: confidence threshold, module wattage, tariff, peak sun hours"
    ]
    add_multiline_textbox(slide, x1 + Emu(100000), col_y + Emu(350000), col_w - Emu(200000), Emu(1800000),
                         input_items, FONT_BODY, Pt(8.5), MED_SLATE, bullet=True)
    
    # Arrow 1→2
    add_pipeline_arrow(slide, x1 + col_w + Emu(20000), Emu(2100000), Emu(160000), ACCENT_ORANGE)
    
    # Column 2: Processing
    x2 = x1 + col_w + col_gap
    c2 = add_card(slide, x2, col_y, col_w, col_h, CARD_BG, RGBColor(0xE2, 0xE8, 0xF0))
    
    add_textbox(slide, x2 + Emu(100000), col_y + Emu(80000), col_w - Emu(200000), Emu(200000),
                "⚙️  PROCESSING", FONT_HEADING, Pt(12), True, ACCENT_ORANGE, PP_ALIGN.CENTER)
    
    proc_items = [
        "Classical IPCV: CLAHE + Ironbow, Sobel gradients, Otsu binarisation",
        "YOLOv9 deep detection: 9 fault classes with class-agnostic NMS",
        "Module geometry: contour detection, sub-pixel corners, homography rectification",
        "IEC TS 62446-3 severity grading, ΔT estimation, and thermal contrast measurement",
        "Economic analysis: power loss (W), energy loss (kWh/yr), revenue loss ($/yr)"
    ]
    add_multiline_textbox(slide, x2 + Emu(100000), col_y + Emu(350000), col_w - Emu(200000), Emu(2100000),
                         proc_items, FONT_BODY, Pt(8), MED_SLATE, bullet=True)
    
    # Arrow 2→3
    add_pipeline_arrow(slide, x2 + col_w + Emu(20000), Emu(2100000), Emu(160000), ACCENT_ORANGE)
    
    # Column 3: Output
    x3 = x2 + col_w + col_gap
    c3 = add_card(slide, x3, col_y, col_w, col_h, CARD_BG, RGBColor(0xE2, 0xE8, 0xF0))
    
    add_textbox(slide, x3 + Emu(100000), col_y + Emu(80000), col_w - Emu(200000), Emu(200000),
                "📊  OUTPUT", FONT_HEADING, Pt(12), True, ACCENT_GREEN, PP_ALIGN.CENTER)
    
    output_items = [
        "6 visualization layers (raw, Ironbow, Sobel, Otsu, contours, detection overlay)",
        "Defect register with IEC severity class, ΔT, power loss per defect",
        "Plant KPIs: health score, kW dissipated, annual revenue loss",
        "Interactive cockpit with guided tour mode",
        "Printable 2-page A4 audit certificate"
    ]
    add_multiline_textbox(slide, x3 + Emu(100000), col_y + Emu(350000), col_w - Emu(200000), Emu(1800000),
                         output_items, FONT_BODY, Pt(8.5), MED_SLATE, bullet=True)
    
    # Bottom: key differentiator
    diff_card = add_card(slide, MARGIN_L, Emu(3700000), CONTENT_W, Emu(550000), DARK_BG)
    
    tb = slide.shapes.add_textbox(Emu(580000), Emu(3770000), Emu(7800000), Emu(400000))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r1 = p.add_run()
    r1.text = "Key Differentiator:  "
    r1.font.name = FONT_HEADING
    r1.font.size = Pt(10)
    r1.font.bold = True
    r1.font.color.rgb = ACCENT_ORANGE
    r2 = p.add_run()
    r2.text = "HELIOS-AI is not a model wrapper. The YOLOv9 detector is one of six pipeline stages. Classical IPCV, module geometry, domain physics (IEC grading), economics, and interactive visualization each produce their own verifiable outputs."
    r2.font.name = FONT_BODY
    r2.font.size = Pt(9)
    r2.font.bold = False
    r2.font.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
    
    add_footer_bar(slide)


def build_slide_4_architecture(prs):
    """Slide 4: System Architecture — editable PowerPoint diagram"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    
    add_slide_title(slide, "System Architecture",
                    "End-to-end pipeline from thermal image upload to interactive audit report")
    
    # ── Architecture diagram built with native shapes ──
    # Top row: Client components
    # Middle: API layer
    # Bottom: Core modules
    
    box_h = Emu(400000)
    small_box_h = Emu(350000)
    
    # ─── Row 1: Browser / Client ───
    y1 = Emu(920000)
    
    # Section label
    label_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN_L, y1, Emu(1200000), Emu(280000))
    set_shape_fill(label_card, ACCENT_BLUE)
    set_no_border(label_card)
    tf = label_card.text_frame
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run()
    r.text = "BROWSER"
    r.font.name = FONT_MONO
    r.font.size = Pt(8)
    r.font.bold = True
    r.font.color.rgb = WHITE
    
    client_boxes = [
        ("Cockpit UI", "Layers · Pins\nZoom/Pan"),
        ("Presenter Mode", "Guided Tour\nEngine"),
        ("IPCV Lab", "Histogram\nHomography"),
        ("Audit Certificate", "2-page A4\nPrint/PDF"),
    ]
    cx = Emu(1750000)
    cw = Emu(1650000)
    for title, sub in client_boxes:
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, y1, cw, Emu(280000))
        set_shape_fill(box, RGBColor(0xDB, 0xEA, 0xFE))
        set_shape_border(box, ACCENT_BLUE, 1)
        tf = box.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run()
        r.text = title
        r.font.name = FONT_HEADING
        r.font.size = Pt(7.5)
        r.font.bold = True
        r.font.color.rgb = NAVY
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = sub
        r2.font.name = FONT_BODY
        r2.font.size = Pt(6)
        r2.font.color.rgb = SLATE
        cx += cw + Emu(60000)
    
    # ─── Down arrows from client to API ───
    y_arrow1 = y1 + Emu(280000)
    for ax in [Emu(2500000), Emu(5800000)]:
        arr = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, ax, y_arrow1, Emu(100000), Emu(140000))
        set_shape_fill(arr, SLATE)
        set_no_border(arr)
    
    # ─── Row 2: FastAPI Service ───
    y2 = Emu(1450000)
    
    api_label = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN_L, y2, Emu(1200000), Emu(280000))
    set_shape_fill(api_label, ACCENT_ORANGE)
    set_no_border(api_label)
    tf = api_label.text_frame
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run()
    r.text = "FASTAPI"
    r.font.name = FONT_MONO
    r.font.size = Pt(8)
    r.font.bold = True
    r.font.color.rgb = WHITE
    
    api_boxes = [
        ("api.py", "/inspect · /sample\n/rectify · /health"),
        ("Validation", "Size · Format\nRange checks"),
        ("Thread Pool", "run_in_threadpool\nModel lock"),
    ]
    ax = Emu(1750000)
    aw = Emu(2100000)
    for title, sub in api_boxes:
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, ax, y2, aw, Emu(280000))
        set_shape_fill(box, RGBColor(0xFF, 0xED, 0xD5))
        set_shape_border(box, ACCENT_ORANGE, 1)
        tf = box.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run()
        r.text = title
        r.font.name = FONT_HEADING
        r.font.size = Pt(7.5)
        r.font.bold = True
        r.font.color.rgb = NAVY
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = sub
        r2.font.name = FONT_BODY
        r2.font.size = Pt(6)
        r2.font.color.rgb = SLATE
        ax += aw + Emu(80000)
    
    # ─── Down arrows from API to Core ───
    y_arrow2 = y2 + Emu(280000)
    for ax_pos in [Emu(2700000), Emu(5000000)]:
        arr = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, ax_pos, y_arrow2, Emu(100000), Emu(140000))
        set_shape_fill(arr, SLATE)
        set_no_border(arr)
    
    # ─── Row 3: Vision & Analytics Core ───
    y3 = Emu(1970000)
    
    core_label = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN_L, y3, Emu(1200000), Emu(280000))
    set_shape_fill(core_label, ACCENT_TEAL)
    set_no_border(core_label)
    tf = core_label.text_frame
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run()
    r.text = "CORE"
    r.font.name = FONT_MONO
    r.font.size = Pt(8)
    r.font.bold = True
    r.font.color.rgb = WHITE
    
    core_boxes = [
        ("predictor.py", "Pipeline Orchestration\nStages 0-6"),
        ("solar_iec_analyzer.py", "IPCV Filters · IEC Grading\nKPIs · Economics"),
        ("ipcv_geometry.py", "Contours · Homography\nRectification"),
    ]
    cx = Emu(1750000)
    cw = Emu(2100000)
    for title, sub in core_boxes:
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, y3, cw, Emu(280000))
        set_shape_fill(box, RGBColor(0xCC, 0xFB, 0xF1))
        set_shape_border(box, ACCENT_TEAL, 1)
        tf = box.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run()
        r.text = title
        r.font.name = FONT_HEADING
        r.font.size = Pt(7.5)
        r.font.bold = True
        r.font.color.rgb = NAVY
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = sub
        r2.font.name = FONT_BODY
        r2.font.size = Pt(6)
        r2.font.color.rgb = SLATE
        cx += cw + Emu(80000)
    
    # ─── Row 4: Model & Weights ───
    y4 = Emu(2450000)
    
    # Down arrow to model
    arr = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Emu(2700000), y3 + Emu(280000), Emu(100000), Emu(120000))
    set_shape_fill(arr, SLATE)
    set_no_border(arr)
    
    model_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(1750000), y4, Emu(2100000), Emu(280000))
    set_shape_fill(model_box, DARK_BG)
    set_no_border(model_box)
    tf = model_box.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run()
    r.text = "model_loader.py"
    r.font.name = FONT_HEADING
    r.font.size = Pt(7.5)
    r.font.bold = True
    r.font.color.rgb = WHITE
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = "YOLOv9 + Inference Lock"
    r2.font.name = FONT_BODY
    r2.font.size = Pt(6)
    r2.font.color.rgb = LIGHT_SLATE
    
    # Weights cylinder-like box
    weights_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(4100000), y4, Emu(2200000), Emu(280000))
    set_shape_fill(weights_box, RGBColor(0x7C, 0x3A, 0xED))
    set_no_border(weights_box)
    tf = weights_box.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run()
    r.text = "Th_G_v9.pt"
    r.font.name = FONT_MONO
    r.font.size = Pt(8)
    r.font.bold = True
    r.font.color.rgb = WHITE
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = "25.5M params · 103.7 GFLOPs"
    r2.font.name = FONT_BODY
    r2.font.size = Pt(6)
    r2.font.color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
    
    # Arrow between model_loader and weights
    arr2 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Emu(3900000), y4 + Emu(80000), Emu(160000), Emu(120000))
    set_shape_fill(arr2, SLATE)
    set_no_border(arr2)
    
    # ─── Side box: Benchmark ───
    bench_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(6600000), y4, Emu(2100000), Emu(280000))
    set_shape_fill(bench_box, CARD_BG)
    set_shape_border(bench_box, SLATE, 1)
    tf = bench_box.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run()
    r.text = "evaluate_benchmark.py"
    r.font.name = FONT_HEADING
    r.font.size = Pt(7)
    r.font.bold = True
    r.font.color.rgb = NAVY
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = "Latency · FLOPs · Robustness"
    r2.font.name = FONT_BODY
    r2.font.size = Pt(6)
    r2.font.color.rgb = SLATE
    
    # ─── Bottom: Data flow summary ───
    flow_y = Emu(2900000)
    add_textbox(slide, MARGIN_L, flow_y, CONTENT_W, Emu(200000),
                "Data Flow Summary", FONT_HEADING, Pt(11), True, NAVY)
    
    # Pipeline stages as a horizontal flow
    stages = [
        ("Stage 0", "Ingestion\n& Normalise", ACCENT_BLUE),
        ("Stage 1", "Classical\nIPCV", ACCENT_TEAL),
        ("Stage 2", "YOLOv9\nDetection", RGBColor(0x7C, 0x3A, 0xED)),
        ("Stage 3", "Module\nGeometry", ACCENT_BLUE),
        ("Stage 4", "IEC 62446-3\nEnrichment", ACCENT_ORANGE),
        ("Stage 5", "Plant KPIs\n& Economics", ACCENT_GREEN),
        ("Stage 6", "Evidence\nPackaging", DARK_BG),
    ]
    
    sx = Emu(500000)
    sw = Emu(1050000)
    sy = Emu(3150000)
    sh = Emu(500000)
    gap = Emu(80000)
    arrow_w = Emu(60000)
    
    for i, (label, desc, color) in enumerate(stages):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, sx, sy, sw, sh)
        set_shape_fill(box, color)
        set_no_border(box)
        tf = box.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run()
        r.text = label
        r.font.name = FONT_MONO
        r.font.size = Pt(7)
        r.font.bold = True
        r.font.color.rgb = WHITE
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(6.5)
        r2.font.color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
        
        if i < len(stages) - 1:
            arr = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, 
                sx + sw + Emu(5000), sy + Emu(180000), arrow_w, Emu(100000))
            set_shape_fill(arr, SLATE)
            set_no_border(arr)
        
        sx += sw + gap
    
    # Output box
    out_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(500000), Emu(3750000), Emu(8200000), Emu(350000))
    set_shape_fill(out_box, CARD_BG)
    set_shape_border(out_box, RGBColor(0xE2, 0xE8, 0xF0), 1)
    
    tb = slide.shapes.add_textbox(Emu(600000), Emu(3770000), Emu(8000000), Emu(300000))
    tff = tb.text_frame
    tff.word_wrap = True
    p = tff.paragraphs[0]
    r1 = p.add_run()
    r1.text = "Output:  "
    r1.font.name = FONT_HEADING
    r1.font.size = Pt(8)
    r1.font.bold = True
    r1.font.color.rgb = NAVY
    r2 = p.add_run()
    r2.text = "JSON API response (~0.7 MB) with 6 base64 image layers, enriched defect register, IPCV statistics, geometry data, KPIs, and per-stage timing telemetry"
    r2.font.name = FONT_BODY
    r2.font.size = Pt(7.5)
    r2.font.color.rgb = MED_SLATE
    
    add_footer_bar(slide)


def build_slide_5_detection(prs):
    """Slide 5: Core Detection Methodology"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    
    add_slide_title(slide, "Core Detection Methodology",
                    "YOLOv9 thermal detection with class-agnostic NMS and IEC severity mapping")
    
    # Left: Detection pipeline
    left_w = Emu(4200000)
    
    add_textbox(slide, MARGIN_L, Emu(900000), left_w, Emu(200000),
                "Detection Pipeline", FONT_HEADING, Pt(12), True, NAVY)
    
    # Visual pipeline — vertical flow
    pipe_stages = [
        ("Thermal Frame Input", "Aspect-true resize (long side 640px)\nEXIF transpose · 16-bit percentile stretch", ACCENT_BLUE),
        ("YOLOv9 (GELAN) Forward Pass", "25.5M params · Pre-trained on PV thermal data\nConf ≥ 0.20 candidate floor", RGBColor(0x7C, 0x3A, 0xED)),
        ("Class-Agnostic NMS", "IoU 0.7 — prevents double-counting\nsame module under different labels", ACCENT_ORANGE),
        ("IEC TS 62446-3 Enrichment", "Severity grading · ΔT estimation\nPower loss · Stable defect IDs", ACCENT_RED),
    ]
    
    py = Emu(1120000)
    pw = Emu(3800000)
    ph = Emu(350000)
    
    for i, (title, desc, color) in enumerate(pipe_stages):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(500000), py, pw, ph)
        set_shape_fill(box, color)
        set_no_border(box)
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = title
        r.font.name = FONT_HEADING
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = WHITE
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = desc
        r2.font.name = FONT_BODY
        r2.font.size = Pt(7)
        r2.font.color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
        
        if i < len(pipe_stages) - 1:
            arr = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Emu(2200000), py + ph, Emu(100000), Emu(80000))
            set_shape_fill(arr, SLATE)
            set_no_border(arr)
            py += ph + Emu(80000)
        else:
            py += ph
    
    # Right: 9 fault classes table
    right_x = Emu(4700000)
    right_w = Emu(4000000)
    
    add_textbox(slide, right_x, Emu(900000), right_w, Emu(200000),
                "9 Detected Fault Classes", FONT_HEADING, Pt(12), True, NAVY)
    
    # Class table
    faults = [
        ("StringReversedPolarity", "Class 3", "100%", ACCENT_RED),
        ("StringOpenCircuit", "Class 3", "100%", ACCENT_RED),
        ("MultiDiode", "Class 3", "85%", ACCENT_RED),
        ("MultiHotSpot", "Class 3", "45%", ACCENT_RED),
        ("MultiByPassed", "Class 2", "66.6%", ACCENT_AMBER),
        ("SingleDiode", "Class 2", "33.3%", ACCENT_AMBER),
        ("SingleHotSpot", "Class 2", "20%", ACCENT_AMBER),
        ("OpenPV", "Class 2", "100%", ACCENT_AMBER),
        ("SingleByPassed", "Class 1", "33.3%", ACCENT_BLUE),
    ]
    
    # Table header
    ty = Emu(1120000)
    header_bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, right_x, ty, right_w, Emu(180000))
    set_shape_fill(header_bg, DARK_BG)
    set_no_border(header_bg)
    
    cols_x = [right_x + Emu(60000), right_x + Emu(1700000), right_x + Emu(2600000), right_x + Emu(3300000)]
    cols_w = [Emu(1600000), Emu(850000), Emu(650000), Emu(650000)]
    headers = ["Fault Class", "IEC Severity", "Loss %", "Scope"]
    for cx, cw, header in zip(cols_x, cols_w, headers):
        add_textbox(slide, cx, ty + Emu(20000), cw, Emu(150000),
                    header, FONT_MONO, Pt(7), True, WHITE)
    
    # Table rows
    ty += Emu(180000)
    scopes = ["string", "string", "module", "module", "module", "module", "module", "module", "module"]
    
    for i, (name, iec, loss, color) in enumerate(faults):
        row_bg_color = CARD_BG if i % 2 == 0 else WHITE
        row_bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, right_x, ty, right_w, Emu(170000))
        set_shape_fill(row_bg, row_bg_color)
        set_no_border(row_bg)
        
        # Severity dot
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, cols_x[0] - Emu(10000), ty + Emu(55000), Emu(60000), Emu(60000))
        set_shape_fill(dot, color)
        set_no_border(dot)
        
        add_textbox(slide, cols_x[0] + Emu(60000), ty + Emu(15000), cols_w[0] - Emu(60000), Emu(140000),
                    name, FONT_MONO, Pt(6.5), False, NAVY)
        add_textbox(slide, cols_x[1], ty + Emu(15000), cols_w[1], Emu(140000),
                    iec, FONT_BODY, Pt(7), True, color)
        add_textbox(slide, cols_x[2], ty + Emu(15000), cols_w[2], Emu(140000),
                    loss, FONT_MONO, Pt(7), False, NAVY)
        add_textbox(slide, cols_x[3], ty + Emu(15000), cols_w[3], Emu(140000),
                    scopes[i], FONT_BODY, Pt(7), False, SLATE)
        ty += Emu(170000)
    
    # Bottom note
    add_textbox(slide, right_x, ty + Emu(30000), right_w, Emu(250000),
                "Class 3 = Critical (≤24h) · Class 2 = Medium (≤30 days) · Class 1 = Monitor",
                FONT_BODY, Pt(7.5), False, SLATE)
    
    add_footer_bar(slide)


def build_slide_6_thermal_ipcv(prs):
    """Slide 6: Classical IPCV & Thermal Analysis"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    
    add_slide_title(slide, "Classical IPCV & Thermal Analysis",
                    "Foundational image processing and domain physics beyond deep learning")
    
    # ─── Top row: 3 IPCV technique cards ───
    card_w = Emu(2650000)
    card_h = Emu(1500000)
    card_gap = Emu(120000)
    cy = Emu(920000)
    
    techniques = [
        ("CLAHE + Ironbow", ACCENT_ORANGE, [
            "Contrast-Limited Adaptive Histogram Equalization",
            "Clip limit 2.5, 8×8 tile grid",
            "Mapped to FLIR Inferno palette — standard thermographer view",
            "Reveals subtle temperature gradients invisible in raw grayscale"
        ]),
        ("Sobel Gradient Magnitude", ACCENT_TEAL, [
            "3×3 Sobel kernel extracts |∇I| = √(Gx² + Gy²)",
            "Highlights busbars, cell edges, and thermal transitions",
            "Min-max normalised, rendered in Viridis palette",
            "Exposes internal cell structure and fracture lines"
        ]),
        ("Otsu Binarisation", ACCENT_BLUE, [
            "Otsu's method maximises between-class variance σ²B",
            "Independent NumPy re-derivation cross-checked with OpenCV",
            "Sample frame: t* = 103, separability η = 0.82",
            "Cleanly segments hot module surfaces from cool background"
        ]),
    ]
    
    cx = MARGIN_L
    for title, color, items in techniques:
        card = add_card(slide, cx, cy, card_w, card_h, WHITE, RGBColor(0xE2, 0xE8, 0xF0))
        
        # Color accent top bar
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, cx, cy, card_w, Emu(30000))
        set_shape_fill(bar, color)
        set_no_border(bar)
        
        add_textbox(slide, cx + Emu(80000), cy + Emu(60000), card_w - Emu(160000), Emu(200000),
                    title, FONT_HEADING, Pt(10), True, NAVY)
        add_multiline_textbox(slide, cx + Emu(80000), cy + Emu(280000), card_w - Emu(160000), Emu(1100000),
                             items, FONT_BODY, Pt(7.5), MED_SLATE, bullet=True)
        cx += card_w + card_gap
    
    # ─── Bottom section: Thermal Signature + Geometry ───
    bottom_y = Emu(2550000)
    half_w = Emu(4050000)
    
    # Left: Thermal Signature Measurement
    ts_card = add_card(slide, MARGIN_L, bottom_y, half_w, Emu(1100000), CARD_BG, RGBColor(0xE2, 0xE8, 0xF0))
    
    add_textbox(slide, MARGIN_L + Emu(80000), bottom_y + Emu(40000), Emu(3800000), Emu(200000),
                "Thermal Signature Measurement", FONT_HEADING, Pt(10), True, ACCENT_RED)
    
    ts_items = [
        "Two independent ΔT figures reported separately:",
        "  1. Estimated ΔT: class-conditional prior scaled by confidence",
        "  2. Image-measured: hotspot P98 vs Otsu-segmented healthy reference",
        "Annulus-based reference avoids cold ground inflating contrast",
        "Contrast-to-Noise Ratio (CNR) quantifies detection reliability"
    ]
    add_multiline_textbox(slide, MARGIN_L + Emu(80000), bottom_y + Emu(260000), Emu(3800000), Emu(800000),
                         ts_items, FONT_BODY, Pt(7.5), MED_SLATE, bullet=False)
    
    # Right: Module Geometry & Homography
    geo_x = MARGIN_L + half_w + Emu(200000)
    geo_card = add_card(slide, geo_x, bottom_y, half_w, Emu(1100000), CARD_BG, RGBColor(0xE2, 0xE8, 0xF0))
    
    add_textbox(slide, geo_x + Emu(80000), bottom_y + Emu(40000), Emu(3800000), Emu(200000),
                "Module Geometry & Homography", FONT_HEADING, Pt(10), True, ACCENT_BLUE)
    
    geo_items = [
        "Otsu mask → findContours → approxPolyDP (ε sweep)",
        "Least-squares edge refinement (Huber fitLine per side)",
        "Corners = robust line intersections (sub-pixel accuracy)",
        "4-point DLT homography → warpPerspective (640×640)",
        "Validated: 0.48 px mean corner error on synthetic 28° oblique"
    ]
    add_multiline_textbox(slide, geo_x + Emu(80000), bottom_y + Emu(260000), Emu(3800000), Emu(800000),
                         geo_items, FONT_BODY, Pt(7.5), MED_SLATE, bullet=True)
    
    # ─── Image statistics card ───
    stats_y = Emu(3780000)
    stats_items = [
        ("η = 0.82", "Otsu\nSeparability"),
        ("t* = 103", "Otsu\nThreshold"),
        ("7.53 bits", "Shannon\nEntropy"),
        ("63.7%", "Hot Pixel\nFraction"),
        ("60.87", "RMS\nContrast"),
        ("10%", "Edge\nDensity"),
    ]
    sx = MARGIN_L
    sw = Emu(1350000)
    for val, label in stats_items:
        scard = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, sx, stats_y, sw, Emu(350000))
        set_shape_fill(scard, DARK_BG)
        set_no_border(scard)
        add_textbox(slide, sx + Emu(30000), stats_y + Emu(20000), sw - Emu(60000), Emu(160000),
                    val, FONT_MONO, Pt(12), True, ACCENT_ORANGE, PP_ALIGN.CENTER)
        add_textbox(slide, sx + Emu(30000), stats_y + Emu(190000), sw - Emu(60000), Emu(130000),
                    label, FONT_BODY, Pt(6.5), False, LIGHT_SLATE, PP_ALIGN.CENTER)
        sx += sw + Emu(50000)
    
    add_footer_bar(slide)


def build_slide_7_demo(prs):
    """Slide 7: Implementation & Practical Demonstration"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    
    add_slide_title(slide, "Implementation & Practical Demonstration",
                    "The interactive cockpit — a complete inspection workstation")
    
    # Main cockpit screenshot
    cockpit_img = os.path.join(DOCS, "cockpit.jpg")
    if os.path.exists(cockpit_img):
        slide.shapes.add_picture(cockpit_img, Emu(500000), Emu(920000), Emu(4800000), Emu(2000000))
        add_textbox(slide, Emu(500000), Emu(2950000), Emu(4800000), Emu(150000),
                    "Cockpit: Ironbow layer with severity pins, KPI ribbon, and defect telemetry",
                    FONT_BODY, Pt(7), False, SLATE, PP_ALIGN.CENTER)
    
    # IPCV lab screenshot
    lab_img = os.path.join(DOCS, "ipcv-lab.jpg")
    if os.path.exists(lab_img):
        slide.shapes.add_picture(lab_img, Emu(500000), Emu(3200000), Emu(4800000), Emu(900000))
        add_textbox(slide, Emu(500000), Emu(4120000), Emu(4800000), Emu(150000),
                    "IPCV Lab: Interactive histogram, Otsu analysis, homography, economics",
                    FONT_BODY, Pt(7), False, SLATE, PP_ALIGN.CENTER)
    
    # Right side: Feature cards
    feat_x = Emu(5500000)
    feat_w = Emu(3200000)
    
    features = [
        ("🖥️ Interactive Cockpit", [
            "6 cross-fading visualization layers",
            "Severity-coded pins with reticle lock",
            "Zoom/pan (1-6×) with counter-scaled pins",
            "Click any pin → telemetry drawer: 4-modality macro crops, ΔT, SLA, field action"
        ], ACCENT_BLUE),
        ("⚡ Guided Tour Mode", [
            "Hands-free automated presentation",
            "Walks through IPCV layers + every defect",
            "Narrated severity, ΔT, power loss per defect",
            "Pause/resume, prev/next, speed control"
        ], ACCENT_ORANGE),
        ("📄 Audit Certificate", [
            "2-page A4 printable inspection report",
            "IEC TS 62446-3 compliance stamp",
            "Defect register with work order",
            "Thermographer & O&M sign-off blocks"
        ], ACCENT_GREEN),
        ("📊 Data Exports", [
            "CSV defect register export",
            "Full JSON inspection data export",
            "RESTful API with OpenAPI docs"
        ], ACCENT_TEAL),
    ]
    
    fy = Emu(920000)
    fh = Emu(680000)
    
    for title, items, color in features:
        # Accent bar
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, feat_x, fy, Emu(30000), fh)
        set_shape_fill(bar, color)
        set_no_border(bar)
        
        add_textbox(slide, feat_x + Emu(60000), fy + Emu(10000), feat_w - Emu(100000), Emu(180000),
                    title, FONT_HEADING, Pt(9), True, NAVY)
        add_multiline_textbox(slide, feat_x + Emu(60000), fy + Emu(170000), feat_w - Emu(100000), Emu(480000),
                             items, FONT_BODY, Pt(7), MED_SLATE, bullet=True,
                             spacing_before=Pt(1), spacing_after=Pt(1))
        fy += fh + Emu(60000)
    
    add_footer_bar(slide)


def build_slide_8_results(prs):
    """Slide 8: Evaluation & Results"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    
    add_slide_title(slide, "Evaluation & Results",
                    "Benchmark performance, detection results, and robustness analysis")
    
    # ─── Left: Performance metrics ───
    left_w = Emu(4200000)
    
    add_textbox(slide, MARGIN_L, Emu(900000), left_w, Emu(180000),
                "Performance Benchmarks (CPU, 20 iterations)", FONT_HEADING, Pt(10), True, NAVY)
    
    # Latency table
    ty = Emu(1100000)
    perf_data = [
        ("HELIOS Preprocessing", "3.1 ms", "3.6 ms"),
        ("YOLO Letterbox/Tensor", "1.5 ms", "2.5 ms"),
        ("Neural Net Forward Pass", "336.4 ms", "369.6 ms"),
        ("NMS / Postprocessing", "0.8 ms", "1.3 ms"),
        ("IPCV Layers (CLAHE, Sobel, Otsu)", "7.1 ms", "8.1 ms"),
        ("Full Pipeline (End-to-End)", "419.9 ms", "445.4 ms"),
    ]
    
    # Table header
    hdr = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN_L, ty, left_w, Emu(160000))
    set_shape_fill(hdr, DARK_BG)
    set_no_border(hdr)
    add_textbox(slide, MARGIN_L + Emu(60000), ty + Emu(15000), Emu(2400000), Emu(130000),
                "Stage", FONT_MONO, Pt(7), True, WHITE)
    add_textbox(slide, MARGIN_L + Emu(2600000), ty + Emu(15000), Emu(700000), Emu(130000),
                "Mean", FONT_MONO, Pt(7), True, WHITE, PP_ALIGN.RIGHT)
    add_textbox(slide, MARGIN_L + Emu(3400000), ty + Emu(15000), Emu(700000), Emu(130000),
                "p95", FONT_MONO, Pt(7), True, WHITE, PP_ALIGN.RIGHT)
    
    ty += Emu(160000)
    for i, (stage, mean, p95) in enumerate(perf_data):
        bg_color = CARD_BG if i % 2 == 0 else WHITE
        row = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN_L, ty, left_w, Emu(140000))
        set_shape_fill(row, bg_color)
        set_no_border(row)
        
        is_highlight = i == 2 or i == 5  # forward pass and full pipeline
        font_bold = is_highlight
        font_color = NAVY if is_highlight else MED_SLATE
        
        add_textbox(slide, MARGIN_L + Emu(60000), ty + Emu(10000), Emu(2400000), Emu(120000),
                    stage, FONT_BODY, Pt(7), font_bold, font_color)
        add_textbox(slide, MARGIN_L + Emu(2600000), ty + Emu(10000), Emu(700000), Emu(120000),
                    mean, FONT_MONO, Pt(7), font_bold, font_color, PP_ALIGN.RIGHT)
        add_textbox(slide, MARGIN_L + Emu(3400000), ty + Emu(10000), Emu(700000), Emu(120000),
                    p95, FONT_MONO, Pt(7), font_bold, font_color, PP_ALIGN.RIGHT)
        ty += Emu(140000)
    
    # Model stats
    ty += Emu(60000)
    model_stats = [
        ("2.93 FPS", "Detection"),
        ("2.38 FPS", "End-to-End"),
        ("25.5M", "Parameters"),
        ("103.7", "GFLOPs"),
    ]
    msx = MARGIN_L
    msw = Emu(1000000)
    for val, label in model_stats:
        ms = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, msx, ty, msw, Emu(320000))
        set_shape_fill(ms, CARD_BG)
        set_shape_border(ms, RGBColor(0xE2, 0xE8, 0xF0), 1)
        add_textbox(slide, msx + Emu(30000), ty + Emu(20000), msw - Emu(60000), Emu(160000),
                    val, FONT_MONO, Pt(12), True, ACCENT_ORANGE, PP_ALIGN.CENTER)
        add_textbox(slide, msx + Emu(30000), ty + Emu(190000), msw - Emu(60000), Emu(100000),
                    label, FONT_BODY, Pt(7), False, SLATE, PP_ALIGN.CENTER)
        msx += msw + Emu(50000)
    
    # ─── Right: Robustness + Detection results ───
    right_x = Emu(4700000)
    right_w = Emu(4000000)
    
    add_textbox(slide, right_x, Emu(900000), right_w, Emu(180000),
                "Label-Free Robustness Analysis", FONT_HEADING, Pt(10), True, NAVY)
    
    # Robustness table
    ry = Emu(1100000)
    
    # Header
    rhdr = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, right_x, ry, right_w, Emu(160000))
    set_shape_fill(rhdr, DARK_BG)
    set_no_border(rhdr)
    
    r_col_x = [right_x + Emu(60000), right_x + Emu(1800000), right_x + Emu(2500000), 
               right_x + Emu(3000000), right_x + Emu(3500000)]
    r_headers = ["Perturbation", "Dets", "Recall", "F1", "Class Agr."]
    for cx, hdr_text in zip(r_col_x, r_headers):
        add_textbox(slide, cx, ry + Emu(15000), Emu(600000), Emu(130000),
                    hdr_text, FONT_MONO, Pt(6.5), True, WHITE)
    
    ry += Emu(160000)
    robustness_data = [
        ("Brightness +20 DL", "9", "1.00", "1.00", "1.00"),
        ("Contrast ×0.8", "9", "1.00", "1.00", "1.00"),
        ("Gamma 0.8", "9", "1.00", "1.00", "1.00"),
        ("Horizontal flip", "9", "1.00", "1.00", "1.00"),
        ("JPEG quality 40", "8", "0.89", "0.94", "1.00"),
        ("Half-res resample", "7", "0.78", "0.88", "0.86"),
        ("Gaussian blur σ=1.2", "4", "0.44", "0.61", "0.75"),
        ("Gaussian noise σ=8", "2", "0.22", "0.36", "1.00"),
    ]
    
    for i, (pert, dets, recall, f1, cls_agr) in enumerate(robustness_data):
        bg_color = CARD_BG if i % 2 == 0 else WHITE
        row = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, right_x, ry, right_w, Emu(135000))
        set_shape_fill(row, bg_color)
        set_no_border(row)
        
        vals = [pert, dets, recall, f1, cls_agr]
        for cx, v in zip(r_col_x, vals):
            fcolor = MED_SLATE
            add_textbox(slide, cx, ry + Emu(10000), Emu(600000), Emu(115000),
                        v, FONT_MONO if cx != r_col_x[0] else FONT_BODY, Pt(6.5), False, MED_SLATE)
        ry += Emu(135000)
    
    # Mean F1
    add_textbox(slide, right_x, ry + Emu(10000), right_w, Emu(130000),
                "Mean F1 = 0.85  |  Invariant to photometric changes; sensitive to noise/blur",
                FONT_BODY, Pt(7.5), True, NAVY)
    
    # ─── Detection results from sample frame ───
    ry += Emu(180000)
    add_textbox(slide, right_x, ry, right_w, Emu(180000),
                "Sample Frame Detection Results", FONT_HEADING, Pt(10), True, NAVY)
    
    det_y = ry + Emu(200000)
    det_items = [
        ("9 defects detected", "8× Class 3 (Critical), 1× Class 2 (Medium)"),
        ("Mean confidence", "78.8% across all detections"),
        ("Fault breakdown", "4× MultiDiode, 4× MultiHotSpot, 1× SingleHotSpot"),
        ("Deterministic", "Identical boxes across all 20 benchmark iterations"),
    ]
    for label, value in det_items:
        tb = slide.shapes.add_textbox(right_x, det_y, right_w, Emu(120000))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r1 = p.add_run()
        r1.text = label + ":  "
        r1.font.name = FONT_HEADING
        r1.font.size = Pt(7.5)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r2 = p.add_run()
        r2.text = value
        r2.font.name = FONT_BODY
        r2.font.size = Pt(7.5)
        r2.font.color.rgb = MED_SLATE
        det_y += Emu(140000)
    
    add_footer_bar(slide)


def build_slide_9_contributions(prs):
    """Slide 9: Key Contributions, Limitations & Future Scope"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    
    add_slide_title(slide, "Key Contributions, Limitations & Future Scope")
    
    # Three columns
    col_w = Emu(2650000)
    col_gap = Emu(130000)
    col_y = Emu(880000)
    col_h = Emu(2800000)
    
    # Column 1: Contributions
    c1_x = MARGIN_L
    c1 = add_card(slide, c1_x, col_y, col_w, col_h, CARD_BG, RGBColor(0xE2, 0xE8, 0xF0))
    bar1 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, c1_x, col_y, col_w, Emu(30000))
    set_shape_fill(bar1, ACCENT_GREEN)
    set_no_border(bar1)
    
    add_textbox(slide, c1_x + Emu(80000), col_y + Emu(60000), col_w - Emu(160000), Emu(200000),
                "✅ Key Contributions", FONT_HEADING, Pt(11), True, ACCENT_GREEN)
    
    contribs = [
        "Multi-layered system: not just a YOLO wrapper — 6 distinct pipeline stages",
        "Classical IPCV rigor: CLAHE, Sobel, Otsu with mathematical verification",
        "IEC TS 62446-3 compliance: industry-standard severity grading and reporting",
        "Two independent ΔT measurements: class prior + image-measured contrast",
        "Sub-pixel homography: 0.48 px corner accuracy with LSQ edge refinement",
        "Quantitative economics: kW, kWh/yr, $/yr yield-loss estimation",
        "Production engineering: thread-safe inference, warm-up, input validation, Docker-ready",
        "Interactive cockpit with guided presentation tour"
    ]
    add_multiline_textbox(slide, c1_x + Emu(80000), col_y + Emu(280000), col_w - Emu(160000), Emu(2400000),
                         contribs, FONT_BODY, Pt(7.5), MED_SLATE, bullet=True,
                         spacing_before=Pt(1), spacing_after=Pt(2))
    
    # Column 2: Limitations
    c2_x = c1_x + col_w + col_gap
    c2 = add_card(slide, c2_x, col_y, col_w, col_h, CARD_BG, RGBColor(0xE2, 0xE8, 0xF0))
    bar2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, c2_x, col_y, col_w, Emu(30000))
    set_shape_fill(bar2, ACCENT_AMBER)
    set_no_border(bar2)
    
    add_textbox(slide, c2_x + Emu(80000), col_y + Emu(60000), col_w - Emu(160000), Emu(200000),
                "⚠️ Limitations", FONT_HEADING, Pt(11), True, ACCENT_AMBER)
    
    limitations = [
        "Temperatures are relative: 8-bit AGC-scaled images give approximate ΔT, not absolute calibrated values",
        "Single-frame analysis: no multi-frame de-duplication or geo-referencing",
        "Health score is per-frame triage, not a plant-capacity metric",
        "Pre-trained model: detection quality bounded by existing weights; sensitive to noise and blur",
        "No lens distortion correction: assumes planar module tables",
        "Front-end assets load from public CDNs (not fully offline)"
    ]
    add_multiline_textbox(slide, c2_x + Emu(80000), col_y + Emu(280000), col_w - Emu(160000), Emu(2400000),
                         limitations, FONT_BODY, Pt(7.5), MED_SLATE, bullet=True,
                         spacing_before=Pt(1), spacing_after=Pt(2))
    
    # Column 3: Future Work
    c3_x = c2_x + col_w + col_gap
    c3 = add_card(slide, c3_x, col_y, col_w, col_h, CARD_BG, RGBColor(0xE2, 0xE8, 0xF0))
    bar3 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, c3_x, col_y, col_w, Emu(30000))
    set_shape_fill(bar3, ACCENT_BLUE)
    set_no_border(bar3)
    
    add_textbox(slide, c3_x + Emu(80000), col_y + Emu(60000), col_w - Emu(160000), Emu(200000),
                "🔭 Future Scope", FONT_HEADING, Pt(11), True, ACCENT_BLUE)
    
    future = [
        "Radiometric R-JPEG decoding (DJI Thermal SDK) for absolute ΔT",
        "GPS/EXIF geo-tagging with orthomosaic integration and frame de-duplication",
        "Lens distortion calibration before metric rectification",
        "ONNX / TensorRT export for real-time GPU inference and batched processing",
        "Denoising / focus-quality gate based on robustness findings",
        "Unit test suite and CI/CD pipeline",
        "Bundle front-end assets for fully offline demos"
    ]
    add_multiline_textbox(slide, c3_x + Emu(80000), col_y + Emu(280000), col_w - Emu(160000), Emu(2400000),
                         future, FONT_BODY, Pt(7.5), MED_SLATE, bullet=True,
                         spacing_before=Pt(1), spacing_after=Pt(2))
    
    add_footer_bar(slide)


def build_slide_10_conclusion(prs):
    """Slide 10: Conclusion & Q&A"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    
    # Centered layout
    # Main title
    add_textbox(slide, Emu(1143000), Emu(800000), Emu(6858000), Emu(600000),
                "HELIOS-AI", FONT_HEADING, Pt(36), True, NAVY, PP_ALIGN.CENTER)
    add_textbox(slide, Emu(1143000), Emu(1350000), Emu(6858000), Emu(250000),
                "Autonomous Aerial Thermography & PV Anomaly Intelligence",
                FONT_HEADING, Pt(14), False, SLATE, PP_ALIGN.CENTER)
    
    # Summary points in a card
    summary_card = add_card(slide, Emu(1200000), Emu(1750000), Emu(6700000), Emu(1800000), CARD_BG, RGBColor(0xE2, 0xE8, 0xF0))
    
    summary = [
        "✓  Detects 9 types of PV faults from a single aerial thermal frame using YOLOv9",
        "✓  Enriches every detection with classical IPCV analysis, IEC TS 62446-3 grading, and thermal physics",
        "✓  Estimates real-world financial impact: power dissipation (kW) and annual revenue loss ($/yr)",
        "✓  Provides an interactive cockpit with guided tour, IPCV lab, and printable audit certificate",
        "✓  Complete pipeline runs in ~0.42s on CPU — production-ready with Docker deployment",
        "✓  All results are deterministic, benchmarked, and robustness-tested"
    ]
    add_multiline_textbox(slide, Emu(1400000), Emu(1850000), Emu(6300000), Emu(1600000),
                         summary, FONT_BODY, Pt(10), NAVY, bullet=False,
                         spacing_before=Pt(3), spacing_after=Pt(3))
    
    # Thank you + Q&A
    add_textbox(slide, Emu(1143000), Emu(3800000), Emu(6858000), Emu(500000),
                "THANK YOU", FONT_HEADING, Pt(30), True, NAVY, PP_ALIGN.CENTER)
    add_textbox(slide, Emu(1143000), Emu(4200000), Emu(6858000), Emu(250000),
                "Questions & Discussion", FONT_HEADING, Pt(14), False, SLATE, PP_ALIGN.CENTER)
    
    # Team info
    add_textbox(slide, Emu(585788), Emu(4550569), Emu(6072300), Emu(153900),
                "Arpit Shukla  ·  Ashish Gautam  ·  Amit Kumar Bhorayat  ·  Alok Kumar Shukla",
                FONT_BODY, Pt(8), False, SLATE, PP_ALIGN.CENTER)
    add_textbox(slide, Emu(1143000), Emu(4700000), Emu(6858000), Emu(153900),
                "Mentor: Dr. Divyansh Thakur  |  MNNIT Allahabad",
                FONT_BODY, Pt(8), False, SLATE, PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════
# MAIN: Build the presentation
# ════════════════════════════════════════════════════════════

def main():
    print("Loading template...")
    prs = Presentation(TEMPLATE)
    
    # Remove all existing slides (they are from a different project)
    while len(prs.slides) > 0:
        rId = prs.slides._sldIdLst[0].get(qn('r:id'))
        prs.part.drop_rel(rId)
        prs.slides._sldIdLst.remove(prs.slides._sldIdLst[0])
    
    print("Building Slide 1: Project Introduction...")
    build_slide_1_intro(prs)
    
    print("Building Slide 2: Problem Statement & Motivation...")
    build_slide_2_problem(prs)
    
    print("Building Slide 3: Proposed Solution...")
    build_slide_3_solution(prs)
    
    print("Building Slide 4: System Architecture...")
    build_slide_4_architecture(prs)
    
    print("Building Slide 5: Core Detection Methodology...")
    build_slide_5_detection(prs)
    
    print("Building Slide 6: Classical IPCV & Thermal Analysis...")
    build_slide_6_thermal_ipcv(prs)
    
    print("Building Slide 7: Implementation & Demonstration...")
    build_slide_7_demo(prs)
    
    print("Building Slide 8: Evaluation & Results...")
    build_slide_8_results(prs)
    
    print("Building Slide 9: Contributions, Limitations & Future...")
    build_slide_9_contributions(prs)
    
    print("Building Slide 10: Conclusion & Q&A...")
    build_slide_10_conclusion(prs)
    
    print(f"Saving to {OUTPUT}...")
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    prs.save(OUTPUT)
    print(f"Done! {len(prs.slides)} slides created.")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
