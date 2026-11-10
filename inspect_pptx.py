from pptx import Presentation
from pptx.util import Inches, Pt, Emu

prs = Presentation('team15.pptx')

w = prs.slide_width
h = prs.slide_height
print(f'Slide width: {w} EMU = {w/914400:.2f} in')
print(f'Slide height: {h} EMU = {h/914400:.2f} in')
print(f'Aspect ratio: {w/h:.4f} (16:9 = {16/9:.4f})')
print()

for i, master in enumerate(prs.slide_masters):
    print(f'Slide Master {i}')
    for j, layout in enumerate(master.slide_layouts):
        print(f'  Layout {j}: {layout.name}')
        for ph in layout.placeholders:
            print(f'    PH idx={ph.placeholder_format.idx}, type={ph.placeholder_format.type}, name={ph.name}')
    print()

print(f'Total slides: {len(prs.slides)}')
for idx, slide in enumerate(prs.slides):
    print(f'\nSlide {idx+1}: layout="{slide.slide_layout.name}"')
    for shape in slide.shapes:
        print(f'  Shape: type={shape.shape_type}, name="{shape.name}"')
        print(f'    pos=({shape.left}, {shape.top}), size=({shape.width}, {shape.height})')
        if shape.has_text_frame:
            for para in shape.text_frame.paragraphs:
                text = para.text[:120] if para.text else ''
                font_info = ''
                if para.runs:
                    r = para.runs[0]
                    font_info = f' font={r.font.name}, size={r.font.size}, bold={r.font.bold}'
                    try:
                        if r.font.color and r.font.color.type:
                            font_info += f', color={r.font.color.rgb}'
                    except:
                        pass
                if text:
                    print(f'    Text: "{text}"{font_info}')
        if shape.shape_type == 13:
            print(f'    Image: {shape.image.content_type}')

