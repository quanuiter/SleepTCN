"""Build the approved vector study-design schematic; no inference or data changes."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib import font_manager

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent.parent/'figures')
parser.add_argument('--language', choices=['en', 'vi'], default='en')
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'Reports/paper_en/main.tex': '71fe0296e22a9b70658a85b46ccecdbef9229b6e12a63adbe26c64188c9db178',
    'Reports/paper_en/supplement.tex': '28c1ae98a10e490d3a0e5e8bf22692559e64ba421d65d869424eeb653414d7c9',
}
for name in ['arial.ttf', 'arialbd.ttf', 'ariali.ttf']:
    font = Path(f'C:/Windows/Fonts/{name}')
    if font.exists(): font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8,
                     'text.color': '#263643', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42,
                     'savefig.facecolor': 'white'})
INK = '#263643'
MUTED = '#61717E'
LINE = '#93A5B2'
TEAL = '#317C88'
TEAL_PALE = '#EFF7F8'
BLUE = '#466A94'
BLUE_PALE = '#F0F4F9'
ORANGE = '#B2763F'
ORANGE_PALE = '#FAF4ED'
VI = {
    'Model configurations':'Cấu hình mô hình', 'EEG input':'Tín hiệu EEG',
    'Epoch encoder':'Bộ mã hóa epoch', 'Sequence model':'Mô hình chuỗi',
    'Raw EEG':'EEG thô', 'Calibrated amplitude (µV)':'Biên độ hiệu chuẩn (µV)',
    '15-CNN encoder':'Bộ mã hóa 15-CNN', 'Current':'Hiện tại',
    'Previous':'Liền trước', 'Next':'Liền sau',
    '15 branches × 5 outputs = 75 features':'15 nhánh × 5 đầu ra = 75 đặc trưng',
    '6 dilated blocks':'6 khối giãn', 'Raw':'Thô',
    'Band-pass\nFixed scale (÷100)':'Lọc thông dải\nChia cố định (÷100)',
    'Band-pass only':'Chỉ lọc thông dải',
    'Band-pass\nRecord-wise z-score':'Lọc thông dải\nz-score theo bản ghi',
    'ResNet-1D encoder':'Bộ mã hóa ResNet-1D', 'Current epoch':'Epoch hiện tại',
    'Stem\n32':'Stem\n32',
    'Residual blocks → global average pooling':'Khối residual → lấy trung bình toàn cục',
    'Single-channel EEG  ·  100 Hz  ·  30-s epochs':'EEG đơn kênh  ·  100 Hz  ·  epoch 30 giây',
    'Sleep stages':'Giai đoạn ngủ', 'Evaluation protocol':'Quy trình đánh giá',
    '78 subjects · 153 recordings':'78 đối tượng · 153 bản ghi',
    '8 train':'8 train', '1 validation':'1 validation', '1 test':'1 test',
    '10 subject-wise folds; out-of-fold predictions':'10 fold theo đối tượng; dự đoán ngoài fold',
    'Source models':'Mô hình nguồn', '10 fold checkpoints':'10 checkpoint theo fold',
    'SHHS1 test set':'Tập kiểm tra SHHS1', '180 participants':'180 người tham gia',
    'Probability averaging':'Lấy trung bình xác suất',
}

fig, ax = plt.subplots(figsize=(7.1, 5.5))
fig.subplots_adjust(left=.016, right=.984, bottom=.015, top=.985)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

def label(x, y, text, size=8, weight='normal', color=INK, ha='left', va='center'):
    if args.language == 'vi': text = VI.get(text, text)
    return ax.text(x, y, text, fontsize=size, weight=weight, color=color, ha=ha,
                   va=va, linespacing=1.2)

def frame(x, y, w, h, fill='white', border=LINE, radius=.7, lw=.65):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                          linewidth=lw, edgecolor=border, facecolor=fill)
    ax.add_patch(patch)
    return patch

def line(points, color=LINE, lw=.8):
    ax.plot([p[0] for p in points], [p[1] for p in points], color=color, lw=lw,
            solid_capstyle='round', zorder=1)

def arrow(start, end, color=INK, lw=.9):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': lw,
                            'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': 8}, zorder=2)

def chip(x, y, text, w=4.8, color=TEAL):
    frame(x, y-1.6, w, 3.2, fill=color, border=color, radius=.5, lw=0)
    label(x+w/2, y, text, size=7.6, color='white', weight='bold', ha='center')

label(1, 97, 'a', size=11, weight='bold')
label(4, 97, 'Model configurations', size=10.5, weight='bold')
for x, text in [(1, 'EEG input'), (32, 'Epoch encoder'), (80, 'Sequence model')]:
    label(x, 91.5, text, size=8.2, color=MUTED, weight='bold')
    line([(x,89.5), (x+(25 if x==1 else 35 if x==32 else 19),89.5)], color='#D7DFE5', lw=.7)

# One shared raw encoder is reused by E1, rather than duplicating it as a new network.
frame(1, 66.5, 24.5, 20, BLUE_PALE, '#C6D4E0')
label(3, 82.7, 'Raw EEG', size=9, weight='bold')
label(3, 77.7, 'Calibrated amplitude (µV)', size=7.6, color=MUTED)
chip(3, 71.2, 'E0', color=INK)
chip(9, 71.2, 'E1', color=INK)
arrow((25.5,76.5),(32,76.5), BLUE)

frame(32, 64, 35, 23, 'white', '#9BB8C2')
label(34, 83.8, '15-CNN encoder', size=9, weight='bold', color=TEAL)
for x, context in [(34, 'Current'), (45, 'Previous'), (56, 'Next')]:
    label(x+4.5, 79.7, context, size=7.4, ha='center', color=MUTED)
    for j in range(5):
        frame(x, 76.2-j*2.3, 9, 1.8, TEAL_PALE, '#86ACB6', radius=.2, lw=.45)
        # Two convolution symbols per branch; no fictitious measured activations.
        ax.add_patch(Rectangle((x+.8,76.45-j*2.3),1.0,1.3,facecolor='#82B0B9',edgecolor='none'))
        ax.add_patch(Rectangle((x+2.3,76.45-j*2.3),1.0,1.3,facecolor='#5795A2',edgecolor='none'))
        line([(x+4.1,77.0-j*2.3),(x+7.8,77.0-j*2.3)], '#6D9BA6', .6)
label(49.5, 61.5, '15 branches × 5 outputs = 75 features', size=7.5, ha='center', color=MUTED)

# Fork of the same feature package into E0 and E1 sequence models.
line([(67,76.5),(72.5,76.5)], BLUE)
line([(72.5,69.5),(72.5,82.5)], BLUE)
arrow((72.5,82.5),(80,82.5), BLUE)
arrow((72.5,69.5),(80,69.5), BLUE)
label(69.9, 76.5, '75-d', size=7.2, color=BLUE, ha='center', va='bottom')

frame(80, 77, 19, 10.5, BLUE_PALE, '#A6BCD0')
chip(81.2, 84.7, 'E0', w=4.3, color=INK)
label(87,84.7,'BiLSTM',size=8.4,weight='bold',color=BLUE)
for y, xs in [(81.6,[83,88,93]), (78.9,[93,88,83])]:
    for x in xs:
        ax.add_patch(Circle((x,y),.65,facecolor='white',edgecolor=BLUE,lw=.6))
    arrow((xs[0]+(.7 if xs[0]<xs[-1] else -.7),y),
          (xs[-1]-(.7 if xs[0]<xs[-1] else -.7),y),BLUE,.65)

def tcn(x, y, w, h, title, code=None):
    frame(x,y,w,h,ORANGE_PALE,'#D5B89B')
    if code:
        chip(x+1.2,y+h-2.6,code,w=4.3,color=INK)
        label(x+7,y+h-2.6,title,size=8.4,weight='bold',color=ORANGE)
    else:
        label(x+1.2,y+h-2.5,title,size=8.4,weight='bold',color=ORANGE)
    for j in range(6):
        xx=x+1.3+j*2.65
        frame(xx,y+2.7,1.8,2.8,fill='#E1C19E',border='#BD956C',radius=.2,lw=.4)
        if j<5: arrow((xx+1.8,y+4.1),(xx+2.65,y+4.1),ORANGE,.5)
    label(x+w/2,y+1.3,'6 dilated blocks',size=7.2,ha='center',color=MUTED)

tcn(80,63,19,12,'TCN','E1')

# Signal variants precede separately fitted ResNet feature encoders.
frame(1,34,24.5,22.5,'white','#C6D4E0')
for y, code, description in [
    (53.3,'E2','Raw'),
    (47.7,'E3','Band-pass\nFixed scale (÷100)'),
    (42.1,'E4','Band-pass only'),
    (36.5,'E6','Band-pass\nRecord-wise z-score'),
]:
    chip(2.2,y,code,w=4.3,color=INK)
    label(7.7,y,description,size=7.15)
arrow((25.5,45.2),(32,45.2),TEAL)

frame(32,34,35,22.5,'white','#9BB8C2')
label(34,53.3,'ResNet-1D encoder',size=9,weight='bold',color=TEAL)
label(34,49.7,'Current epoch',size=7.5,color=MUTED)
xs = [34,40.2,46.4,52.6,59.3]
widths = [4.8,4.8,4.8,4.8,5.7]
for j,(xx,ww,name) in enumerate(zip(xs,widths,['Stem\n32','64','128','128','GAP'])):
    frame(xx,40,ww,5.0,TEAL_PALE,'#86ACB6',radius=.3,lw=.6)
    label(xx+ww/2,42.5,name,size=7.2,ha='center',color=TEAL)
    if j<4: arrow((xx+ww,42.5),(xs[j+1],42.5),TEAL,.6)
    if j in [1,2,3]:
        verts=[(xx-.45,42.5),(xx-.45,47.3),(xx+ww+.45,47.3),(xx+ww+.45,43.3)]
        patch=PathPatch(MPath(verts,[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4]),
                        fill=False,edgecolor=TEAL,lw=.65)
        ax.add_patch(patch)
label(49.5,36.5,'Residual blocks → global average pooling',size=7.15,ha='center',color=MUTED)
arrow((67,45.2),(80,45.2),TEAL)
label(73.5,47.7,'128-d',size=7.2,color=TEAL,ha='center')
tcn(80,39.2,19,12,'TCN')
label(89.5,36.5,'E2 / E3 / E4 / E6',size=7.5,ha='center',color=MUTED)

# Common output is specified once instead of repeating five-class boilerplate.
label(1,29.9,'Single-channel EEG  ·  100 Hz  ·  30-s epochs',size=7.8,color=MUTED)
label(68,29.9,'Sleep stages',size=7.6,color=MUTED)
for i,stage in enumerate(['W','N1','N2','N3','REM']):
    frame(80+i*3.8,28.1,3.4,3.6,'#F3F5F7','#CBD4DD',radius=.2,lw=.45)
    label(81.7+i*3.8,29.9,stage,size=7.1,ha='center')
line([(1,25.2),(99,25.2)],color='#CBD4DD',lw=.7)

# A separate evaluation lane with visible fold roles, not a wall of caveat text.
label(1,22.4,'b',size=11,weight='bold')
label(4,22.4,'Evaluation protocol',size=10.5,weight='bold')
label(1,17.8,'Sleep-EDF',size=9.3,weight='bold')
label(1,14.2,'78 subjects · 153 recordings',size=7.6,color=MUTED)
for j in range(10):
    fill = '#D8E7ED' if j<8 else '#E9D0AD' if j==8 else '#698BA7'
    frame(1+j*3.1,8.7,2.5,3.5,fill,fill,radius=.25,lw=0)
for x, color, text in [(1,'#D8E7ED','8 train'),(15.5,'#E9D0AD','1 validation'),(30,'#698BA7','1 test')]:
    ax.add_patch(Rectangle((x,5.9),.85,.85,facecolor=color,edgecolor='none'))
    label(x+1.6,6.4,text,size=7.2,color=MUTED)
label(1,2.3,'10 subject-wise folds; out-of-fold predictions',size=7.2,color=MUTED)

arrow((34,11.2),(45,11.2),BLUE,.8)
for dx,dy in [(1.5,1.5),(.75,.75),(0,0)]:
    frame(47+dx,7.8+dy,10.5,5.5,BLUE_PALE,'#A6BCD0',radius=.4,lw=.6)
    line([(49+dx,10.5+dy),(55.5+dx,10.5+dy)],BLUE,.6)
label(53,17.8,'Source models',size=9,weight='bold',ha='center')
label(53,4.6,'10 fold checkpoints',size=7.3,color=MUTED,ha='center')
arrow((61,11.2),(73,11.2),BLUE,.8)
label(74.5,17.8,'SHHS1 test set',size=9.3,weight='bold')
label(74.5,14.2,'180 participants',size=7.6,color=MUTED)
label(74.5,9.6,'Probability averaging',size=7.5)
label(74.5,5.8,'E0 / E3 / E6',size=7.7,weight='bold',color=BLUE)

# Account for actual manuscript insertion width (16.4 cm).
font_factor = 7.1*2.54/16.4
for obj in fig.findobj(matplotlib.text.Text):
    obj.set_fontsize(obj.get_fontsize()*font_factor)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
outside=[]
for obj in fig.findobj(matplotlib.text.Text):
    if not obj.get_visible() or not obj.get_text(): continue
    extent=obj.get_window_extent(renderer)
    if extent.x0 < -2 or extent.y0 < -2 or extent.x1 > fig.bbox.x1+2 or extent.y1 > fig.bbox.y1+2:
        outside.append(obj.get_text())
assert not outside, outside
basename = f'study_design_{args.language}'
for ext in ['png','svg','pdf']:
    fig.savefig(OUT/f'{basename}.{ext}',dpi=300)
plt.close(fig)
svg=ET.parse(OUT/f'{basename}.svg')
assert not svg.findall('.//{http://www.w3.org/2000/svg}image')
assert svg.findall('.//{http://www.w3.org/2000/svg}text')
manifest={
    'status':'approved_for_manuscript', 'replaces_preview':'D_study_design_overview',
    'preview_basis_sha256':SOURCES, 'width_cm_at_insertion':16.4,
    'data_changes':False, 'font':'Arial; SVG text remains editable',
    'design_changes':['Shared 15-CNN encoder and explicit E0/E1 fork',
        'Three context groups with five CNN branches each',
        'ResNet stem/three residual blocks/global pooling schematic',
        'Separate fold/ensemble evaluation lane',
        'Detailed qualifications moved to the proposed caption'],
    'checks':{'no_text_outside_canvas':True,'svg_without_raster':True},
}
(OUT/f'{basename}_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(OUT/f'{basename}.pdf')
