from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                Spacer, PageBreak, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

PAGE = landscape(A4)

# ── OUTPUT PATH ────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    r"C:/Users/subra/Downloads/model_comparison_4DATASETS.pdf",
    pagesize=PAGE,
    leftMargin=0.8*cm, rightMargin=0.8*cm,
    topMargin=1.0*cm,  bottomMargin=1.0*cm
)

# ── STYLES ─────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
title_style = ParagraphStyle('T',  parent=styles['Title'],    fontSize=14, spaceAfter=3,
                             alignment=TA_CENTER, textColor=colors.HexColor('#0d1b2a'),
                             fontName='Helvetica-Bold')
sub_style   = ParagraphStyle('S',  parent=styles['Normal'],   fontSize=8.5, spaceAfter=2,
                             alignment=TA_CENTER, textColor=colors.HexColor('#444'))
h2_style    = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=10,
                             spaceBefore=8, spaceAfter=3,
                             textColor=colors.HexColor('#ffffff'),
                             backColor=colors.HexColor('#16213e'),
                             fontName='Helvetica-Bold',
                             leftIndent=4, rightIndent=4, borderPad=4)
body_style  = ParagraphStyle('B',  parent=styles['Normal'],   fontSize=8, leading=11, spaceAfter=3)
note_style  = ParagraphStyle('N',  parent=styles['Normal'],   fontSize=7.5, leading=10,
                             textColor=colors.HexColor('#555'))

# ── PALETTE ────────────────────────────────────────────────────────────────────
C_HDR_BG  = colors.HexColor('#0d1b2a')   # near-black navy
C_VTI     = colors.HexColor('#1a4480')   # strong blue
C_DRF     = colors.HexColor('#1a5c32')   # deep green
C_GEN     = colors.HexColor('#5c1a5c')   # deep purple
C_NRF     = colors.HexColor('#8b4000')   # burnt amber
C_ALT     = colors.HexColor('#f0f4f8')   # light blue-grey stripe
C_BORDER  = colors.HexColor('#c0c8d0')

# Row highlight colours for each dataset (lighter versions of header colours)
C_VTI_L   = colors.HexColor('#dde8f5')
C_DRF_L   = colors.HexColor('#d5eadb')
C_GEN_L   = colors.HexColor('#ead5ea')
C_NRF_L   = colors.HexColor('#f5e6cc')

story = []

# (Title Page and Legend removed as requested)

# ════════════════════════════════════════════════════════════════════════════════
# DATA  — all values as [Acc%, F1%, FPR%, FNR%]
# ════════════════════════════════════════════════════════════════════════════════
main_data = {
    'clean': {
        'VGG16':       {'vti':[91.07,91.17,5.09,10.02], 'drf':[99.44,99.44,0.27,0.67],  'gen':[99.96,99.96,0.01,0.04],  'nrf':[98.41,99.23,0.28,0.44]},
        'MobileNet':   {'vti':[90.71,90.79,5.25,10.04], 'drf':[97.50,97.44,1.34,3.28],  'gen':[99.28,99.28,0.17,0.72],  'nrf':[98.41,98.66,0.26,0.44]},
        'EfficientNet':{'vti':[90.35,90.42,5.41,10.24], 'drf':[98.33,98.34,0.84,1.79],  'gen':[100.00,100.00,0.00,0.00],'nrf':[98.41,99.23,0.28,0.44]},
        'Xception':    {'vti':[90.16,90.23,5.55,10.52], 'drf':[98.06,98.04,0.90,1.69],  'gen':None,                     'nrf':[98.41,98.66,0.26,0.44]},
        'ViT':         {'vti':[91.80,91.63,5.33,10.70], 'drf':[99.72,99.72,0.15,0.23],  'gen':[69.64,68.75,8.23,32.29], 'nrf':[95.24,94.05,0.79,5.30]},
        'Swin':        {'vti':[89.98,90.39,5.44,9.69],  'drf':[100.00,100.00,0.00,0.00],'gen':[73.72,73.32,7.27,27.65], 'nrf':[99.21,99.60,0.23,0.57]},
        'VIM':         {'vti':[92.63,92.27,3.97,9.4],   'drf':[99.1,99.4,0.4,0.7],         'gen':[99.4,99.4,0.11,0.63],    'nrf':[43.9,47.6,10.5,73.8]},
    },
    'snr20': {
        'VGG16':       {'vti':[94.77,94.78,3.42,6.73],  'drf':[98.88,98.88,0.59,1.29],  'gen':[99.99,99.99,0.00,0.02],  'nrf':[100.00,100.00,0.00,0.00]},
        'MobileNet':   {'vti':[95.47,95.44,3.10,6.14],  'drf':[94.36,94.33,3.03,6.52],  'gen':[99.34,99.34,0.16,0.66],  'nrf':[100.00,100.00,0.00,0.00]},
        'EfficientNet':{'vti':[95.44,95.38,3.36,6.55],  'drf':[97.37,97.37,1.39,3.05],  'gen':[100.00,100.00,0.00,0.00],'nrf':[97.93,98.51,0.36,0.66]},
        'Xception':    {'vti':[95.37,95.31,3.25,6.55],  'drf':[97.10,97.11,1.47,2.78],  'gen':None,                     'nrf':[99.31,99.69,0.13,0.22]},
        'ViT':         {'vti':[94.21,94.12,4.22,8.38],  'drf':[99.44,99.44,0.28,0.57],  'gen':[69.51,68.67,8.25,32.25], 'nrf':[99.31,99.69,0.13,0.22]},
        'Swin':        {'vti':[94.81,94.81,3.41,6.43],  'drf':[99.05,99.05,0.48,0.88],  'gen':[72.79,72.34,7.62,28.53], 'nrf':[97.93,97.38,0.34,1.12]},
        'VIM':         {'vti':[93.0,92.4,3.63,8.5],     'drf':[100.0,100.0,0.0,0.0],        'gen':[99.8,99.8,0.04,0.23],    'nrf':[95.6,95.7,0.7,1.1]},
    },
    'snr15': {
        'VGG16':       {'vti':[97.06,97.06,1.96,4.50],  'drf':[99.44,99.44,0.31,0.73],  'gen':[99.97,99.97,0.01,0.03],  'nrf':[98.90,99.15,0.22,0.76]},
        'MobileNet':   {'vti':[96.57,96.53,2.54,5.03],  'drf':[95.56,95.53,2.43,5.03],  'gen':[99.27,99.27,0.18,0.72],  'nrf':[98.90,98.57,0.22,0.95]},
        'EfficientNet':{'vti':[96.41,96.40,2.30,4.63],  'drf':[97.22,97.20,1.52,3.38],  'gen':[100.00,100.00,0.00,0.00],'nrf':[98.23,98.89,0.37,0.98]},
        'Xception':    {'vti':[96.90,96.86,2.40,4.59],  'drf':[97.22,97.25,1.40,2.82],  'gen':None,                     'nrf':[98.56,98.86,0.30,0.87]},
        'ViT':         {'vti':[93.46,93.35,4.92,9.41],  'drf':[98.89,98.91,0.49,0.91],  'gen':[69.62,68.80,8.21,32.04], 'nrf':[97.69,97.30,0.45,2.02]},
        'Swin':        {'vti':[95.75,95.73,2.88,5.56],  'drf':[99.72,99.72,0.16,0.23],  'gen':[72.33,71.73,7.71,29.10], 'nrf':[99.33,99.14,0.12,0.56]},
        'VIM':         {'vti':[93.7,93.2,3.2,7.5],      'drf':[100.0,100.0,0.0,0.0],        'gen':[99.4,99.4,0.1,0.59],     'nrf':[96.9,96.9,0.6,3.1]},
    },
    'snr10': {
        'VGG16':       {'vti':[94.56,94.53,3.71,7.74],  'drf':[97.49,97.49,1.20,2.37],  'gen':[99.90,99.90,0.03,0.10],  'nrf':[98.35,98.26,0.39,1.52]},
        'MobileNet':   {'vti':[94.74,94.69,3.63,7.27],  'drf':[90.90,90.85,5.19,10.06], 'gen':[98.21,98.21,0.44,1.76],  'nrf':[99.17,99.21,0.26,1.30]},
        'EfficientNet':{'vti':[94.77,94.75,3.40,7.05],  'drf':[95.76,95.73,2.20,4.57],  'gen':[99.96,99.96,0.01,0.04],  'nrf':[99.17,99.21,0.26,1.30]},
        'Xception':    {'vti':[95.12,95.06,3.49,7.23],  'drf':[95.92,95.90,2.07,4.02],  'gen':None,                     'nrf':[98.35,97.53,0.39,2.89]},
        'ViT':         {'vti':[94.49,94.40,4.13,8.21],  'drf':[97.82,97.82,1.06,1.73],  'gen':[67.86,66.92,8.57,33.74], 'nrf':[99.17,97.28,0.12,1.30]},
        'Swin':        {'vti':[94.49,94.49,3.35,6.67],  'drf':[97.88,97.87,1.08,1.88],  'gen':[70.77,70.02,8.10,30.74], 'nrf':[99.17,99.21,0.26,1.30]},
        'VIM':         {'vti':[95.1,94.6,2.37,5.8],     'drf':[100.0,100.0,0.0,0.0],        'gen':[98.7,98.7,0.22,1.29],    'nrf':[97.7,97.7,0.4,0.6]},
    },
    'snr5': {
        'VGG16':       {'vti':[96.36,96.34,2.13,4.97],  'drf':[96.94,96.73,1.66,4.81],  'gen':[98.94,98.94,0.26,1.05],  'nrf':[99.19,98.94,0.25,1.79]},
        'MobileNet':   {'vti':[96.90,96.87,2.18,4.06],  'drf':[90.83,90.47,6.80,12.32], 'gen':[92.87,92.87,1.71,7.02],  'nrf':[99.19,98.94,0.25,1.79]},
        'EfficientNet':{'vti':[96.72,96.71,2.13,3.87],  'drf':[92.78,92.53,6.45,10.86], 'gen':[99.34,99.35,0.17,0.65],  'nrf':[99.19,98.94,0.25,1.79]},
        'Xception':    {'vti':[97.27,97.24,1.96,3.89],  'drf':[95.56,95.39,2.43,5.83],  'gen':None,                     'nrf':[98.78,98.34,0.37,2.68]},
        'ViT':         {'vti':[94.72,94.59,4.02,8.27],  'drf':[96.39,96.46,1.66,3.11],  'gen':[63.15,61.99,9.54,38.44], 'nrf':[98.79,98.69,0.31,1.90]},
        'Swin':        {'vti':[95.81,95.83,2.37,5.21],  'drf':[97.50,97.50,1.28,2.58],  'gen':[66.06,65.10,9.12,35.20], 'nrf':[99.19,98.94,0.25,1.79]},
        'VIM':         {'vti':[94.1,93.5,2.9,6.9],      'drf':[96.6,97.1,1.5,3.7],          'gen':[92.2,92.2,1.31,7.48],    'nrf':[95.5,95.4,1.1,5.0]},
    },
    'snr0': {
        'VGG16':       {'vti':[92.70,92.54,5.06,10.85], 'drf':[86.54,85.17,15.86,21.62],'gen':[92.88,92.88,1.72,7.02],  'nrf':[99.27,98.64,0.11,0.51]},
        'MobileNet':   {'vti':[91.44,91.35,4.99,10.82], 'drf':[83.69,82.36,16.79,24.22],'gen':[70.19,70.06,7.14,29.44], 'nrf':[100.00,100.00,0.00,0.00]},
        'EfficientNet':{'vti':[92.39,92.27,4.84,10.27], 'drf':[84.48,82.99,17.23,23.72],'gen':[95.05,95.05,1.22,4.91],  'nrf':[100.00,100.00,0.00,0.00]},
        'Xception':    {'vti':[93.68,93.59,4.26,8.90],  'drf':[86.21,84.74,15.83,22.12],'gen':None,                     'nrf':[99.27,99.64,0.22,0.51]},
        'ViT':         {'vti':[92.91,92.78,4.68,10.30], 'drf':[89.56,89.13,11.32,15.68],'gen':[54.10,53.38,11.52,47.03],'nrf':[98.54,98.64,0.24,0.71]},
        'Swin':        {'vti':[93.54,93.50,3.98,8.25],  'drf':[90.06,89.44,11.87,15.66],'gen':[53.86,53.29,11.67,47.33],'nrf':[100.00,100.00,0.00,0.00]},
        'VIM':         {'vti':[95.1,94.7,2.4,5.7],      'drf':[90.1,92.7,4.1,7.2],          'gen':[66.6,66.6,5.62,32.01],   'nrf':[96.6,96.7,0.6,3.0]},
    },
}

MODEL_ORDER = ['VGG16', 'MobileNet', 'EfficientNet', 'Xception', 'ViT', 'Swin', 'VIM']

# Noisy RF negative SNR data — [Acc, F1%, FPR%, FNR%]
nrf_neg = {
    '-20': {'VGG16':[67.42,53.20,8.37,49.59], 'MobileNet':[61.36,49.37,9.41,50.96], 'EfficientNet':[65.91,52.26,8.60,50.40],
            'Xception':[66.67,52.29,8.36,49.82], 'ViT':[64.39,56.04,8.70,45.60], 'Swin':[70.45,62.61,7.54,38.79],
            'VIM':[56.6,48.9,9.8,54.8]},
    '-18': {'VGG16':[75.00,69.68,6.60,34.08], 'MobileNet':[69.70,61.11,7.78,44.12], 'EfficientNet':[69.70,63.28,7.82,42.37],
            'Xception':[75.76,74.89,6.30,30.29], 'ViT':[68.94,61.12,7.94,41.21], 'Swin':[76.52,76.25,6.03,27.90],
            'VIM':[69.7,65.4,7.6,48.6]},
    '-16': {'VGG16':[88.28,77.74,3.25,23.68], 'MobileNet':[84.14,74.70,4.21,26.87], 'EfficientNet':[82.76,70.60,4.36,30.68],
            'Xception':[88.97,77.77,2.95,24.08], 'ViT':[84.14,74.55,4.36,27.11], 'Swin':[91.72,81.96,2.24,18.81],
            'VIM':[75.9,74.0,5.5,37.3]},
    '-14': {'VGG16':[95.35,93.71,1.18,8.62],  'MobileNet':[93.02,89.18,1.87,14.51], 'EfficientNet':[94.57,91.29,1.30,10.93],
            'Xception':[92.25,90.92,1.99,11.70], 'ViT':[90.70,89.37,2.36,13.95], 'Swin':[93.02,92.26,1.67,9.60],
            'VIM':[83.6,83.1,3.9,25.7]},
    '-12': {'VGG16':[99.31,99.51,0.24,0.79],  'MobileNet':[94.48,94.00,1.27,6.35],  'EfficientNet':[97.93,97.91,0.58,2.39],
            'Xception':[97.93,97.57,0.46,3.00], 'ViT':[94.48,94.01,1.26,5.73], 'Swin':[96.55,95.11,0.81,6.47],
            'VIM':[84.2,84.2,3.3,20.2]},
    '-10': {'VGG16':[99.37,99.29,0.10,0.18],  'MobileNet':[98.10,97.01,0.30,2.67],  'EfficientNet':[98.73,98.52,0.28,1.48],
            'Xception':[98.73,98.17,0.28,2.73], 'ViT':[97.47,96.92,0.49,3.22], 'Swin':[96.84,97.46,0.51,0.92],
            'VIM':[87.7,86.8,2.7,19.9]},
    '-8':  {'VGG16':[100.00,100.00,0.00,0.00],'MobileNet':[98.60,99.30,0.42,0.99],  'EfficientNet':[99.30,98.65,0.10,0.49],
            'Xception':[99.30,99.65,0.21,0.49], 'ViT':[99.30,99.65,0.21,0.49], 'Swin':[100.00,100.00,0.00,0.00],
            'VIM':[90.1,90.0,2.4,10.3]},
    '-6':  {'VGG16':[100.00,100.00,0.00,0.00],'MobileNet':[97.69,97.58,0.53,3.52],  'EfficientNet':[99.23,99.58,0.20,0.57],
            'Xception':[99.23,99.58,0.20,0.57], 'ViT':[99.23,99.58,0.20,0.57], 'Swin':[100.00,100.00,0.00,0.00],
            'VIM':[94.9,94.8,1.2,5.2]},
    '-4':  {'VGG16':[100.00,100.00,0.00,0.00],'MobileNet':[99.24,99.64,0.14,0.23],  'EfficientNet':[100.00,100.00,0.00,0.00],
            'Xception':[98.48,98.27,0.32,1.81], 'ViT':[97.73,96.88,0.39,2.49], 'Swin':[99.24,99.64,0.14,0.23],
            'VIM':[95.0,95.0,1.0,5.5]},
    '-2':  {'VGG16':[100.00,100.00,0.00,0.00],'MobileNet':[100.00,100.00,0.00,0.00],'EfficientNet':[100.00,100.00,0.00,0.00],
            'Xception':[100.00,100.00,0.00,0.00], 'ViT':[98.53,98.29,0.24,1.40], 'Swin':[100.00,100.00,0.00,0.00],
            'VIM':[94.0,94.1,1.4,4.6]},
}

# VTI Dataset negative SNR — Vision Mamba (VIM) only — [Acc%, F1%, FPR%, FNR%]
# Other models were not evaluated at negative SNR for VTI; shown as '-'
vti_neg = {
    '-15': {'VGG16':None, 'MobileNet':None, 'EfficientNet':None, 'Xception':None, 'ViT':None, 'Swin':None, 'VIM':[79.3, 77.9, 7.4, 22.1]},
    '-10': {'VGG16':None, 'MobileNet':None, 'EfficientNet':None, 'Xception':None, 'ViT':None, 'Swin':None, 'VIM':[87.4, 86.2, 4.6, 13.9]},
    '-5':  {'VGG16':None, 'MobileNet':None, 'EfficientNet':None, 'Xception':None, 'ViT':None, 'Swin':None, 'VIM':[93.2, 92.4, 3.0,  8.1]},
}

# ── HELPERS ────────────────────────────────────────────────────────────────────
def fmt(v):
    return '-' if v is None else f"{v:.2f}"

def row4(x):
    if x is None:
        return ['-', '-', '-', '-']
    return [fmt(x[0]), fmt(x[1]), fmt(x[2]), fmt(x[3])]

# ── LAYOUT CONSTANTS ───────────────────────────────────────────────────────────
# Usable width = 29.7 cm − 2×0.8 cm = 28.1 cm
# Main table: 1 name col + 16 metric cols
NAME_W = 2.8*cm
MET_W  = 1.52*cm   # 2.8 + 16×1.52 = 27.12 cm  ✓ fits

# ── MAIN SNR TABLE BUILDER ─────────────────────────────────────────────────────
def make_snr_table(snr_key):
    h0 = ['',
           'VTI Dataset',        '', '', '',
           'DroneRF Dataset',    '', '', '',
           'Genesys Spectrogram', '', '', '',
           'Noisy RF Dataset',    '', '', '']
    h1 = ['Model',
           'Acc%','F1%','FPR%','FNR%',
           'Acc%','F1%','FPR%','FNR%',
           'Acc%','F1%','FPR%','FNR%',
           'Acc%','F1%','FPR%','FNR%']
    rows = [h0, h1]
    for m in MODEL_ORDER:
        d = main_data[snr_key][m]
        rows.append([m] + row4(d['vti']) + row4(d['drf']) + row4(d['gen']) + row4(d['nrf']))

    cw = [NAME_W] + [MET_W] * 16
    t  = Table(rows, colWidths=cw, repeatRows=2)

    # Base alternating + border styles
    ts_cmds = [
        # ── dataset header spans
        ('SPAN', (1,0),  (4,0)),
        ('SPAN', (5,0),  (8,0)),
        ('SPAN', (9,0),  (12,0)),
        ('SPAN', (13,0), (16,0)),
        # ── dataset header bg colours
        ('BACKGROUND', (1,0),  (4,0),  C_VTI),
        ('BACKGROUND', (5,0),  (8,0),  C_DRF),
        ('BACKGROUND', (9,0),  (12,0), C_GEN),
        ('BACKGROUND', (13,0), (16,0), C_NRF),
        # ── metric sub-header row
        ('BACKGROUND', (0,1), (-1,1), C_HDR_BG),
        ('TEXTCOLOR',  (0,0), (-1,1), colors.white),
        # ── fonts
        ('FONTNAME',  (0,0), (-1,1),  'Helvetica-Bold'),
        ('FONTNAME',  (0,2), (-1,-1), 'Helvetica'),
        ('FONTSIZE',  (0,0), (-1,-1), 7.2),
        # ── alignment
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',       (0,2), (0,-1),  'LEFT'),
        ('LEFTPADDING', (0,2), (0,-1),  5),
        # ── model name column bold
        ('FONTNAME', (0,2), (0,-1), 'Helvetica-Bold'),
        # ── alternating row shading (data rows only)
        *[('BACKGROUND', (0,r), (-1,r), C_ALT) for r in range(2, 2+len(MODEL_ORDER), 2)],
        # ── grid
        ('GRID',       (0,0), (-1,-1), 0.3, C_BORDER),
        ('LINEBELOW',  (0,1), (-1,1),  1.0, colors.white),
        ('TOPPADDING',    (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        # ── vertical dividers between dataset blocks
        ('LINEAFTER', (4,0),  (4,-1),  1.4, colors.white),
        ('LINEAFTER', (8,0),  (8,-1),  1.4, colors.white),
        ('LINEAFTER', (12,0), (12,-1), 1.4, colors.white),
        # ── outer border
        ('BOX', (0,0), (-1,-1), 1.2, C_HDR_BG),
    ]

    t.setStyle(TableStyle(ts_cmds))
    return t

# ── POSITIVE / ZERO SNR SECTIONS ───────────────────────────────────────────────
snr_sections = [
    ('clean', 'SNR  CLEAN'),
    ('snr20', 'SNR  +20 dB'),
    ('snr15', 'SNR  +15 dB'),
    ('snr10', 'SNR  +10 dB'),
    ('snr5',  'SNR  +5 dB'),
    ('snr0',  'SNR  0 dB'),
]

for key, label in snr_sections:
    story.append(Paragraph(label, h2_style))
    story.append(Spacer(1, 1*mm))
    story.append(make_snr_table(key))
    story.append(PageBreak()) # FORCE BREAK AFTER EACH TABLE

# ════════════════════════════════════════════════════════════════════════════════
# NEGATIVE SNR  —  Noisy RF Dataset ONLY  (−20 dB to −2 dB in 2 dB steps)
# ════════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("Negative SNR Levels — Noisy RF Dataset Only", h2_style))
story.append(Spacer(1, 3*mm))

neg_snrs = ['-20', '-18', '-16', '-14', '-12', '-10', '-8', '-6', '-4', '-2']

# FIX: usable width = 28.1 cm.  Split into two chunks of 5 SNR levels.
# Each chunk: 1 name col (3.0 cm) + 5×4 metric cols.
# Available for metrics = 28.1 − 3.0 = 25.1 cm → per metric col = 25.1 / 20 ≈ 1.255 cm
NEG_NAME_W = 3.0*cm
NEG_MET_W  = 1.25*cm   # 3.0 + 20×1.25 = 28.0 cm  ✓ fits

# Alternating amber shades for negative SNR columns
NEG_COLORS = [colors.HexColor('#5a2d00'), colors.HexColor('#7a4200')]

for chunk_start in range(0, len(neg_snrs), 5):
    chunk = neg_snrs[chunk_start:chunk_start + 5]
    n_snr = len(chunk)

    h0 = ['']
    h1 = ['Model']
    for s in chunk:
        h0 += [f'SNR {s} dB', '', '', '']
        h1 += ['Acc%', 'F1%', 'FPR%', 'FNR%']

    rows = [h0, h1]
    for m in MODEL_ORDER:
        row = [m]
        for s in chunk:
            row += row4(nrf_neg[s][m])
        rows.append(row)

    cw = [NEG_NAME_W] + [NEG_MET_W] * (4 * n_snr)
    t  = Table(rows, colWidths=cw, repeatRows=2)

    ts = TableStyle([
        # SNR-level header spans
        *[('SPAN', (1 + i*4, 0), (4 + i*4, 0)) for i in range(n_snr)],
        # alternating dark amber shades per SNR block
        *[('BACKGROUND', (1 + i*4, 0), (4 + i*4, 0), NEG_COLORS[i % 2]) for i in range(n_snr)],
        # metric header row
        ('BACKGROUND', (0,1), (-1,1), C_HDR_BG),
        ('TEXTCOLOR',  (0,0), (-1,1), colors.white),
        ('FONTNAME',   (0,0), (-1,1), 'Helvetica-Bold'),
        ('FONTNAME',   (0,2), (-1,-1),'Helvetica'),
        ('FONTNAME',   (0,2), (0,-1), 'Helvetica-Bold'),   # model name bold
        ('FONTSIZE',   (0,0), (-1,-1), 7.5),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',       (0,2), (0,-1), 'LEFT'),
        ('LEFTPADDING', (0,2), (0,-1), 5),
        # alternating row shading
        *[('BACKGROUND', (0,r), (-1,r), C_ALT) for r in range(2, 2+len(MODEL_ORDER), 2)],
        # grid & dividers
        ('GRID',    (0,0), (-1,-1), 0.3, C_BORDER),
        ('BOX',     (0,0), (-1,-1), 1.2, C_HDR_BG),
        ('LINEBELOW', (0,1), (-1,1), 1.0, colors.white),
        *[('LINEAFTER', (4 + i*4, 0), (4 + i*4, -1), 1.2, colors.white) for i in range(n_snr - 1)],
        ('TOPPADDING',    (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ])
    t.setStyle(ts)
    story.append(t)
    story.append(PageBreak()) # FORCE BREAK AFTER EACH CHUNK

# ════════════════════════════════════════════════════════════════════════════════
# NEGATIVE SNR  —  VTI Dataset  (−15 dB, −10 dB, −5 dB  |  Vision Mamba only)
# ════════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("Negative SNR Levels — VTI Dataset (Vision Mamba Only)", h2_style))
story.append(Spacer(1, 3*mm))

vti_neg_snrs = ['-15', '-10', '-5']

# All 3 levels fit in one chunk: 1 name col (3.0 cm) + 3×4 metric cols
VTI_NEG_NAME_W = 3.0*cm
VTI_NEG_MET_W  = 1.55*cm   # 3.0 + 12×1.55 = 21.6 cm  ✓ fits comfortably

# Alternating blue shades for VTI negative SNR column headers
VTI_NEG_COLORS = [colors.HexColor('#0d3b6e'), colors.HexColor('#1a5499')]

n_vti_neg = len(vti_neg_snrs)
h0_vti = ['']
h1_vti = ['Model']
for s in vti_neg_snrs:
    h0_vti += [f'SNR {s} dB', '', '', '']
    h1_vti += ['Acc%', 'F1%', 'FPR%', 'FNR%']

rows_vti = [h0_vti, h1_vti]
for m in MODEL_ORDER:
    row = [m]
    for s in vti_neg_snrs:
        row += row4(vti_neg[s][m])
    rows_vti.append(row)

cw_vti = [VTI_NEG_NAME_W] + [VTI_NEG_MET_W] * (4 * n_vti_neg)
t_vti  = Table(rows_vti, colWidths=cw_vti, repeatRows=2)

ts_vti = TableStyle([
    # SNR-level header spans
    *[('SPAN', (1 + i*4, 0), (4 + i*4, 0)) for i in range(n_vti_neg)],
    # alternating blue shades per SNR block
    *[('BACKGROUND', (1 + i*4, 0), (4 + i*4, 0), VTI_NEG_COLORS[i % 2]) for i in range(n_vti_neg)],
    # metric header row
    ('BACKGROUND', (0,1), (-1,1), C_HDR_BG),
    ('TEXTCOLOR',  (0,0), (-1,1), colors.white),
    ('FONTNAME',   (0,0), (-1,1), 'Helvetica-Bold'),
    ('FONTNAME',   (0,2), (-1,-1), 'Helvetica'),
    ('FONTNAME',   (0,2), (0,-1),  'Helvetica-Bold'),   # model name bold
    ('FONTSIZE',   (0,0), (-1,-1), 7.5),
    ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
    ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
    ('ALIGN',       (0,2), (0,-1), 'LEFT'),
    ('LEFTPADDING', (0,2), (0,-1), 5),
    # alternating row shading
    *[('BACKGROUND', (0,r), (-1,r), C_ALT) for r in range(2, 2+len(MODEL_ORDER), 2)],
    # grid & dividers between SNR blocks
    ('GRID',    (0,0), (-1,-1), 0.3, C_BORDER),
    ('BOX',     (0,0), (-1,-1), 1.2, C_HDR_BG),
    ('LINEBELOW', (0,1), (-1,1), 1.0, colors.white),
    *[('LINEAFTER', (4 + i*4, 0), (4 + i*4, -1), 1.2, colors.white) for i in range(n_vti_neg - 1)],
    ('TOPPADDING',    (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
])
t_vti.setStyle(ts_vti)
story.append(t_vti)
story.append(PageBreak())

# ── BUILD ──────────────────────────────────────────────────────────────────────
doc.build(story)
print("Done — PDF written successfully.")