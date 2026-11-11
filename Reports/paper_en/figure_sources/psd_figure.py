"""Write PGFPlots sources for the cohort power-spectrum figure (English and Vietnamese).

Values are read without recomputation from runs/cohort_psd/psd_summary.json, written by
scripts/analyze_cohort_psd.py (post-hoc descriptive; median and interquartile range over recordings).

Run from this folder: python psd_figure.py
Then compile psd_en.tex with pdflatex and psd_vi.tex with xelatex and keep the PDFs beside the papers.
"""
import json
from pathlib import Path

from results_figures import COLORS, TEXT, VI_OUT, header

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parents[2] / 'runs' / 'cohort_psd' / 'psd_summary.json'
XMIN = 0.1  # the 0.05 Hz bin adjoins the removed segment mean and is not shown
FLOOR = -32.0  # values below the plotted range are clamped so that the curves leave the axis cleanly

LABELS = {
    'en': dict(raw='(a) Unfiltered input', band='(b) Band-pass input', x='Frequency (Hz)',
               y='Power spectral density (dB re 1\\,$\\mu$V$^2$/Hz)',
               edf='Sleep-EDF, Fpz-Cz ($n=153$)', shhs='SHHS1 holdout, C4-A1 ($n=500$)'),
    'vi': dict(raw='(a) Đầu vào không lọc', band='(b) Đầu vào lọc thông dải', x='Tần số (Hz)',
               y='Mật độ phổ công suất (dB so với 1\\,$\\mu$V$^2$/Hz)',
               edf='Sleep-EDF, Fpz-Cz ($n=153$)', shhs='Mẫu SHHS1 giữ lại, C4-A1 ($n=500$)'),
}


def points(frequencies, values):
    return ' '.join(f'({f},{max(v, FLOOR):.2f})' for f, v in zip(frequencies, values))


def cohort(summary, color, name, legend):
    keep = [i for i, value in enumerate(summary['frequency_hz']) if value >= XMIN]
    summary = {k: [v[i] for i in keep] for k, v in summary.items() if k.startswith(('frequency', 'psd'))}
    f = summary['frequency_hz']
    band = points(f, summary['psd_db_q75']) + ' ' + points(f[::-1], summary['psd_db_q25'][::-1])
    lines = [f'\\addplot[draw=none,fill={color},fill opacity=0.18,forget plot] coordinates {{{band}}} -- cycle;',
             f'\\addplot[{color},line width=1.2pt,forget plot] coordinates {{{points(f, summary["psd_db_median"])}}};']
    if legend:
        lines += [f'\\addlegendimage{{{color},line width=1.2pt}}', f'\\addlegendentry{{{name}}}']
    return lines


def figure(lang, cohorts):
    s = LABELS[lang]
    common = ('width=7.6cm,height=5.8cm,xmode=log,xmin=0.1,xmax=50,ymin=-30,ymax=32,'
              'xtick={0.1,0.5,1,5,10,30},xticklabels={0.1,0.5,1,5,10,30},'
              'grid=major,major grid style={gridc},axis line style={muted},'
              'tick label style={font=\\footnotesize},label style={font=\\small},'
              'title style={font=\\small\\bfseries},'
              'legend style={font=\\footnotesize,draw=none,fill=white,fill opacity=0.85,text opacity=1},'
              'legend cell align=left')
    passband = ['\\draw[muted,dashed] (axis cs:0.5,-30) -- (axis cs:0.5,32);',
                '\\draw[muted,dashed] (axis cs:30,-30) -- (axis cs:30,32);']
    out = [header(lang, 'Median power spectra (interquartile range shaded) of both cohorts.')
           .replace('transcribed from locked tables', 'read from runs/cohort_psd/psd_summary.json')
           .replace('results_figures.py', 'psd_figure.py'),
           '\\begin{tikzpicture}',
           '\\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.1cm,yticklabels at=edge left},'
           + common + f',xlabel={{{s["x"]}}}]']
    for index, (variant, title) in enumerate((('paper_raw_v1', s['raw']), ('bandpass_v2', s['band']))):
        options = f'title={{{title}}}'
        if index == 0:
            options += f',ylabel={{{s["y"]}}},legend style={{at={{(0.03,0.04)}},anchor=south west}}'
        out.append(f'\\nextgroupplot[{options}]')
        out += cohort(cohorts['sleep_edf'][variant], 'sone', s['edf'], index == 0)
        out += cohort(cohorts['shhs1_holdout'][variant], 'stwo', s['shhs'], index == 0)
        out += passband
    out += ['\\end{groupplot}', '\\end{tikzpicture}', '\\end{document}', '']
    return '\n'.join(out)


def main():
    cohorts = json.loads(SUMMARY.read_text(encoding='utf8'))['cohorts']
    assert COLORS and TEXT
    (HERE / 'psd_en.tex').write_text(figure('en', cohorts), encoding='utf8')
    (VI_OUT / 'psd_vi.tex').write_text(figure('vi', cohorts), encoding='utf8')


if __name__ == '__main__':
    main()
