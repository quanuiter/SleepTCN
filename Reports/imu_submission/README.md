# IMU submission bundle

This folder is a journal-specific derivative of `Reports/paper_en`. The scientific methods, result
values and statistical outputs are unchanged. The IMU adaptation is limited to:

- a light medical-informatics and cross-cohort-reliability framing in the abstract, introduction and
  discussion;
- three directly relevant articles previously published in *Informatics in Medicine Unlocked*;
- a separate highlights file and one-page cover-letter source;
- presentation corrections that distinguish primary, secondary and post-hoc evidence, keep the
  supporting silhouette/context analyses in the supplement, and show only locked primary seed-42
  configurations in the main speed--performance figure.

The current rendered package contains a 13-page manuscript and a 6-page supplement. The vector source
for Figure 2 is retained in `figure_sources/gate8_primary_speedup_f1_en.tex`.

## Build

Build the supplement before the manuscript because `main.tex` imports supplement labels:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error cover_letter.tex
```

To regenerate Figure 2, run `pdflatex` from `figure_sources` with job name
`gate8_primary_speedup_f1_en` and output directory `..\figures` before building the manuscript.

## Submission notes

- Select **Original Research** in the submission system.
- Upload the supplement as a separate file.
- The four highlights are within Elsevier's general maximum of 85 characters per bullet (excluding the
  bullet marker) and avoid model acronyms. Elsevier's general guidance requests a separate Word file;
  use `highlights.docx` if the IMU portal requests highlights.
- The official IMU Guide for Authors returned HTTP 403 during preparation on 9 September 2026. Check the
  live portal for current abstract, keyword, file-format, review-anonymity and line-number requirements.
- Check in the live portal whether a graphical abstract is required or optional; none is asserted here.
- Confirm that Nguyễn Hồ Duy Trí has consented to being named in the Acknowledgements before uploading
  the final manuscript. His contribution is acknowledged as academic supervision, topic guidance and
  constructive comments; he is not listed as an author or assigned a CRediT role.
- Keep the SHHS/NSRR acknowledgement wording in the manuscript. It includes both the SHHS cooperative
  agreements and the NSRR support identifiers required by the dataset provider.
- Confirm the cover letter's originality/no-concurrent-submission statements remain true at upload time.
- Replace the manuscript's fixed experimental commit with the final public revision/tag after the IMU
  editorial changes are pushed.
