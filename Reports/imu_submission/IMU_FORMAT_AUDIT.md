# IMU format and completeness audit

Checked on 9 September 2026. This audit distinguishes verified package checks from requirements that
must still be confirmed in the live Editorial Manager portal because the official IMU Guide for Authors
returned HTTP 403 in the preparation environment.

## Verified in the package

| Item | Status | Evidence or action |
|---|---|---|
| Article type | Ready | Cover letter identifies the submission as Original Research, an article type listed for IMU. |
| Title page | Present | Title, author names, affiliation and corresponding-author e-mail are in `main.tex`; no unverified ORCID is asserted. |
| Structured abstract | Present | Background, Methods, Results and Conclusions; approximately 235 words. |
| Keywords | Present | Six keywords follow the abstract. |
| Main text | Present | Introduction, Related work, Methods, Results, Discussion and Conclusion are complete. |
| Declarations | Present | Data availability, ethics, funding, acknowledgements, CRediT, competing interests and AI disclosure are included. |
| Named acknowledgement | Restored | Nguyễn Hồ Duy Trí is thanked for academic supervision, topic guidance and constructive comments. |
| SHHS/NSRR acknowledgement | Corrected | The provider-requested SHHS cooperative agreements, institutions and NSRR support identifiers are included. |
| References | Present | Numbered citations and bibliography are generated; directly relevant IMU articles are cited. |
| Main figures | Curated | Two central figures remain in the manuscript. Figure 2 is vector artwork and contains only locked primary seed-42 E0/E3/E6 results, with the two aggregate metrics identified explicitly. |
| Highlights | Ready | Four separate bullets, each no more than 85 characters excluding its bullet marker; TXT and DOCX versions are provided. |
| Float placement | Corrected | Tables 1, 2, 5, 6, 7 and 8 are locally anchored so no table interrupts a sentence or subsection narrative. |
| Supplement | Separate | The supplementary material is supplied as its own 6-page PDF and TeX source; supporting silhouette/context analyses remain here and the former near-blank final page has been removed. |
| PDF metadata | Present | Both manuscript and supplement PDFs contain title and author metadata; the supplement also contains a descriptive subject field. |
| Cover letter | Separate | One-page cover letter is supplied as PDF and TeX source. |

## Confirm in the live submission portal

- Whether IMU currently requires continuous line numbers or an anonymous review manuscript.
- Whether a graphical abstract is required, optional or not requested for this article type.
- Whether the portal asks for highlights at initial submission; upload `highlights.docx` if requested.
- The journal's current limits for abstract length and keyword count.
- The complete institutional postal address requested in the author metadata fields.
- Nguyễn Hồ Duy Trí's consent to being named in the Acknowledgements.
- The final public Git tag or commit that should replace the fixed experimental revision after editorial
  source changes are committed.

## Files to upload

1. `SleepTCN_IMU_Manuscript.pdf`
2. `SleepTCN_IMU_Supplement.pdf`
3. `highlights.docx` if requested
4. `SleepTCN_IMU_Cover_Letter.pdf`
5. `SleepTCN_IMU_Source.zip` if source files are requested
