#!/bin/bash
set -e
cd ~/Documents/EatThisMushroom
SUB=tmlr_submission
W=/tmp/_tmlr.md
cp MANUSCRIPT_CAUTIONARY.md "$W"
# strip md title + subtitle headings (title block comes from \title)
perl -ni -e 'print unless /^# A False Floor$/ || /^### How a weak baseline and a curated benchmark/' "$W"
# citations -> pandoc syntax (slurp mode handles line-wrapped ones)
perl -0777 -i -pe '
s/\(Musgrave et\s+al\., \*A Metric Learning Reality Check\*, 2020\)/[\@musgrave2020]/g;
s/\(Dacrema et al\., 2019\)/[\@dacrema2019]/g;
s/\(Dehghani et al\., \*The Benchmark Lottery\*,\s+2021;\s*Bouthillier et al\. on benchmark variance, 2021\)/[\@dehghani2021; \@bouthillier2021]/g;
s/\(Torralba & Efros 2011; WILDS, Koh et al\. 2021; ImageNet-A, Hendrycks\s+et al\. 2021\)/[\@torralba2011; \@koh2021; \@hendrycks2021]/g;
s/\*\*McGreivy & Hakim \(2024[^)]*\)\*\*/\\citet{mcgreivy2024}/g;
s/\*\*Kuznetsov et al\. \(2026, \*npj Science of Food\*\s+10:121\)\*\*/\\citet{kuznetsov2026}/g;
s/Pfennig et al\.\s*,?\s*\*Nature\*\s+2001/[\@pfennig2001]/g;
s/Symula et al\.\s*,?\s*\*Proc\. R\. Soc\. B\*\s+2001/[\@symula2001]/g;
s/Vetter \(misidentification lit\.\)/[\@vetter2000]/g;
s/misidentification, Vetter;/misidentification, [\@vetter2000];/g;
s/\(Symula 2001; Darst & Cummings 2006\)/[\@symula2001; \@darst2006]/g;
s/\(Pfennig 2001\)/[\@pfennig2001]/g;
s/\(Vetter\)/[\@vetter2000]/g;
s/\(Symula 2001\)/[\@symula2001]/g;
s/Darst & Cummings,?\s+\*Nature\* 2006/[\@darst2006]/g;
s/Darst & Cummings 2006/[\@darst2006]/g;
s/masked-latent \*\*I-JEPA\*\* \(ViT-H\/14\)/masked-latent **I-JEPA** [\@assran2023] (ViT-H\/14)/;
s/self-distilled \*\*DINOv2\*\* \(ViT-L\/14\)/self-distilled **DINOv2** [\@oquab2024] (ViT-L\/14)/;
s/contrastive \*\*CLIP\*\*\s*\n?\(ViT-L\/14\)/contrastive **CLIP** [\@radford2021] (ViT-L\/14)/;
' "$W"
# yield notation "@80"/"@60" is not a citation -- convert so pandoc does not parse it as @key
perl -0777 -i -pe 's/\@(\d+)/ at $1/g;' "$W"
# split abstract / body (drop References section; bibtex regenerates it)
awk '/^## Abstract$/{f=1;next} /^## 1\. Introduction$/{f=0} f' "$W" > /tmp/_abs.md
awk '/^## 1\. Introduction$/{f=1} /^## References$/{f=0} f' "$W" > /tmp/_body.md
# pandoc -> latex fragments
pandoc /tmp/_abs.md  -t latex --natbib -o "$SUB/_abstract.tex"
pandoc /tmp/_body.md -t latex --natbib -o "$SUB/_body.tex"
# compile
cd "$SUB"
pdflatex -interaction=nonstopmode main.tex >/tmp/tex1.log 2>&1 || true
bibtex main               >/tmp/bib.log  2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/tmp/tex2.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/tmp/tex3.log 2>&1 || true
