
all: report.pdf

report.pdf: report.tex references.bib
	pdflatex report.tex
	biber report
	pdflatex report.tex
	pdflatex report.tex

clean:
	rm -f *.aux *.blg *.bbl *.toc *.log *.lof *.lot *.log.xml *.bcf *.out *.run.xml
