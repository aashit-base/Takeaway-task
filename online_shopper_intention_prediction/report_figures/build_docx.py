"""Turn report_summary.md into report_summary.docx. Needs python-docx (pandoc was not available).

Run from the online_shopper_intention_prediction folder:
    python report_figures/build_docx.py
"""
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

PAGE_WIDTH = 6.7  # usable width in inches
DARK = RGBColor(0x10, 0x42, 0x81)


def shade(cell, colour):
    """Fill a table cell with a background colour."""
    fill = OxmlElement("w:shd")
    fill.set(qn("w:val"), "clear")
    fill.set(qn("w:color"), "auto")
    fill.set(qn("w:fill"), colour)
    cell._tc.get_or_add_tcPr().append(fill)


def write_runs(paragraph, text, size=None, italic=False):
    """Write text into a paragraph. Double stars make bold, used only inside tables."""
    for part in re.split(r"(\*\*.+?\*\*)", text):
        if not part:
            continue
        bold = part.startswith("**")
        run = paragraph.add_run(part.strip("*") if bold else part)
        run.bold = bold
        run.italic = italic
        if size:
            run.font.size = Pt(size)


def add_table(doc, lines):
    """Add a markdown table. The first column is wider, and the rest share the remaining width."""
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    rows = [rows[0]] + rows[2:]
    n_cols = len(rows[0])
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    first = 2.7 if n_cols == 5 else 1.2 if n_cols > 7 else 1.0
    other = (PAGE_WIDTH - first) / (n_cols - 1)
    size = 8 if n_cols > 7 else 9
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            cell.width = Inches(first if j == 0 else other)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.keep_with_next = True  # keeps the table and its note together
            if j > 0:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            write_runs(paragraph, f"**{text}**" if i == 0 else text, size=size)
            if i == 0:
                shade(cell, "DCE6F2")
    # Repeat the header row if a table crosses a page, and do not split rows.
    header = table.rows[0]._tr.get_or_add_trPr()
    header.append(OxmlElement("w:tblHeader"))
    for row in table.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


def build(source="report_summary.md", target="report_summary.docx"):
    doc = Document()
    section = doc.sections[0]
    section.left_margin = section.right_margin = Inches(0.9)
    section.top_margin = section.bottom_margin = Inches(0.7)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(4)
    for name, size in [("Heading 1", 13), ("Heading 2", 11)]:
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = DARK
        style.paragraph_format.space_before = Pt(7)
        style.paragraph_format.space_after = Pt(3)

    lines = open(source, encoding="utf-8").read().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, block)
        elif line.startswith("# "):
            title = doc.add_paragraph()
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = title.add_run(line[2:])
            run.bold = True
            run.font.size = Pt(16)
            run.font.color.rgb = DARK
            i += 1
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=1)
            i += 1
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=2)
            i += 1
        elif line.startswith("!["):
            caption, path = re.match(r"!\[(.*)\]\((.*)\)", line).groups()
            width = {"model_comparison": 5.2, "ranking_heatmap": 3.1, "lstm_accuracy": 3.3, "threshold_cost": 3.3}[re.search(r"(\w+)\.png", path).group(1)]
            picture = doc.add_paragraph()
            picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
            picture.paragraph_format.keep_with_next = True
            picture.paragraph_format.space_after = Pt(2)
            picture.add_run().add_picture(path, width=Inches(width))
            text = doc.add_paragraph()
            text.alignment = WD_ALIGN_PARAGRAPH.CENTER
            write_runs(text, caption, size=9, italic=True)
            i += 1
        elif line.startswith("- "):
            item = doc.add_paragraph(style="List Bullet")
            item.paragraph_format.space_after = Pt(2)
            write_runs(item, line[2:])
            i += 1
        elif re.match(r"Table \d+\. ", line):  # caption above a table
            caption = doc.add_paragraph()
            caption.paragraph_format.keep_with_next = True
            caption.paragraph_format.space_before = Pt(5)
            caption.paragraph_format.space_after = Pt(2)
            write_runs(caption, line, size=9.5)
            for run in caption.runs:
                run.bold = True
            i += 1
        elif line.startswith("Under the table: "):  # small note below a table
            note = doc.add_paragraph()
            note.paragraph_format.space_before = Pt(2)
            write_runs(note, line[len("Under the table: "):], size=8.5, italic=True)
            i += 1
        elif line.startswith("Authors: ") or line.startswith("Course: "):
            centred = doc.add_paragraph()
            centred.alignment = WD_ALIGN_PARAGRAPH.CENTER
            centred.paragraph_format.space_after = Pt(1)
            write_runs(centred, line)
            i += 1
        else:
            write_runs(doc.add_paragraph(), line)
            i += 1
    doc.save(target)


build()
print("Saved report_summary.docx")
