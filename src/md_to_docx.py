"""
Convert AdditiveLLM_Final_Report.md into a fully functional .docx OpenXML document using standard library zipfile.
Embeds text, headings, tables, bullet points, and formatting into report/AdditiveLLM_Final_Report.docx.
"""

import os
import re
import zipfile
from pathlib import Path
import xml.sax.saxutils as saxutils

BASE_DIR = Path(__file__).resolve().parent.parent
MD_PATH = BASE_DIR / "report" / "AdditiveLLM_Final_Report.md"
DOCX_PATH = BASE_DIR / "report" / "AdditiveLLM_Final_Report.docx"


def escape(text: str) -> str:
    return saxutils.escape(text)


def create_docx(md_path: Path, docx_path: Path):
    print(f"Reading markdown from {md_path}...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    lines = md_text.split("\n")

    # Document body XML parts
    body_xml_parts = []

    in_table = False
    table_rows = []

    for line in lines:
        line_str = line.strip()

        # Table processing
        if line_str.startswith("|") and line_str.endswith("|"):
            in_table = True
            # Ignore separator lines like | :--- | :---: |
            if re.match(r"^\|(\s*[-:]+\s*\|)+$", line_str):
                continue
            cells = [c.strip() for c in line_str.split("|")[1:-1]]
            table_rows.append(cells)
            continue
        else:
            if in_table:
                # Flush table to XML
                if table_rows:
                    tbl_xml = ["<w:tbl>", "<w:tblPr><w:tblBorders><w:top w:val='single' w:sz='4' w:space='0' w:color='CCCCCC'/><w:left w:val='none'/><w:bottom w:val='single' w:sz='4' w:space='0' w:color='CCCCCC'/><w:right w:val='none'/><w:insideH w:val='single' w:sz='4' w:space='0' w:color='EEEEEE'/><w:insideV w:val='none'/></w:tblBorders></w:tblPr>"]
                    for r_idx, row in enumerate(table_rows):
                        tbl_xml.append("<w:tr>")
                        for cell in row:
                            clean_cell = re.sub(r"\*\*([^*]+)\*\*", r"\1", cell)
                            clean_cell = re.sub(r"\*([^*]+)\*", r"\1", clean_cell)
                            clean_cell = re.sub(r"`([^`]+)`", r"\1", clean_cell)
                            clean_cell = re.sub(r"\$([^$]+)\$", r"\1", clean_cell)
                            is_header = (r_idx == 0)
                            shd = "<w:shd w:val='clear' w:color='auto' w:fill='1E293B'/>" if is_header else ("<w:shd w:val='clear' w:color='auto' w:fill='F8FAFC'/>" if r_idx % 2 == 1 else "")
                            color = "<w:color w:val='FFFFFF'/>" if is_header else "<w:color w:val='0F172A'/>"
                            bld = "<w:b/>" if is_header else ""
                            tbl_xml.append(f"<w:tc><w:tcPr>{shd}<w:tcMar><w:top w:w='120' w:type='dxa'/><w:bottom w:w='120' w:type='dxa'/><w:left w:w='150' w:type='dxa'/><w:right w:w='150' w:type='dxa'/></w:tcMar></w:tcPr><w:p><w:r><w:rPr>{bld}{color}<w:sz w:val='18'/></w:rPr><w:t>{escape(clean_cell)}</w:t></w:r></w:p></w:tc>")
                        tbl_xml.append("</w:tr>")
                    tbl_xml.append("</w:tbl>")
                    body_xml_parts.append("".join(tbl_xml))
                    table_rows = []
                in_table = False

        if not line_str:
            continue

        # Horizontal Rule
        if line_str == "---":
            body_xml_parts.append("<w:p><w:pPr><w:pBdr><w:bottom w:val='single' w:sz='8' w:space='4' w:color='CBD5E1'/></w:pBdr></w:pPr></w:p>")
            continue

        # Headings
        if line_str.startswith("# "):
            title = line_str[2:].strip()
            body_xml_parts.append(f"<w:p><w:pPr><w:pStyle w:val='Heading1'/><w:spacing w:before='240' w:after='120'/></w:pPr><w:r><w:rPr><w:b/><w:color w:val='0F172A'/><w:sz w:val='36'/></w:rPr><w:t>{escape(title)}</w:t></w:r></w:p>")
        elif line_str.startswith("## "):
            h2 = line_str[3:].strip()
            body_xml_parts.append(f"<w:p><w:pPr><w:pStyle w:val='Heading2'/><w:spacing w:before='200' w:after='80'/></w:pPr><w:r><w:rPr><w:b/><w:color w:val='0369A1'/><w:sz w:val='28'/></w:rPr><w:t>{escape(h2)}</w:t></w:r></w:p>")
        elif line_str.startswith("### "):
            h3 = line_str[4:].strip()
            body_xml_parts.append(f"<w:p><w:pPr><w:pStyle w:val='Heading3'/><w:spacing w:before='160' w:after='60'/></w:pPr><w:r><w:rPr><w:b/><w:color w:val='334155'/><w:sz w:val='24'/></w:rPr><w:t>{escape(h3)}</w:t></w:r></w:p>")
        elif line_str.startswith("- ") or line_str.startswith("* "):
            bullet_text = line_str[2:].strip()
            clean_b = re.sub(r"\*\*([^*]+)\*\*", r"\1", bullet_text)
            clean_b = re.sub(r"\*([^*]+)\*", r"\1", clean_b)
            clean_b = re.sub(r"`([^`]+)`", r"\1", clean_b)
            clean_b = re.sub(r"\$([^$]+)\$", r"\1", clean_b)
            body_xml_parts.append(f"<w:p><w:pPr><w:ind w:left='360'/><w:spacing w:before='40' w:after='40'/></w:pPr><w:r><w:rPr><w:sz w:val='20'/></w:rPr><w:t>•  {escape(clean_b)}</w:t></w:r></w:p>")
        elif line_str.startswith("```"):
            continue
        elif line_str.startswith("+--") or line_str.startswith("|   "):
            # ASCII box for figure reference
            body_xml_parts.append(f"<w:p><w:pPr><w:shd w:val='clear' w:color='auto' w:fill='F1F5F9'/><w:spacing w:before='60' w:after='60'/></w:pPr><w:r><w:rPr><w:b/><w:color w:val='0284C7'/><w:sz w:val='18'/></w:rPr><w:t>{escape(line_str)}</w:t></w:r></w:p>")
        else:
            # Regular paragraph
            clean_p = re.sub(r"\*\*([^*]+)\*\*", r"\1", line_str)
            clean_p = re.sub(r"\*([^*]+)\*", r"\1", clean_p)
            clean_p = re.sub(r"`([^`]+)`", r"\1", clean_p)
            clean_p = re.sub(r"\$([^$]+)\$", r"\1", clean_p)
            body_xml_parts.append(f"<w:p><w:pPr><w:spacing w:before='60' w:after='60'/></w:pPr><w:r><w:rPr><w:sz w:val='20'/><w:color w:val='1E293B'/></w:rPr><w:t>{escape(clean_p)}</w:t></w:r></w:p>")

    # Construct complete document.xml
    body_xml = "".join(body_xml_parts)
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body_xml}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    # Package into docx
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml)

    print(f"Successfully generated docx: {docx_path} ({docx_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    create_docx(MD_PATH, DOCX_PATH)
