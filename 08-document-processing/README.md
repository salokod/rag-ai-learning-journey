# Module 08: Document Processing & Data Pipelines

## Goal
Build a robust document ingestion pipeline that handles real-world manufacturing documents — PDFs, Word docs, and structured data — preparing them for your RAG system.

---

## Concepts

### Real Documents Are Messy

Manufacturing environments have:
- **PDFs** — SOPs, specifications, drawings (often scanned)
- **Word docs** — Procedures, work instructions, templates
- **Spreadsheets** — Inspection data, BOM lists, calibration records
- **Legacy formats** — Plain text files, HTML pages from old systems

### The Ingestion Pipeline

```
Raw Documents → Parse/Extract → Clean → Chunk → Embed → Store
     ↓              ↓            ↓        ↓       ↓       ↓
  PDF/DOCX     Text content   Remove    Split   Vectors  ChromaDB
               + metadata     noise     smart
```

---

## Exercise 1: PDF Processing

```python
# 08-document-processing/ex1_pdf_processing.py
"""Parse PDF documents for RAG ingestion."""

from pypdf import PdfReader
import os

# Create a sample PDF for demonstration
# In your real project, you'd point this at actual manufacturing PDFs
def create_sample_pdf():
    """Create a sample manufacturing SOP as PDF."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        filename = "08-document-processing/sample_sop.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 750, "SOP-WLD-001: MIG Welding Procedure")
        c.setFont("Helvetica", 11)

        lines = [
            "Department: Welding | Revision: C | Date: 2024-01-15",
            "",
            "1. SCOPE",
            "This procedure covers GMAW (MIG) welding of carbon steel",
            "components per AWS D1.1 structural welding code.",
            "",
            "2. SAFETY REQUIREMENTS",
            "- Welding helmet with auto-darkening lens (shade 10-13)",
            "- Leather welding gloves and FR clothing",
            "- Ensure adequate ventilation or use fume extraction",
            "- Fire watch required when welding near combustibles",
            "",
            "3. PROCEDURE",
            "3.1 Prepare base metal: remove rust, oil, and mill scale",
            "3.2 Set up welder per WPS (see specification WPS-201)",
            "3.3 Tack weld components per assembly drawing",
            "3.4 Complete weld passes per WPS sequence",
            "3.5 Allow to cool naturally - do not quench",
            "",
            "4. INSPECTION",
            "4.1 Visual inspection per AWS D1.1 Section 6",
            "4.2 Record results on Form QC-107",
            "4.3 UT testing required for critical joints (see drawing)",
        ]

        y = 720
        for line in lines:
            c.drawString(72, y, line)
            y -= 16

        c.save()
        return filename
    except ImportError:
        print("Note: reportlab not installed. Using text-based demo instead.")
        return None


def process_pdf(filepath: str) -> list[dict]:
    """Extract text and metadata from a PDF."""
    reader = PdfReader(filepath)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages.append({
                "page_number": i + 1,
                "text": text.strip(),
                "char_count": len(text),
                "source": os.path.basename(filepath),
            })

    return pages


def chunk_document(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Chunk extracted pages into RAG-ready pieces."""
    full_text = "\n\n".join(p["text"] for p in pages)
    source = pages[0]["source"] if pages else "unknown"

    # Try section-based chunking first
    import re
    sections = re.split(r'\n(?=\d+\.)', full_text)

    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if len(section) > 50:  # Skip tiny fragments
            chunks.append({
                "id": f"{source}-chunk-{i}",
                "text": section,
                "metadata": {
                    "source": source,
                    "chunk_index": i,
                    "chunk_method": "section",
                },
            })

    return chunks


# Demo
pdf_path = create_sample_pdf()
if pdf_path:
    pages = process_pdf(pdf_path)
    print(f"=== Extracted {len(pages)} pages ===")
    for p in pages:
        print(f"  Page {p['page_number']}: {p['char_count']} chars")

    chunks = chunk_document(pages)
    print(f"\n=== Created {len(chunks)} chunks ===")
    for chunk in chunks:
        print(f"  {chunk['id']}: {chunk['text'][:80]}...")
else:
    print("Demo: In practice, you'd process your actual PDF SOPs here.")
    print("Install reportlab to generate sample PDFs: pip install reportlab")

print("\n=== PDF Processing Tips ===")
print("1. Scanned PDFs need OCR (use pytesseract or unstructured)")
print("2. Tables in PDFs are HARD — use unstructured or camelot library")
print("3. Always preserve source + page number in metadata for citations")
print("4. Manufacturing drawings (engineering drawings) need special handling")
```

---

## Exercise 2: Word Document Processing

```python
# 08-document-processing/ex2_docx_processing.py
"""Process Word documents for RAG ingestion."""

from docx import Document
import os


def create_sample_docx():
    """Create a sample manufacturing work instruction."""
    doc = Document()
    doc.add_heading("Work Instruction: WI-ASM-004", level=1)
    doc.add_paragraph("Assembly of Hydraulic Manifold Block")
    doc.add_paragraph("")
    doc.add_heading("1. Required Materials", level=2)
    doc.add_paragraph("• Manifold block (P/N HM-2200)")
    doc.add_paragraph("• O-ring kit (P/N OR-KIT-440)")
    doc.add_paragraph("• Torque wrench (1/4\" drive, 5-25 Nm)")
    doc.add_paragraph("")
    doc.add_heading("2. Assembly Steps", level=2)
    doc.add_paragraph("2.1 Clean all port surfaces with isopropyl alcohol")
    doc.add_paragraph("2.2 Install O-rings in each port groove — verify no twists")
    doc.add_paragraph("2.3 Install fittings hand-tight, then torque to 15 Nm")
    doc.add_paragraph("2.4 Connect test port and pressurize to 3000 PSI")
    doc.add_paragraph("2.5 Hold pressure for 5 minutes — zero leakage allowed")
    doc.add_paragraph("")
    doc.add_heading("3. Quality Sign-off", level=2)
    doc.add_paragraph("Record serial number and test results on Form QC-HM-001")

    filepath = "08-document-processing/sample_work_instruction.docx"
    doc.save(filepath)
    return filepath


def process_docx(filepath: str) -> list[dict]:
    """Extract structured content from a Word document."""
    doc = Document(filepath)
    sections = []
    current_section = {"heading": "", "content": []}

    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            # Save previous section
            if current_section["content"]:
                sections.append({
                    "heading": current_section["heading"],
                    "text": "\n".join(current_section["content"]),
                    "source": os.path.basename(filepath),
                })
            current_section = {"heading": para.text, "content": []}
        elif para.text.strip():
            current_section["content"].append(para.text)

    # Don't forget the last section
    if current_section["content"]:
        sections.append({
            "heading": current_section["heading"],
            "text": "\n".join(current_section["content"]),
            "source": os.path.basename(filepath),
        })

    return sections


# Process the sample document
filepath = create_sample_docx()
sections = process_docx(filepath)

print(f"=== Extracted {len(sections)} sections from DOCX ===\n")
for section in sections:
    print(f"Section: {section['heading']}")
    print(f"  Content: {section['text'][:100]}...")
    print()

print("=== DOCX Processing Advantages ===")
print("Word documents preserve STRUCTURE (headings, lists, tables)")
print("This makes section-based chunking much more reliable than PDFs")
```

---

## Exercise 3: Building a Document Ingestion Pipeline

```python
# 08-document-processing/ex3_ingestion_pipeline.py
"""A complete document ingestion pipeline that feeds your RAG system."""

import chromadb
import os
import hashlib
from pathlib import Path


class DocumentIngestionPipeline:
    """Ingest documents from multiple formats into a vector store."""

    def __init__(self, collection_name: str = "manufacturing_docs"):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "Ingested manufacturing documents"},
        )
        self.stats = {"processed": 0, "chunks": 0, "errors": 0}

    def _make_chunk_id(self, source: str, index: int) -> str:
        """Create a deterministic chunk ID."""
        content = f"{source}-{index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _chunk_text(self, text: str, source: str, chunk_size: int = 500) -> list[dict]:
        """Smart chunking: try sections first, fall back to paragraphs, then fixed size."""
        import re

        # Strategy 1: Section headers (numbered)
        sections = re.split(r'\n(?=\d+[\.\)])', text)
        if len(sections) > 1:
            method = "section"
            pieces = sections
        else:
            # Strategy 2: Paragraphs (double newline)
            paragraphs = text.split("\n\n")
            if len(paragraphs) > 1:
                method = "paragraph"
                pieces = paragraphs
            else:
                # Strategy 3: Fixed size with overlap
                method = "fixed"
                pieces = []
                for i in range(0, len(text), chunk_size - 50):
                    pieces.append(text[i : i + chunk_size])

        chunks = []
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if len(piece) > 30:  # Skip tiny fragments
                chunks.append({
                    "id": self._make_chunk_id(source, i),
                    "text": piece,
                    "metadata": {
                        "source": source,
                        "chunk_index": i,
                        "chunk_method": method,
                        "char_count": len(piece),
                    },
                })

        return chunks

    def ingest_text(self, text: str, source: str, extra_metadata: dict = None):
        """Ingest a plain text document."""
        chunks = self._chunk_text(text, source)
        if not chunks:
            return

        metadata_list = []
        for chunk in chunks:
            meta = chunk["metadata"]
            if extra_metadata:
                meta.update(extra_metadata)
            metadata_list.append(meta)

        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=metadata_list,
        )

        self.stats["processed"] += 1
        self.stats["chunks"] += len(chunks)
        print(f"  ✓ Ingested '{source}': {len(chunks)} chunks ({chunks[0]['metadata']['chunk_method']})")

    def search(self, query: str, n_results: int = 3, **filters) -> list[dict]:
        """Search the ingested documents."""
        kwargs = {"query_texts": [query], "n_results": n_results}
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)
        return [
            {
                "text": doc,
                "source": meta["source"],
                "score": 1 - dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def get_stats(self) -> dict:
        """Return ingestion statistics."""
        return {**self.stats, "total_in_store": self.collection.count()}


# === Use the pipeline ===
pipeline = DocumentIngestionPipeline()

# Ingest various document types
documents = {
    "SOP-WLD-001": {
        "text": """1. SCOPE
This procedure covers GMAW welding of carbon steel per AWS D1.1.

2. SAFETY
Welding helmet (shade 10-13), leather gloves, FR clothing required.
Ensure ventilation or use fume extraction. Fire watch for combustibles.

3. PROCEDURE
3.1 Clean base metal of rust, oil, mill scale
3.2 Set up welder per WPS-201
3.3 Tack weld per assembly drawing
3.4 Complete weld passes per WPS sequence
3.5 Cool naturally — do not quench

4. INSPECTION
Visual inspection per AWS D1.1 Section 6
Record on Form QC-107""",
        "metadata": {"department": "welding", "type": "SOP"},
    },
    "SOP-CNC-042": {
        "text": """CNC Daily Startup:
1. Visual inspection of machine and work area
2. Check coolant level, refill if below MIN
3. Check way oil level, refill if below MIN
4. Power on, home all axes
5. Run spindle warmup (O9000): 500 RPM 5 min, 2000 RPM 5 min
6. Verify axis positions with indicator
7. Check air pressure: min 80 PSI
8. Log on daily checklist form""",
        "metadata": {"department": "machining", "type": "SOP"},
    },
    "SPEC-MT-302": {
        "text": """Torque Specification MT-302 for Frame Assembly #4200
Fasteners: Grade 8 zinc plated
M8 bolts: 25-30 Nm
M10 bolts: 45-55 Nm
M12 bolts: 80-100 Nm
Sequence: Star pattern per diagram
Tool: Calibrated torque wrench ±2%
QC: 10% sampling after assembly
Document on Form QC-110""",
        "metadata": {"department": "assembly", "type": "specification"},
    },
}

print("=== Ingesting Documents ===")
for name, doc in documents.items():
    pipeline.ingest_text(doc["text"], source=name, extra_metadata=doc["metadata"])

stats = pipeline.get_stats()
print(f"\n=== Pipeline Stats ===")
print(f"Documents processed: {stats['processed']}")
print(f"Total chunks: {stats['chunks']}")
print(f"In vector store: {stats['total_in_store']}")

# Test retrieval
print("\n=== Retrieval Tests ===")
queries = [
    "What PPE do I need for welding?",
    "How do I start up the CNC machine?",
    "What torque for M10 bolts?",
]

for query in queries:
    results = pipeline.search(query, n_results=2)
    print(f"\nQ: {query}")
    for r in results:
        print(f"  [{r['source']}] (score: {r['score']:.3f}) {r['text'][:80]}...")
```

---

## Takeaways

1. **Real documents need real parsing** — PDFs, DOCX, and other formats each have quirks
2. **Preserve metadata** — source file, page number, section heading are all vital for citations
3. **Chunking strategy depends on document structure** — use section-based for SOPs, fixed-size as fallback
4. **Build a reusable pipeline** — you'll ingest dozens of documents for your capstone
5. **The quality of your chunks directly determines RAG quality** — garbage in, garbage out

## Setting the Stage for Module 09

You now have a complete RAG system: documents ingested, chunked, embedded, and retrievable. But here's the critical question: **how do you know it's actually good?** "It looks right to me" doesn't cut it in production. Module 09 begins the deep dive into **evaluation** — the skill that separates hobby projects from professional AI systems.
