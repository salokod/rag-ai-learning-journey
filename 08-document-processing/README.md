# Module 08: Document Processing

## Real Documents Are Messy

So far, we've been working with clean text strings we typed ourselves. In reality, your manufacturing knowledge lives in PDFs, Word docs, and other formats that don't just hand you clean text.

Let's see what that looks like. First, make sure you have the libraries:

```bash
pip install pypdf python-docx reportlab
```

---

## Step 1: Create a Sample PDF

We'll create a manufacturing SOP as a PDF, then try to extract text from it. This simulates working with the PDFs your company already has.

```bash
touch 08-document-processing/doc_workshop.py
```

```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_sample_pdf(filename: str) -> str:
    """Create a sample manufacturing SOP as a PDF."""
    c = canvas.Canvas(filename, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 750, "SOP-WLD-001: MIG Welding Procedure")

    # Metadata line
    c.setFont("Helvetica", 10)
    c.drawString(72, 730, "Department: Welding | Revision: C | Date: 2024-01-15")

    # Body
    c.setFont("Helvetica", 11)
    lines = [
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

    y = 710
    for line in lines:
        c.drawString(72, y, line)
        y -= 16

    c.save()
    return filename

pdf_path = create_sample_pdf("08-document-processing/sample_sop.pdf")
print(f"Created: {pdf_path}")
```

Run it. You should now have a PDF file. Open it if you want -- it looks like a real SOP.

---

## Step 2: Extract Text from the PDF

Now let's see what happens when we try to read it back:

```python
from pypdf import PdfReader

reader = PdfReader("08-document-processing/sample_sop.pdf")
print(f"Pages: {len(reader.pages)}")

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f"\n--- Page {i + 1} ({len(text)} chars) ---")
    print(text)
```

Look at the output carefully. A few things to notice:
- The text comes out mostly clean because we created a simple PDF
- But the structure is flattened -- headings, bullet points, and spacing may not look the same
- Line breaks might appear in odd places

Now imagine this with a scanned PDF from a 10-year-old SOP binder. The text extraction would be much worse (or nonexistent without OCR). For now, we're working with "born digital" PDFs, which is the easier case.

Let's wrap this into a function:

```python
import os

def extract_pdf(filepath: str) -> list[dict]:
    """Extract text and metadata from each page of a PDF."""
    reader = PdfReader(filepath)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "page": i + 1,
                "text": text.strip(),
                "chars": len(text.strip()),
                "source": os.path.basename(filepath),
            })

    return pages

pages = extract_pdf("08-document-processing/sample_sop.pdf")
print(f"Extracted {len(pages)} pages")
for p in pages:
    print(f"  Page {p['page']}: {p['chars']} chars")
```

That's the raw material. But a full page of text is too long for a single RAG chunk -- we'll need to split it up. We'll get to chunking in a few steps.

---

## Step 3: Create and Process a Word Document

Word documents are actually easier to work with because they preserve structure -- headings, paragraphs, lists are all tagged.

```python
from docx import Document

def create_sample_docx(filename: str) -> str:
    """Create a sample work instruction as a Word doc."""
    doc = Document()

    doc.add_heading("Work Instruction: WI-ASM-004", level=1)
    doc.add_paragraph("Assembly of Hydraulic Manifold Block")

    doc.add_heading("1. Required Materials", level=2)
    doc.add_paragraph("- Manifold block (P/N HM-2200)")
    doc.add_paragraph("- O-ring kit (P/N OR-KIT-440)")
    doc.add_paragraph("- Torque wrench (1/4\" drive, 5-25 Nm)")

    doc.add_heading("2. Assembly Steps", level=2)
    doc.add_paragraph("2.1 Clean all port surfaces with isopropyl alcohol")
    doc.add_paragraph("2.2 Install O-rings in each port groove - verify no twists")
    doc.add_paragraph("2.3 Install fittings hand-tight, then torque to 15 Nm")
    doc.add_paragraph("2.4 Connect test port and pressurize to 3000 PSI")
    doc.add_paragraph("2.5 Hold pressure for 5 minutes - zero leakage allowed")

    doc.add_heading("3. Quality Sign-off", level=2)
    doc.add_paragraph("Record serial number and test results on Form QC-HM-001")
    doc.add_paragraph("Both assembler and QC inspector must sign off")

    doc.save(filename)
    return filename

docx_path = create_sample_docx("08-document-processing/sample_work_instruction.docx")
print(f"Created: {docx_path}")
```

Now extract it with structure preserved:

```python
def extract_docx(filepath: str) -> list[dict]:
    """Extract structured sections from a Word document."""
    doc = Document(filepath)
    sections = []
    current = {"heading": "", "content": []}

    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            # Save the previous section
            if current["content"]:
                sections.append({
                    "heading": current["heading"],
                    "text": "\n".join(current["content"]),
                    "source": os.path.basename(filepath),
                })
            # Start a new section
            current = {"heading": para.text, "content": []}
        elif para.text.strip():
            current["content"].append(para.text)

    # Don't forget the last section
    if current["content"]:
        sections.append({
            "heading": current["heading"],
            "text": "\n".join(current["content"]),
            "source": os.path.basename(filepath),
        })

    return sections

sections = extract_docx("08-document-processing/sample_work_instruction.docx")
print(f"Extracted {len(sections)} sections:\n")
for s in sections:
    print(f"  [{s['heading']}]")
    print(f"  {s['text'][:100]}...")
    print()
```

Notice what we get: clean sections, each with a heading and its content. This is much better than a raw text dump. Each section is already a natural chunk -- "Required Materials" is one chunk, "Assembly Steps" is another.

**This is why Word docs are your friend for RAG.** The structure is built in. PDFs throw it all away and give you a flat string.

---

## Step 4: Chunking Strategies -- Side by Side

Chunking is how you split a document into pieces for embedding. The strategy you choose directly affects what your RAG system retrieves. Let's try three approaches on the same text and see the difference.

First, let's get a decent-sized text to work with:

```python
sample_text = """1. SCOPE
This procedure covers GMAW (MIG) welding of carbon steel components per AWS D1.1 structural welding code. It applies to all welding operations in Building 3.

2. SAFETY REQUIREMENTS
All welders must wear: auto-darkening helmet (shade 10-13), leather welding gloves, FR clothing, and steel-toe boots. Safety glasses must be worn under the helmet. Ensure adequate ventilation or use fume extraction system. Fire watch is required when welding within 35 feet of combustible materials. Hot work permit required per SOP-SAFE-003.

3. PROCEDURE
3.1 Prepare base metal: remove all rust, oil, paint, and mill scale from weld zone plus 1 inch on each side.
3.2 Fit-up parts per assembly drawing. Gap tolerance: 1/16 inch maximum.
3.3 Set up welder per WPS-201: Wire ER70S-6, 0.035 inch, Gas 75/25 Ar/CO2 at 25-30 CFH.
3.4 Tack weld at 6 inch intervals. Minimum tack size: 1 inch long.
3.5 Complete weld passes per WPS sequence. Interpass temp 400F max.
3.6 Allow to cool naturally. Do not quench with water or air.

4. INSPECTION
4.1 Visual inspection per AWS D1.1 Section 6 within 24 hours of welding.
4.2 Acceptance criteria: no cracks, no incomplete fusion, undercut max 1/32 inch, porosity max 3/8 inch in any 12 inch length.
4.3 Record results on Form QC-107. Inspector badge number and date required.
4.4 UT testing required for all critical joints as marked on drawing.
4.5 Failed welds: mark with yellow paint, document on NCR form, notify welding supervisor."""
```

**Strategy 1: Fixed-size chunks.** Just chop the text every N characters with some overlap:

```python
def chunk_fixed(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """Fixed-size chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 20]

fixed_chunks = chunk_fixed(sample_text)
print(f"Fixed-size: {len(fixed_chunks)} chunks\n")
for i, chunk in enumerate(fixed_chunks):
    print(f"  Chunk {i}: ({len(chunk)} chars) {chunk[:60]}...")
```

Look at the output. Notice how chunks cut right through the middle of sentences and sections. Chunk 2 might start halfway through the safety section and end in the middle of the procedure section. That's a problem -- when RAG retrieves this chunk, the context is fragmented.

**Strategy 2: Section-based chunks.** Split on section headers:

```python
import re

def chunk_by_section(text: str) -> list[str]:
    """Split on numbered section headers."""
    sections = re.split(r'\n(?=\d+\.)', text)
    return [s.strip() for s in sections if len(s.strip()) > 30]

section_chunks = chunk_by_section(sample_text)
print(f"Section-based: {len(section_chunks)} chunks\n")
for i, chunk in enumerate(section_chunks):
    print(f"  Chunk {i}: ({len(chunk)} chars) {chunk[:60]}...")
```

Much better. Each chunk is a complete section -- scope, safety, procedure, inspection. When RAG retrieves the safety chunk, it gets ALL the safety info, not a fragment.

**Strategy 3: Paragraph-based chunks.** Split on double newlines:

```python
def chunk_by_paragraph(text: str) -> list[str]:
    """Split on paragraph boundaries (double newlines)."""
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if len(p.strip()) > 30]

para_chunks = chunk_by_paragraph(sample_text)
print(f"Paragraph-based: {len(para_chunks)} chunks\n")
for i, chunk in enumerate(para_chunks):
    print(f"  Chunk {i}: ({len(chunk)} chars) {chunk[:60]}...")
```

Paragraph chunks are smaller than section chunks. That can be good (more precise retrieval) or bad (less context per chunk).

Let's see them all side by side:

```python
print("=== Chunk Count Comparison ===")
print(f"  Fixed (300 char): {len(fixed_chunks)} chunks")
print(f"  Section-based:    {len(section_chunks)} chunks")
print(f"  Paragraph-based:  {len(para_chunks)} chunks")

print("\n=== Average Chunk Size ===")
for name, chunks in [("Fixed", fixed_chunks), ("Section", section_chunks), ("Paragraph", para_chunks)]:
    avg = sum(len(c) for c in chunks) / len(chunks)
    print(f"  {name}: {avg:.0f} chars average")
```

For manufacturing SOPs, section-based chunking usually wins. Here's why: when someone asks "What PPE do I need for welding?", you want the ENTIRE safety section, not a fragment. But if your sections are very long (multi-page procedures), you might need to split further within sections.

---

## Step 5: Build a Reusable Ingestion Pipeline

Now let's combine everything into a class you can reuse. We'll build it piece by piece.

Start with the skeleton:

```python
import chromadb
import hashlib

class IngestionPipeline:
    """Ingest documents from multiple formats into ChromaDB."""

    def __init__(self, collection_name: str = "manufacturing_docs"):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)
        self.stats = {"files": 0, "chunks": 0, "errors": 0}

    def _make_id(self, source: str, index: int) -> str:
        """Deterministic chunk ID from source and index."""
        return hashlib.md5(f"{source}-{index}".encode()).hexdigest()[:12]
```

Add the smart chunking method -- it tries sections first, then paragraphs, then fixed-size as a fallback:

```python
    def _chunk(self, text: str, source: str) -> list[dict]:
        """Smart chunking: section -> paragraph -> fixed-size fallback."""

        # Try section-based first
        sections = re.split(r'\n(?=\d+[\.\)])', text)
        if len(sections) > 1:
            method = "section"
            pieces = sections
        else:
            # Try paragraph-based
            paragraphs = text.split("\n\n")
            if len(paragraphs) > 1:
                method = "paragraph"
                pieces = paragraphs
            else:
                # Fall back to fixed-size
                method = "fixed"
                pieces = []
                for i in range(0, len(text), 450):
                    pieces.append(text[i:i + 500])

        chunks = []
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if len(piece) > 30:
                chunks.append({
                    "id": self._make_id(source, i),
                    "text": piece,
                    "metadata": {
                        "source": source,
                        "chunk_index": i,
                        "chunk_method": method,
                        "char_count": len(piece),
                    },
                })
        return chunks
```

Add methods to ingest different file types:

```python
    def ingest_pdf(self, filepath: str, extra_metadata: dict = None):
        """Ingest a PDF file."""
        try:
            pages = extract_pdf(filepath)
            full_text = "\n\n".join(p["text"] for p in pages)
            source = os.path.basename(filepath)
            self._ingest_text(full_text, source, extra_metadata)
        except Exception as e:
            print(f"  ERROR processing {filepath}: {e}")
            self.stats["errors"] += 1

    def ingest_docx(self, filepath: str, extra_metadata: dict = None):
        """Ingest a Word document using its heading structure."""
        try:
            sections = extract_docx(filepath)
            source = os.path.basename(filepath)

            # Each section becomes its own chunk (already well-structured)
            for i, section in enumerate(sections):
                chunk_text = f"{section['heading']}\n{section['text']}" if section["heading"] else section["text"]
                chunk_id = self._make_id(source, i)
                meta = {
                    "source": source,
                    "heading": section["heading"],
                    "chunk_index": i,
                    "chunk_method": "docx_heading",
                }
                if extra_metadata:
                    meta.update(extra_metadata)

                self.collection.add(
                    ids=[chunk_id],
                    documents=[chunk_text],
                    metadatas=[meta],
                )
                self.stats["chunks"] += 1

            self.stats["files"] += 1
            print(f"  Ingested {source}: {len(sections)} sections (docx_heading)")
        except Exception as e:
            print(f"  ERROR processing {filepath}: {e}")
            self.stats["errors"] += 1

    def ingest_text(self, text: str, source: str, extra_metadata: dict = None):
        """Ingest raw text."""
        self._ingest_text(text, source, extra_metadata)

    def _ingest_text(self, text: str, source: str, extra_metadata: dict = None):
        """Internal: chunk and store text."""
        chunks = self._chunk(text, source)
        if not chunks:
            return

        for chunk in chunks:
            meta = chunk["metadata"]
            if extra_metadata:
                meta.update(extra_metadata)

        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

        self.stats["files"] += 1
        self.stats["chunks"] += len(chunks)
        print(f"  Ingested {source}: {len(chunks)} chunks ({chunks[0]['metadata']['chunk_method']})")
```

Add a search method and stats:

```python
    def search(self, query: str, n_results: int = 3, **filters) -> list[dict]:
        """Search ingested documents."""
        kwargs = {"query_texts": [query], "n_results": n_results}
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)
        return [
            {
                "text": doc,
                "source": meta["source"],
                "score": round(1 - dist, 3),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def get_stats(self) -> dict:
        return {**self.stats, "total_stored": self.collection.count()}
```

---

## Step 6: Ingest Multiple Documents and Query Across Them

Let's put the pipeline to work. We'll ingest the PDF, the Word doc, and some raw text documents, then query across all of them:

```python
pipeline = IngestionPipeline(collection_name="all_docs")

print("=== Ingesting Documents ===")

# Ingest the PDF we created
pipeline.ingest_pdf(
    "08-document-processing/sample_sop.pdf",
    extra_metadata={"department": "welding", "type": "SOP"},
)

# Ingest the Word doc we created
pipeline.ingest_docx(
    "08-document-processing/sample_work_instruction.docx",
    extra_metadata={"department": "assembly", "type": "work_instruction"},
)

# Ingest some raw text docs too
pipeline.ingest_text(
    """CNC Daily Startup Procedure
1. Visual inspection of machine and work area
2. Check coolant level, refill if below MIN
3. Check way oil level, refill if below MIN
4. Power on, home all axes
5. Run spindle warmup (Program O9000): 500 RPM 5 min, 2000 RPM 5 min
6. Verify axis positions with test indicator
7. Check air pressure: minimum 80 PSI
8. Log on daily checklist form""",
    source="SOP-CNC-042",
    extra_metadata={"department": "machining", "type": "SOP"},
)

pipeline.ingest_text(
    """Torque Specification MT-302 for Frame Assembly #4200
Fasteners: Grade 8 zinc plated
M8 bolts: 25-30 Nm
M10 bolts: 45-55 Nm
M12 bolts: 80-100 Nm
Sequence: Star pattern per diagram
Tool: Calibrated torque wrench +/-2%
QC: 10% sampling after assembly
Document on Form QC-110""",
    source="SPEC-MT-302",
    extra_metadata={"department": "assembly", "type": "specification"},
)

stats = pipeline.get_stats()
print(f"\n=== Pipeline Stats ===")
print(f"  Files processed: {stats['files']}")
print(f"  Total chunks: {stats['chunks']}")
print(f"  Errors: {stats['errors']}")
print(f"  In vector store: {stats['total_stored']}")
```

Now the fun part -- query across all of them:

```python
print("\n=== Cross-Document Queries ===\n")

queries = [
    "What PPE do I need for welding?",
    "How do I assemble the hydraulic manifold?",
    "What torque for M10 bolts?",
    "How do I start up the CNC machine?",
    "What form do I use for weld inspection?",
]

for query in queries:
    results = pipeline.search(query, n_results=2)
    print(f"Q: {query}")
    for r in results:
        print(f"  [{r['source']}] (score: {r['score']}) {r['text'][:80]}...")
    print()
```

Notice how queries pull from different source files. "What PPE do I need for welding?" finds the PDF SOP. "How do I assemble the hydraulic manifold?" finds the Word doc. "What torque for M10 bolts?" finds the raw text spec. The pipeline doesn't care what format the original document was -- everything is searchable.

Try filtering by department:

```python
print("=== Filtered: Assembly department only ===")
results = pipeline.search("procedures", n_results=3, department="assembly")
for r in results:
    print(f"  [{r['source']}] {r['text'][:80]}...")
```

---

## Step 7: Understand What Can Go Wrong

Before you move on, let's talk about the reality of document processing in manufacturing. Run this thought experiment:

```python
# What real-world PDFs look like
print("=== Real-World PDF Challenges ===")
problems = [
    ("Scanned PDFs", "Need OCR (pytesseract, unstructured). Text extraction returns empty."),
    ("Tables in PDFs", "Columns and rows get jumbled. Use camelot or unstructured library."),
    ("Engineering drawings", "Mostly graphics. Title block text is extractable, dimensions are not."),
    ("Multi-column layouts", "Text extraction reads left-to-right, jumbling columns together."),
    ("Headers/footers", "Page numbers and doc titles repeat in every chunk."),
    ("Redline markups", "Tracked changes in PDFs may extract both old and new text."),
]

for problem, note in problems:
    print(f"  {problem}")
    print(f"    -> {note}")
    print()
```

For your manufacturing project, here's the practical advice:
- Start with Word docs if you have them -- they're the easiest to process
- Born-digital PDFs (created from Word/software) are next best
- Scanned PDFs need OCR first -- that's a whole separate step
- Engineering drawings are the hardest -- focus on the text-based SOPs first

## Key Takeaways

- **PDFs flatten structure** -- you lose headings, lists, and formatting during extraction
- **Word docs preserve structure** -- headings become natural chunk boundaries
- **Chunking strategy matters** -- section-based beats fixed-size for SOPs and procedures
- **Build a reusable pipeline** -- you'll ingest dozens (maybe hundreds) of documents
- **Preserve metadata** -- source file, department, doc type all help with filtered retrieval
- **The quality of your chunks directly determines RAG quality** -- spend time getting this right

Your RAG system now has a complete front-to-back pipeline: documents go in (any format), get chunked and embedded, and come out as answers. But here's the question Module 09 will tackle: **how do you know it's actually good?** "It looks right to me" doesn't cut it in manufacturing. You need measurement.
