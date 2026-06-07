"""Generate PDF files from markdown FAQ files."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
import markdown2


def markdown_to_pdf(md_file: str, pdf_file: str):
    """Convert a markdown file to PDF."""
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(md_content)
    
    # Create PDF
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=14,
        alignment=TA_LEFT,
    ))
    
    # Split content into lines and process
    lines = md_content.split('\n')
    for line in lines:
        if line.startswith('# '):
            # Main heading
            text = line[2:].strip()
            elements.append(Paragraph(text, styles['Title']))
            elements.append(Spacer(1, 0.3*inch))
        elif line.startswith('## '):
            # Subheading
            text = line[3:].strip()
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph(text, styles['Heading2']))
            elements.append(Spacer(1, 0.1*inch))
        elif line.startswith('### '):
            # Sub-subheading
            text = line[4:].strip()
            elements.append(Paragraph(text, styles['Heading3']))
            elements.append(Spacer(1, 0.05*inch))
        elif line.strip():
            # Regular text
            elements.append(Paragraph(line.strip(), styles['CustomBody']))
            elements.append(Spacer(1, 0.1*inch))
        else:
            # Empty line
            elements.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    doc.build(elements)
    print(f"✓ Created {pdf_file}")


def main():
    """Generate all FAQ PDFs."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # List of markdown files to convert
    md_files = [
        'billing_faq.md',
        'general_issues_faq.md'
    ]
    
    print("Generating PDF files from markdown...")
    print("=" * 50)
    
    for md_file in md_files:
        md_path = os.path.join(data_dir, md_file)
        pdf_file = md_file.replace('.md', '.pdf')
        pdf_path = os.path.join(data_dir, pdf_file)
        
        if os.path.exists(md_path):
            try:
                markdown_to_pdf(md_path, pdf_path)
            except Exception as e:
                print(f"✗ Error creating {pdf_file}: {e}")
        else:
            print(f"✗ Markdown file not found: {md_path}")
    
    print("=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
