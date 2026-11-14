import os
import weasyprint

def main():
    base_dir = os.path.abspath(".")
    html_file = os.path.join(base_dir, "exp012_report.html")
    pdf_file = os.path.join(base_dir, "EXP012_BENCHMARK_REPORT.pdf")
    
    print(f"Loading {html_file}...")
    html_obj = weasyprint.HTML(filename=html_file, base_url=base_dir)
    print("Rendering and compiling to PDF...")
    html_obj.write_pdf(target=pdf_file)
    print(f"Generated PDF at: {pdf_file}")
    
    file_size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
