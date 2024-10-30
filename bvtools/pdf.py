"""
Tools and utilities for working with PDF files.
Some require that `qpdf` is installed on the system
"""

import re
import subprocess

from betterpathlib import Path


def split_pdf(
    input_pdf: str | Path,
    output_dir: str | Path | None = None,
    max_pages: int = 2,
    max_size_mib: int | float | None = None,
) -> list[Path]:
    """
    Split the input PDF into parts with `max_pages` pages or less. If `max_size_mib` is given, the PDF
    if split further, if possible, to adhere.

    Page PDFs are stored in the output_dir, which is inferred if not given.
    """
    input_pdf = Path(input_pdf)
    if not output_dir:
        output_dir = input_pdf.without_suffix(".pdf")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "page_%d.pdf"

    def split(input_pdf_, output_path_, max_pages_):
        split_command = [
            "qpdf",
            f"--split-pages={max_pages_}",
            str(input_pdf_),
            str(output_path_),
        ]
        # Split the PDF into parts with max_pages
        subprocess.run(split_command, check=True)

    split(input_pdf, output_path, max_pages)

    if max_size_mib:
        while True:
            # Split all PDFs that are larger than max_size_mb that can be split (ie has 2 or more pages)
            big_pdfs = [
                p
                for p in output_dir.glob("page_*")
                if (p.size() / (1024 * 1024) > max_size_mib)
                and re.search(r"(\d+)-(\d+)", str(p))
            ]
            if not big_pdfs:
                break
            max_pages = max_pages // 2
            for path in big_pdfs:
                output_path_ = output_dir / path.with_stem(path.stem + "_split_%d")
                split(path, output_path_, max_pages)
                path.unlink()

    return sorted(list(output_dir.glob("page_*")))


def extract_pages(path: Path | str, pages: int | str, output_dir: Path | str | None = None) -> Path:
    """
    Extract given page, pages, or page range from a PDF file to a new file.
    Return the Path of the extracted PDF file

    pages: e.g. '6' or '6-10'
    """
    path = Path(path)
    pages = str(pages)
    if not path.exists():
        raise FileNotFoundError(path)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = path.parent
    output_path = output_dir / (path.stem + f"_pages_{pages}")
    if output_path.exists():
        return output_path
    cmd = ["qpdf", str(path), "--pages", ".", pages, "--", str(output_path)]
    subprocess.run(cmd, check=True)
    return output_path


if __name__ == "__main__":
    import fire

    fire.Fire()
