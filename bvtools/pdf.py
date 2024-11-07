"""
Tools and utilities for working with PDF files.
Some require that `qpdf` is installed on the system
"""

import re
import subprocess

from dataclasses import dataclass

from betterpathlib import Path

import fitz


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

    Requires qpdf
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


def extract_pages(path: Path | str, pages0: int | str, output_dir: Path | str | None = None) -> Path:
    """
    Extract given 0-indexed page, pages, or page range from a PDF file to a new file.
    Return the Path of the extracted PDF file

    pages: e.g. '6' or '6-10'

    Requires qpdf
    """
    path = Path(path)
    pages = str(pages0)
    if not path.exists():
        raise FileNotFoundError(path)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = path.parent
    output_path = output_dir / (path.stem + f"_pages_{pages}.pdf")
    if output_path.exists():
        return output_path
    cmd = ["qpdf", str(path), "--pages", '.']
    if '-' in pages:
        cmd.extend([f"--range={pages}"])
    else:
        cmd.extend([pages])
    cmd.extend(["--", str(output_path)])
    subprocess.run(cmd, check=True)
    return output_path


def show_pdf(path: str | Path, page_number: int | str | None = None) -> None:
    """
    Requires okular
    """
    import subprocess
    args = ["okular", str(path)]
    if page_number:
        args = args + ["-p", str(page_number)]
    subprocess.Popen(args)


def show_pdf_page(page: fitz.Page, bbox: tuple[int, int, int, int] | None = None, dpi:int=200) -> None:
    """Show a page, optionally a given section within it"""
    from PIL import Image
    pix = page.get_pixmap(dpi=dpi)  # type: ignore
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if bbox:
        # Crop the table from the page
        img = img.crop(bbox)
    img.show()

@dataclass
class TesseractPageRotationResult:
    rotation_angle: int
    orientation_in_degrees: int
    orientation_confidence: float

def get_page_rotation(page: fitz.Page, dpi: int | None = 200) -> TesseractPageRotationResult:
    """Detect page rotation using Tesseract"""
    pix = page.get_pixmap(dpi=dpi)  # type: ignore
    img_bytes = pix.tobytes("png")

    tesseract_conf = ["--psm", "0"]
    cmd = [
        "tesseract",
        "stdin",
        "stdout",
    ] + tesseract_conf
    proc = subprocess.run(cmd, input=img_bytes, check=True, stdout=subprocess.PIPE)
    text = proc.stdout.decode()
    try:
        rotation_angle = int(re.search(r"Rotate: (\d+)", text).group(1))
        orientation_in_degrees = int(re.search(r"Orientation in degrees: (\d+)", text).group(1))
        orientation_confidence = float(re.search(r"Orientation confidence: (\d+\.\d+)", text).group(1))
        return TesseractPageRotationResult(rotation_angle=rotation_angle,
                                           orientation_in_degrees=orientation_in_degrees,
                                           orientation_confidence=orientation_confidence)
    except AttributeError:
        raise ValueError("Could not detect rotation angle. " + text)


def correct_page_rotation(page: fitz.Page, dpi: int | None = 200) -> int | None:
    """Correct page rotation using Tesseract

    Returns how many degrees the page was rotated, or None if it was not
    """
    rotation_angle = get_page_rotation(page, dpi=dpi).rotation_angle
    if rotation_angle:
        page.set_rotation(rotation_angle)
        return rotation_angle
    return None


if __name__ == "__main__":
    import fire

    fire.Fire()
