import re
import subprocess

from betterpathlib import Path


def split_pdf(
    input_pdf: str | Path,
    output_dir: str | Path | None = None,
    max_pages: int = 2,
    max_size_mb: int | float | None = None,
) -> list[Path]:
    """
    Split the input PDF into parts with `max_pages` pages or less. If `max_size_mb` is given, the PDF
    if split further, if possible, to adhere.

    Page PDFs are stored in the output_dir, which is inferred if not given.
    """
    input_pdf = Path(input_pdf)
    if not output_dir:
        output_dir = input_pdf.without_suffix(".pdf")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "page_%d.pdf"

    def split(input_pdf_, max_pages_):
        split_command = [
            "qpdf",
            f"--split-pages={max_pages_}",
            str(input_pdf_),
            str(output_path),
        ]
        # Split the PDF into parts with max_pages
        subprocess.run(split_command, check=True)

    split(input_pdf, max_pages)

    if max_size_mb:
        while True:
            # Split all PDFs that are larger than max_size_mb that can be split (ie has 2 or more pages)
            big_pdfs = [
                p
                for p in output_dir.glob("page_*")
                if (p.size() / (1024 * 1024) > max_size_mb)
                and re.search(r"\d+-\d+", str(p))
            ]
            if not big_pdfs:
                break
            max_pages = max_pages // 2
            for p in big_pdfs:
                split(p, max_pages)
                p.unlink()

    return list(output_dir.glob("page_*"))


def extract_pages(path: Path, pages: int | str) -> Path:
    """
    pages: e.g. '6' or '6-10'
    """
    pages = str(pages)
    out_path = path.with_stem(path.stem + f"_pages_{pages}")
    if not path.exists():
        raise FileNotFoundError
    if out_path.exists():
        return out_path
    cmd = ["qpdf", path, "--pages", ".", pages, "--", out_path]
    subprocess.run(cmd, check=True)
    return out_path


if __name__ == "__main__":
    import fire

    fire.Fire()
