import os
import logging
from typing import Optional

import openpyxl
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredPDFLoader

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError("fitz module is required for reading PDF files. Install it using 'pip install PyMuPDF'") from e


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def read_file(file_path: str) -> str:
    """
    Reads the content of a file based on its extension.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The content of the file or an error message if the file type is unsupported.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    logger.info(f"Reading file: {file_path} with extension: {file_extension}")

    if file_extension == '.pdf':
        return read_pdf(file_path)
    elif file_extension in ['.doc', '.docx']:
        return read_doc(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        return read_excel(file_path)
    elif file_extension == '.txt':
        return read_txt(file_path)
    else:
        logger.error(f"Unsupported file type: {file_extension}")
        return "Unsupported file type"


def read_pdf(pdf_path: str) -> str:
    """
    Reads and extracts text from a PDF file using PyMuPDF.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: Extracted text from the PDF or an empty string if an error occurs.
    """
    logger.info(f"Opening PDF file: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        logger.info(f"PDF successfully opened: {pdf_path}")
        logger.info(f"Number of pages in PDF: {len(doc)}")

        all_text = []

        for page_num in range(len(doc)):
            logger.debug(f"Processing page {page_num + 1}...")
            page = doc.load_page(page_num)
            page_text = page.get_text()

            if not page_text.strip():
                logger.warning(f"Page {page_num + 1} has no text.")

            all_text.append(page_text)

        doc.close()
        logger.info("PDF processing complete.")

        return "\n".join(all_text)

    except Exception as e:
        logger.error(f"An error occurred while reading PDF file '{pdf_path}': {e}")
        return ""


def read_doc(doc_path: str) -> str:
    """
    Reads and extracts text from a DOC or DOCX file using UnstructuredWordDocumentLoader.

    Args:
        doc_path (str): The path to the DOC/DOCX file.

    Returns:
        str: Extracted text from the document or an error message if an error occurs.
    """
    logger.info(f"Reading DOC/DOCX file: {doc_path}")
    try:
        loader = UnstructuredWordDocumentLoader(doc_path)
        docs = loader.load()
        if docs:
            logger.info(f"Successfully read DOC/DOCX file: {doc_path}")
            return docs[0].page_content
        else:
            logger.warning(f"No content found in DOC/DOCX file: {doc_path}")
            return ""
    except Exception as e:
        logger.error(f"Error reading DOC/DOCX file '{doc_path}': {e}")
        return f"Error reading DOC file: {str(e)}"


def read_excel(excel_path: str) -> str:
    """
    Reads and extracts text from an Excel file using openpyxl.

    Args:
        excel_path (str): The path to the Excel file.

    Returns:
        str: Extracted text from the Excel file or an error message if an error occurs.
    """
    logger.info(f"Reading Excel file: {excel_path}")
    try:
        wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
        sheet = wb.active
        data = []

        for row in sheet.iter_rows(values_only=True):
            row_data = "\t".join([str(cell) if cell is not None else '' for cell in row])
            data.append(row_data)

        logger.info(f"Successfully read Excel file: {excel_path}")
        return "\n".join(data)

    except Exception as e:
        logger.error(f"Error reading Excel file '{excel_path}': {e}")
        return f"Error reading Excel file: {str(e)}"


def read_txt(txt_path: str) -> str:
    """
    Reads and extracts text from a TXT file.

    Args:
        txt_path (str): The path to the TXT file.

    Returns:
        str: The content of the TXT file or an error message if an error occurs.
    """
    logger.info(f"Reading TXT file: {txt_path}")
    try:
        with open(txt_path, 'r', encoding='utf-8') as txt_file:
            content = txt_file.read().strip()
            logger.info(f"Successfully read TXT file: {txt_path}")
            return content
    except Exception as e:
        logger.error(f"Error reading TXT file '{txt_path}': {e}")
        return f"Error reading TXT file: {str(e)}"