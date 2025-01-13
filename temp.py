import os
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
from PIL import Image

def extract_first_three_pages(input_pdf_path, output_pdf_path):
    """
    Extracts the first three pages from a PDF file and saves them as a new PDF.

    Args:
        input_pdf_path (str): Path to the input PDF file.
        output_pdf_path (str): Path to save the output PDF file.
    """
    with open(input_pdf_path, 'rb') as input_pdf:
        reader = PdfReader(input_pdf)
        writer = PdfWriter()

        # Extract up to the first three pages
        num_pages = min(3, len(reader.pages))
        for page_num in range(num_pages):
            writer.add_page(reader.pages[page_num])
        
        with open(output_pdf_path, 'wb') as output_pdf:
            writer.write(output_pdf)
            print(f"Created new PDF with first three pages: {output_pdf_path}")

def convert_pdf_to_images(pdf_path, output_folder, first_n_pages=3):
    """
    Converts the first N pages of a PDF to images and saves them.

    Args:
        pdf_path (str): The path to the PDF file.
        output_folder (str): The folder to save the images.
        first_n_pages (int): The number of pages to convert.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images = convert_from_path(pdf_path, first_page=1, last_page=first_n_pages, thread_count=3)

    for i, image in enumerate(images):
        resized_image = image.resize((image.width // 2, image.height // 2))  # Resize the image
        output_path = os.path.join(output_folder, f"page_{i + 1}.jpg")
        resized_image.save(output_path, "JPEG")
        print(f"Saved image: {output_path}")

def process_pdfs(input_folder, output_folder):
    """
    Processes all PDFs in a folder:
    - Extracts the first three pages from each PDF.
    - Converts the extracted pages to images.

    Args:
        input_folder (str): Folder containing the input PDFs.
        output_folder (str): Folder to save the processed images.
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist.")
    
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the input folder.")
        return
    
    for pdf_file in pdf_files:
        input_pdf_path = os.path.join(input_folder, pdf_file)
        cropped_pdf_folder = os.path.join(output_folder, "cropped_pdfs")
        cropped_pdf_path = os.path.join(cropped_pdf_folder, f"cropped_{pdf_file}")
        
        # Create folder for cropped PDFs if it doesn't exist
        if not os.path.exists(cropped_pdf_folder):
            os.makedirs(cropped_pdf_folder)
        
        # Extract first three pages
        extract_first_three_pages(input_pdf_path, cropped_pdf_path)
        
        # Convert the cropped PDF to images
        images_folder = os.path.join(output_folder, "images", os.path.splitext(pdf_file)[0])
        convert_pdf_to_images(cropped_pdf_path, images_folder)

if __name__ == "__main__":
    input_dir = "./sample_data"  # Folder containing input PDFs
    output_dir = "./sample_data_images"  # Folder to save cropped PDFs and images

    process_pdfs(input_dir, output_dir)
