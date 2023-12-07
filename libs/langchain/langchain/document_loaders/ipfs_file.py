import json
import time
import logging
import requests
import tempfile
import pandas as pd
from typing import Union, Generator, Callable
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

def identify_file_type_from_bytes(byte_data: bytes) -> str:
    """
    Identify the file type of a given byte stream.

    Args:
        byte_data (bytes): Byte stream to identify the file type of.

    Returns:
        file_type (str): Identified file type of the byte stream.
    """
    try:
        import magic
    except ImportError:
        raise ImportError("The identify_file_type_from_bytes function requires the python-magic library to be installed. Please install it using pip install python-magic.")
    mime = magic.Magic(mime=True)
    file_type = mime.from_buffer(byte_data)
    return file_type

# Reader functions for different file types.
# Each function takes a response object and returns
# the appropriate data format (text, DataFrame, JSON, image).

def pdf_reader(response : requests.Response) -> str:
    """
    Read and extract text from a PDF file contained in a response object.

    Args:
        response (requests.Response): Response object containing PDF content.

    Returns:
        text (str): Extracted text from the PDF.
    """
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("The pdf_reader function requires the PyPDF2 library to be installed. Please install it using pip install PyPDF2.")
    
    with tempfile.TemporaryFile() as temp:
        temp.write(response.content)
        pdf = PyPDF2.PdfReader(temp)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"

        return text
    
def txt_reader(response : requests.Response) -> str:
    """
    Read text content from a response object.

    Args:
        response (requests.Response): Response object containing text content.

    Returns:
        text (str): Decoded text content.
    """
    with tempfile.TemporaryFile() as temp:
        temp.write(response.content)
        temp.seek(0)
        text = temp.read().decode("utf-8")
        return text

def lazy_csv_loader(df : pd.DataFrame) -> Generator[Document, None, None]:
    for index, row in df.iterrows():
        content = ""
        for k, v in row.items():
            # only string values which are not null and string
            if isinstance(v, str) and v.strip() != "":
                content += f"{k.strip()}: {v.strip()}\n"
            else:
                content += f"{k}: {v}\n"
        yield Document(
            page_content=content,
            metadata={
                "index": index,
                "file_type": "text/csv",
            },
        )

def csv_reader(response : requests.Response) -> list[Document]:
    """
    Read CSV content from a response object and return it as a pandas DataFrame.

    Args:
        response (requests.Response): Response object containing CSV content.

    Returns:
        df (pandas.DataFrame): DataFrame containing the CSV data.
    """
    with tempfile.TemporaryFile() as temp:
        temp.write(response.content)
        temp.seek(0)
        df = pd.read_csv(temp)
    
    df = [doc for doc in lazy_csv_loader(df)]

    return df

    
def json_reader(response : requests.Response) -> str:
    """
    Read JSON content from a response object and return it as a dictionary.

    Args:
        response (requests.Response): Response object containing JSON content.

    Returns:
        data (dict): Dictionary containing the JSON data.
    """
    with tempfile.TemporaryFile() as temp:
        temp.write(response.content)
        temp.seek(0)
        data = json.load(temp)
        # JSON to string
        data = json.dumps(data)
        return data

def select_reader(file_type : str) -> Callable:
    """
    Select the appropriate reader function based on the file type.

    Args:
        file_type (str): File type to select the reader for.

    Returns:
        reader (function): Appropriate reader function for the file type.

    Raises:
        Exception: If the file type is unsupported.
    """
    if "pdf" in file_type:
        return pdf_reader
    elif "csv" in file_type:
        return csv_reader
    elif "json" in file_type:
        return json_reader
    # elif "image" in file_type:
        # return image_reader
    elif "text" in file_type:
        return txt_reader
    else:
        raise Exception(f"Unsupported file type: {file_type}")


class IPFSFileDataLoader(BaseLoader):
    base_url: str = "http://127.0.0.1:5001"
    infura_base_url: str = "https://ipfs.infura.io:5001"
    base_api_path: str = "api/v0"
    read_suffix: str = "cat"
    version_suffix: str = "version"
    max_retries: int = 5
    retry_delay: int = 1
    def __init__(self, 
        use_infura: bool =False, 
        infura_api_key: Union[str, None]=None, 
        infura_api_secret: Union[str, None]=None, 
        debug: bool=False
    ) -> None:
        """
        Constructor for IPFSFileloader.

        Args:
            use_infura (bool, optional): Flag to use Infura instead of local IPFS. Defaults to False.
            infura_api_key (str, optional): Infura API key. Required if use_infura is True.
            infura_api_secret (str, optional): Infura API secret. Required if use_infura is True.
            debug (bool, optional): Enable or disable debug logging. Defaults to False.
        """
        self.use_infura = use_infura
        self.infura_api_key = infura_api_key
        self.infura_api_secret = infura_api_secret
        self.debug = debug

        if self.use_infura and (not self.infura_api_key or not self.infura_api_secret):
            raise ValueError("Infura API key and secret are required when using Infura")

        self.check_daemon_running()

    def check_daemon_running(self) -> None:
        """
        Checks if the IPFS daemon is running and accessible, or if the Infura API is reachable.
        """
        if self.use_infura:
            full_url = f"{self.infura_base_url}/{self.base_api_path}/{self.version_suffix}"
            auth = requests.auth.HTTPBasicAuth(self.infura_api_key, self.infura_api_secret)
        else:
            full_url = f"{self.base_url}/{self.base_api_path}/{self.version_suffix}"

        for i in range(self.max_retries):
            try:
                response = requests.post(full_url, auth=auth if self.use_infura else None)
                if response.ok:
                    if self.debug:
                        logger.debug(f"IPFS daemon is running at {full_url}")
                    return
                else:
                    raise Exception("IPFS daemon not running or not accessible")
            except requests.exceptions.ConnectionError:
                time.sleep(self.retry_delay)

        raise Exception("IPFS daemon not running or not accessible")

    def load(
        self, 
        ipfs_hash : str
        ) -> Union[Document, list[Document]]:
        """
        Loads a file from IPFS using its hash, either from local IPFS or Infura.
        """
        if self.use_infura:
            full_url = f"{self.infura_base_url}/{self.base_api_path}/{self.read_suffix}"
            auth = requests.auth.HTTPBasicAuth(self.infura_api_key, self.infura_api_secret)
        else:
            full_url = f"{self.base_url}/{self.base_api_path}/{self.read_suffix}"

        full_url += f"?arg={ipfs_hash}"

        for i in range(self.max_retries):
            try:
                response = requests.post(full_url, auth=auth if self.use_infura else None)
                if response.ok:
                    file_type = identify_file_type_from_bytes(response.content)
                    reader = select_reader(file_type)
                    text = reader(response)
                    # if isinstance() is string
                    if isinstance(text, str):
                        return Document(page_content=text, metadata={"ipfs_hash": ipfs_hash, "file_type": file_type})
                    if isinstance(text, list):
                        # add ipfs_hash to metadata
                        for doc in text:
                            doc.metadata["ipfs_hash"] = ipfs_hash
                        return text
                else:
                    logger.error(f"Failed to load file from IPFS with hash {ipfs_hash}")
            except requests.exceptions.ConnectionError:
                time.sleep(self.retry_delay)

        raise Exception(f"Failed to connect to IPFS daemon for hash {ipfs_hash}")
