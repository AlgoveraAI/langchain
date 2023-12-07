import time
import logging
import requests
from typing import Union, Generator, Callable, List, Tuple
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
from langchain.document_loaders.ipfs_file import IPFSFileDataLoader

def get_logger() -> logging.Logger:
    """
    Initialize and configure a logger for the IPFS directory reader.

    Returns:
        logger (logging.Logger): Configured logger object for the module.
    """
    logger = logging.getLogger("ipfs_directoryreader")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger

logger = get_logger()

class IPFSDirectoryDataLoader(BaseLoader):
    """
    A class to read directories and files from an IPFS server.

    Attributes:
        base_url (str): Base URL of the IPFS server.
        base_api_path (str): Base path for the IPFS API.
        version_suffix (str): Suffix for the version API endpoint.
        dag_get_suffix (str): Suffix for the DAG get API endpoint.
        max_retries (int): Maximum number of retry attempts for a request.
        retry_delay (int): Delay between retry attempts in seconds.
        debug (bool): Flag to enable or disable debug logging.

    Methods:
        __init__(debug=False): Constructor for IPFSDirectoryReader.
        check_daemon_running(): Checks if the IPFS daemon is running.
        get_dag_get(ipfs_hash): Gets the DAG (Directed Acyclic Graph) for a given IPFS hash.
        recursive_dag_get(ipfs_hash): Recursively gets all files in a directory from IPFS.
        load(ipfs_hash): Loads all files in a directory from IPFS.
    """
    base_url: str = "http://127.0.0.1:5001"
    infura_base_url: str = "https://ipfs.infura.io:5001"
    base_api_path: str = "api/v0"
    version_suffix: str = "version"
    dag_get_suffix: str = "dag/get"
    max_retries: int = 5
    retry_delay: int = 1
    
    def __init__(
        self, 
        use_infura: bool=False, 
        infura_api_key: Union[str, None]=None, 
        infura_api_secret: Union[str, None]=None, 
        debug: bool=False
    ) -> None:
        """
        Constructor for IPFSDirectoryReader.

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
                if response.status_code == 200:
                    if self.debug:
                        logger.debug(f"IPFS daemon is running at {full_url}")
                    return
                else:
                    raise Exception("IPFS daemon not running or not accessible")
            except requests.exceptions.ConnectionError:
                time.sleep(self.retry_delay)

        raise Exception("IPFS daemon not running or not accessible")

    def get_dag_get(self, ipfs_hash: str) -> requests.Response:
        """
        Gets the DAG (Directed Acyclic Graph) for a given IPFS hash.
        """
        if self.use_infura:
            full_url = f"{self.infura_base_url}/{self.base_api_path}/{self.dag_get_suffix}"
            auth = requests.auth.HTTPBasicAuth(self.infura_api_key, self.infura_api_secret)
        else:
            full_url = f"{self.base_url}/{self.base_api_path}/{self.dag_get_suffix}"

        full_url += f"?arg={ipfs_hash}"

        for i in range(self.max_retries):
            try:
                response = requests.post(full_url, auth=auth if self.use_infura else None)
                if response.status_code == 200:
                    return response
                else:
                    raise Exception(f"Failed to get DAG for {ipfs_hash} with status code {response.status_code}")
            except requests.exceptions.ConnectionError:
                time.sleep(self.retry_delay)

        raise Exception(f"Could not get DAG for {ipfs_hash}")

    def recursive_dag_get(self, ipfs_hash: str) -> List[Tuple[str, str]]:
        """
        Recursively gets all files in a directory from IPFS.

        Args:
            ipfs_hash (str): The IPFS hash of the directory.

        Returns:
            all_files (list of tuples): A list of tuples containing the IPFS hash and name of each file.
        """
        all_files = []

        def has_file_extension(name):
            """Check if a given name has a file extension."""
            return '.' in name

        def recursive_dag_get_helper(hash):
            """Helper function to perform the recursive DAG get operation."""
            response = self.get_dag_get(hash)
            if response.status_code == 200:
                dag_data = response.json()  # Parse JSON content
            else:
                raise Exception(f"Failed to get DAG for {hash} with status code {response.status_code}")

            links = dag_data.get("Links", [])

            for link in links:
                link_hash = link.get("Hash", {}).get("/")
                link_name = link.get("Name", "")
                
                # Recursively fetch links if it's a directory or a nested structure
                if link_hash:
                    recursive_dag_get_helper(link_hash)

                # If it's a file (determined by the presence of a file extension), add it to the list
                if link_hash and link_name and has_file_extension(link_name):
                    all_files.append((link_hash, link_name))

        recursive_dag_get_helper(ipfs_hash)
        return all_files
    
    def load(self, ipfs_hash: str) -> List[Document]:
        """
        Loads all files in a directory from IPFS.

        Args:
            ipfs_hash (str): The IPFS hash of the directory.

        Returns:
            results (list): A list of tuples containing the file name and its content for each file in the directory.
        """
        all_files = self.recursive_dag_get(ipfs_hash)
        fileloader = IPFSFileDataLoader()
        results = []
        for hash, name in all_files:
            try:
                # Load each file using the IPFSFileloader and append the name and content to the results list
                doc = fileloader.load(hash)
                if isinstance(doc, Document):
                    doc.metadata["source"] = name
                if isinstance(doc, list):
                    for d in doc:
                        d.metadata["source"] = name
                results.append(doc)
            except Exception as e:
                logger.error(f"Failed to load file {name} with hash {hash}: {e}")
                logger.error(f"Skipping file {name} with hash {hash}")

        return results