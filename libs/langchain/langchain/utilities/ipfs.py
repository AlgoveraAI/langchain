import os
import json
import time
import faiss
import shutil
import pickle
import logging
import tempfile
import requests
from typing import Union
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

logger = logging.getLogger(__name__)

class IPFSFAISSUtils:
    base_url: str = "http://127.0.0.1:5001"
    infura_base_url: str = "https://ipfs.infura.io:5001"
    base_api_path: str = "api/v0"
    read_suffix: str = "cat"
    write_suffix: str = "add"
    dag_get_suffix: str = "dag/get"
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
    
    def send_vectorstore_to_ipfs(self, vs: FAISS) -> str:
        index = vs.index
        index_suffix = '.index'
        docstore_suffix = '.docstore'

        # Named temporary file for index
        with tempfile.NamedTemporaryFile(delete=False, suffix=index_suffix) as tmp_index:
            # Save the index to the temporary file
            index_filename = tmp_index.name
            faiss.write_index(index, index_filename)

        # Named temporary file for docstore
        with tempfile.NamedTemporaryFile(delete=False, suffix=docstore_suffix) as tmp_docstore_file:
            stores = {
                'index_to_docstore_id': vs.index_to_docstore_id,
                'docstore': vs.docstore
            }
            pickle.dump(stores, tmp_docstore_file)

        # Prepare files for sending
        files = {
            'index': open(index_filename, 'rb'),
            'docstore': open(tmp_docstore_file.name, 'rb')
        }

        # Send files to IPFS
        if self.use_infura:
            url = f"{self.infura_base_url}/{self.base_api_path}/{self.write_suffix}?wrap-with-directory=true&pin=true"
            auth = requests.auth.HTTPBasicAuth(self.infura_api_key, self.infura_api_secret)
        else:
            url = f"{self.base_url}/{self.base_api_path}/{self.write_suffix}?wrap-with-directory=true&pin=true"
            auth = None

        response = requests.post(url, auth=auth, files=files)

        # Close and remove temporary files
        files['index'].close()
        files['docstore'].close()
        os.remove(index_filename)
        os.remove(tmp_docstore_file.name)

        json_objects = [json.loads(line) for line in response.content.decode().split('\n') if line]

        return json_objects[-1]['Hash']
    
    def get_vectorstore_from_ipfs(self, cid: str) -> FAISS:
        if self.use_infura:
            url = f"{self.infura_base_url}/{self.base_api_path}/{self.dag_get_suffix}?arg={cid}"
            auth = requests.auth.HTTPBasicAuth(self.infura_api_key, self.infura_api_secret)
        else:
            url = f"{self.base_url}/{self.base_api_path}/{self.dag_get_suffix}?arg={cid}"
            auth = None
        response = requests.post(url, auth=auth)
        links = response.json()['Links']

        # Make sure only two links are returned
        assert len(links) == 2

        for link in links:
            if 'index' in link['Name']:
                index_hash = link['Hash']['/']
            if 'docstore' in link['Name']:
                docstore_hash = link['Hash']['/']
        
        # Get index and docstore from IPFS
        if self.use_infura:
            index_url = f"{self.infura_base_url}/{self.base_api_path}/{self.read_suffix}?arg={index_hash}"
            docstore_url = f"{self.infura_base_url}/{self.base_api_path}/{self.read_suffix}?arg={docstore_hash}"
            auth = requests.auth.HTTPBasicAuth(self.infura_api_key, self.infura_api_secret)
        else:
            index_url = f"{self.base_url}/{self.base_api_path}/{self.read_suffix}?arg={index_hash}"
            docstore_url = f"{self.base_url}/{self.base_api_path}/{self.read_suffix}?arg={docstore_hash}"
            auth = None

        index_response = requests.post(index_url, auth=auth)
        docstore_response = requests.post(docstore_url, auth=auth)

        # Save the index to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as tmp_index:
            tmp_index.write(index_response.content)
            tmp_index_path = tmp_index.name

        embeddings = OpenAIEmbeddings()

        # Read index directly from the saved file
        index = faiss.read_index(tmp_index_path)

        # Load docstore data
        docstore_data = pickle.loads(docstore_response.content)
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            index_to_docstore_id=docstore_data['index_to_docstore_id'],
            docstore=docstore_data['docstore']
        )

        return vectorstore
