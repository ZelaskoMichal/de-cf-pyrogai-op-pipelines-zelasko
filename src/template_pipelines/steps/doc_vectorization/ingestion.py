"""Ingestion step class."""

import asyncio
import json
import time
from tempfile import NamedTemporaryFile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from aif.pyrogai.steps.step import Step


class Ingestion(Step):
    """Ingestion step."""

    async def load_document_async(self, path):
        """Asynchronously load a document from the given file path."""
        loader = UnstructuredMarkdownLoader(path)
        content = loader.load()
        return content

    async def run_tasks(self, tasks):
        """Asynchronously run a list of tasks and yield results as they complete."""
        if len(tasks) > 0:
            completed_results = await asyncio.gather(*tasks)
            for result in completed_results:
                if result is None:
                    continue
                for doc in result:
                    yield doc
        del tasks[:]

    async def load(self, tasks):
        """Asynchronously process tasks to split documents into text chunks and aggregate metadata."""
        chunk_params = {"chunk_size": 900, "chunk_overlap": 90}
        text_splitter = RecursiveCharacterTextSplitter(**chunk_params)
        aggs = {"metadatas": [], "texts": [], "ids": []}
        async for doc in self.run_tasks(tasks):
            chunks = text_splitter.split_text(doc.page_content)

            for ic, chunk in enumerate(chunks):
                imetadata = {key: doc.metadata.get(key, None) for key in doc.metadata}
                tid = f"{doc.metadata['source']}/{ic}/pl"
                imetadata["source"] = tid

                aggs["metadatas"].append(imetadata)
                aggs["texts"].append(chunk)
                aggs["ids"].append(tid)

        return aggs

    def run(self):
        """Run Ingestion."""
        self.logger.info("Running ingestion...")

        local_tmp = self.inputs["doc_input_dir"]
        paths = [f for f in local_tmp.glob("**/*.md")]

        start = time.time()
        tasks = [self.load_document_async(path) for path in paths]

        aggs = asyncio.run(self.load(tasks))

        self.logger.info(aggs)
        self.logger.info(time.time() - start)

        with NamedTemporaryFile(mode="w") as f:
            json.dump(aggs, f)
            f.flush()
            self.outputs["aggs.json"] = f.name

        self.logger.info("Ingestion is done.")
