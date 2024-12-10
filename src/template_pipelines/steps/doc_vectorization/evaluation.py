"""Evaluation step class."""

import json
from tempfile import NamedTemporaryFile

import numpy as np
from langchain_community.vectorstores import FAISS

from template_pipelines.utils.doc_vectorization.base_genai_step import BaseGenAIStep


class Evaluation(BaseGenAIStep):
    """Evaluation step."""

    def run(self):
        """Run evaluation."""
        self.logger.info("Running evaluation...")

        vectorstore = FAISS.load_local(
            self.inputs["faiss_vector_db"],
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )

        with open(self.inputs["aggs.json"], "rb") as f:
            aggs = json.load(f)

        size = 20
        sample_ix = np.random.randint(1, len(aggs["ids"]), size=size)
        evaluation_data = {key: [aggs[key][i] for i in sample_ix] for key in aggs}

        count = 0
        evaluation_dict = {"questions": [], "actual_chunks": [], "retrieved_chunks": []}
        for text, ix in zip(evaluation_data["texts"], evaluation_data["ids"]):
            pairs = self.qa_ge_chain.run(text)
            question = pairs[0]["question"]
            retrieved = vectorstore.similarity_search(query=question)[0].metadata["source"]
            evaluation_dict["questions"].append(pairs[0]["question"])
            evaluation_dict["actual_chunks"].append(ix)
            evaluation_dict["retrieved_chunks"].append(retrieved)
            count += ix == retrieved
        evaluation_dict["search_score"] = count / size

        with NamedTemporaryFile(mode="w") as f:
            json.dump(evaluation_dict, f)
            f.flush()
            self.outputs["evaluation.json"] = f.name

        self.logger.info(f"Search accuracy score: {count/size}")
        self.logger.info("Evaluation is done.")
