# Movie-Recommendation-with-Two-Stage-Retrieval
This project was executed through three distinct deliverables focused on high-dimensional data retrieval and model optimization:

**Deliverable 1**: Foundation & Vector Indexing. I implemented Locality-Sensitive Hashing (LSH) or HNSW algorithms to compute similar vector buckets. This established the structural foundation for efficient similarity searching within high-dimensional datasets.

**Deliverable 2**: Comparative LLM Tuning. I optimized three separate Large Language Models (LLMs) through a two-step tuning process. I first tuned the models on raw datasets and then incorporated the information derived from the computed vector buckets to analyze the computational differences and improvements in recommendation quality. This stage included the development of detailed pseudocode and editable vector diagrams to map the data flow.

**Deliverable 3**: RAG Implementation. I applied a Retrieval-Augmented Generation (RAG) pipeline to ground the tuned models. This ensures that the final recommendation process is dynamically informed by retrieved data, minimizing hallucinations and maximizing the precision of the system's output.
