import chromadb
from src.config import CHROMADB_STORE_DIR
from src.rag_engine import RAGEngine

col = chromadb.PersistentClient(path=str(CHROMADB_STORE_DIR)).get_collection("mv_act_knowledge_base")
print("documents in collection:", col.count())   # 23 aana chahiye (46 nahi)

for c in RAGEngine().get_context_for_violation("SPEEDING", "SEVERE"):
    print("-", c.source, "|", c.content[:90].replace("\n", " "))