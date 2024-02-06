from src.vanna.mistral.mistral import Mistral
from src.vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

class MyVanna(ChromaDB_VectorStore, Mistral):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config)
        Mistral.__init__(self, config={'weights_path': '/home/ventisk1ze/llms/mistral-7b-v0.1.Q5_K_M.gguf'})

vn = MyVanna()

vn.connect_to_sqlite('Chinook.sqlite')

df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")

for ddl in df_ddl['sql'].to_list():
  vn.train(ddl=ddl)


vn.ask(question="What are the top 5 artists by sales?")