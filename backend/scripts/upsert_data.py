import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

api_key = "pcsk_6jvZ6E_2YE2SQEgsa8pvMPUR2jbzq7w49Bnk33XXYRp7bqnq9u3mNPVHCxRkLVvXoXBkS1"
pc = Pinecone(api_key=api_key)

# pc.create_index(
#     name="quran-data",
#     dimension=768,
#     metric="cosine",
#     spec=ServerlessSpec(    
#         cloud="aws",
#         region="us-east-1"
#     )
# )

# file_path = "../datasets/quran eng tafseer/main_df.csv"

# df = pd.read_csv(file_path, usecols=['Name', 'Surah', 'Ayat', 'Translation1', 'Tafaseer2',], nrows=8)

# print(df.columns)

model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base',device='cpu')
# print(model)

# df['values'] = df['Tafaseer2'].map(
#     lambda x: (model.encode(x)).tolist())

# df['id'] = df.index

# df['metadata'] = df.apply(lambda x:{
#     'Name': x['Name'],
#     'Surah': x['Surah'],
#     'Ayat': x['Ayat'],
#     'Translation': x['Translation1'],
#     'Tafaseer': x['Tafaseer2'],
# }, axis=1)

# df_upsert = df[['id','values','metadata']]
# df_upsert['id'] = df_upsert['id'].map(lambda x: str(x))
# print(df_upsert)

idx = pc.Index('quran-data')

# idx.upsert_from_dataframe(df_upsert)

# querying
xc = idx.query(vector=model.encode("What is the meaning of Amen?").tolist(), top_k=3, include_metadata=True, include_values=True)

for result in xc['matches']:
    print(f"{round(result['score'],2)}: {result['metadata']['Tafaseer']}")
