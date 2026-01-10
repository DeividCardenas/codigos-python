import pandas as pd

chunk_size = 15000
chunks = pd.read_csv('junio.csv', chunksize=chunk_size)
for i, chunk in enumerate(chunks):
    chunk.to_csv(f'evento_part_{i+1}.csv', index=False)