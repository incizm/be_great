import pandas as pd

sample_size = 50000
chunksize = 200000

samples = []

for chunk in pd.read_csv("./data/full.csv", chunksize=chunksize):
    samples.append(chunk.sample(min(len(chunk), sample_size)))

df_sample = pd.concat(samples).sample(sample_size, random_state=42)
df_sample.to_csv("./data/random_100k.csv", index=False)