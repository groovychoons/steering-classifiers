import pandas as pd

# Hate Speech Dataset Load
df = pd.read_parquet("hf://datasets/ucberkeley-dlab/measuring-hate-speech/measuring-hate-speech.parquet")
df = df[['text', 'hate_speech_score', 'comment_id', 'annotator_id']]

# Remove duplicate comments based on 'comment_id'
df = df.drop_duplicates(subset=['text'])

# Create a new column 'label' based on the hate speech score
df['label'] = df['hate_speech_score'].apply(lambda x: 1 if x > 1 else (0 if x < -3 else None))

# Remove rows without a label
df = df.dropna(subset=['label'])

df['label'] = df['label'].astype(int)

# Print the head of the dataframe and the number of rows
print(f"Number of rows: {len(df)}")
label_counts = df['label'].value_counts()
label_percentages = df['label'].value_counts(normalize=True) * 100

print(f"Number of label 0: {label_counts[0]} ({label_percentages[0]:.2f}%)")
print(f"Number of label 1: {label_counts[1]} ({label_percentages[1]:.2f}%)")

df.to_csv("../data/hate_speech_binary.csv", index=False)
