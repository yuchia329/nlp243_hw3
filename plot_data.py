from datasets import load_dataset
import matplotlib.pyplot as plt
ptb = load_dataset('ptb-text-only/ptb_text_only')
train_texts = [item["sentence"] for item in  ptb['train']]
print(len(train_texts))
sentence_lengths = [len(sentence.split()) for sentence in train_texts]

# Plot histogram
plt.hist(sentence_lengths, bins=10, edgecolor='black')
plt.title("Histogram of Sentence Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.savefig("image/sent_length.jpg")