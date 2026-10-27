from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import numpy as np
import pandas as pd

sentences = [
    "The ancient temples of Kyoto attract many visitors.",
    "Tourists visit Kyoto to explore its historic temples.",
    "Japanese students often use trains to travel to school.",
    "Tokyo is famous for its busy streets and modern technology.",
    "A wormhole could connect two distant regions of spacetime.",
    "A theoretical tunnel through spacetime might link faraway parts of the universe.",
    "Scientists study distant galaxies to understand how the universe evolves.",
    "An artist left empty space between shapes in a drawing.",
    "The painter mixed yellow with blue to create green.",
    "She used a pencil to sketch a detailed portrait."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)


def cosine_similarity_manual(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return float(dot_product / (magnitude_a * magnitude_b))


def euclidean_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return float(np.linalg.norm(a - b))


print("Number of sentences:", len(sentences))
print("Embedding dimensions:", embeddings.shape[1])


print("\nMANUAL CALCULATION CHECK")

A = np.array([1, 2, 2], dtype=float)
B = np.array([2, 1, 2], dtype=float)
C = np.array([1, 0, 0], dtype=float)

for name, x, y in [("A and B", A, B), ("A and C", A, C)]:
    dot = np.dot(x, y)
    mag_x = np.linalg.norm(x)
    mag_y = np.linalg.norm(y)
    cosine = cosine_similarity_manual(x, y)

    print(f"\n{name}")
    print("Dot Product:", dot)
    print("Magnitude of first vector:", mag_x)
    print("Magnitude of second vector:", mag_y)
    print("Cosine Similarity:", round(cosine, 4))


print("\nZERO VECTOR CHECK")

zero_vector = np.array([0, 0, 0], dtype=float)

print(
    "Cosine with zero vector:",
    cosine_similarity_manual(A, zero_vector)
)


print("\nCOMPARISON WITH W6D3")

predicted_pairs = [
    (0, 1),
    (4, 5),
    (7, 8),
    (2, 3),
    (5, 9)
]

for i, j in predicted_pairs:
    manual_score = cosine_similarity_manual(
        embeddings[i], embeddings[j]
    )

    sklearn_score = float(
        cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[j].reshape(1, -1)
        )[0][0]
    )

    print(
        f"Sentences {i + 1} and {j + 1}: "
        f"Manual = {manual_score:.6f}, "
        f"Sklearn = {sklearn_score:.6f}, "
        f"Match = {np.isclose(manual_score, sklearn_score, atol=1e-6)}"
    )


print("\nALL SENTENCE PAIRS")

results = []

for i, j in combinations(range(len(sentences)), 2):
    results.append({
        "Sentence 1 Number": i + 1,
        "Sentence 2 Number": j + 1,
        "Sentence 1": sentences[i],
        "Sentence 2": sentences[j],
        "Cosine Similarity": cosine_similarity_manual(
            embeddings[i], embeddings[j]
        ),
        "Euclidean Distance": euclidean_distance(
            embeddings[i], embeddings[j]
        )
    })

df = pd.DataFrame(results)

df["Cosine Similarity"] = df["Cosine Similarity"].round(4)
df["Euclidean Distance"] = df["Euclidean Distance"].round(4)


print("\nTOP 5 PAIRS BY COSINE SIMILARITY")

print(
    df.sort_values(
        "Cosine Similarity", ascending=False
    ).head(5).to_string(index=False)
)


print("\nTOP 5 CLOSEST PAIRS BY EUCLIDEAN DISTANCE")

print(
    df.sort_values(
        "Euclidean Distance", ascending=True
    ).head(5).to_string(index=False)
)


cosine_order = df.sort_values(
    "Cosine Similarity", ascending=False
)[["Sentence 1 Number", "Sentence 2 Number"]].values.tolist()

distance_order = df.sort_values(
    "Euclidean Distance", ascending=True
)[["Sentence 1 Number", "Sentence 2 Number"]].values.tolist()

print("\nDo the complete rankings match?", cosine_order == distance_order)


print("\nVECTOR MAGNITUDE EXPERIMENT")

v1 = embeddings[0]
v2 = embeddings[1]

original_cosine = cosine_similarity_manual(v1, v2)
original_distance = euclidean_distance(v1, v2)

scaled_v1 = v1 * 10

scaled_cosine = cosine_similarity_manual(scaled_v1, v2)
scaled_distance = euclidean_distance(scaled_v1, v2)

print("Original cosine:", round(original_cosine, 6))
print("Cosine after scaling:", round(scaled_cosine, 6))

print("Original Euclidean distance:", round(original_distance, 6))
print("Distance after scaling:", round(scaled_distance, 6))


print("\nMINI SEMANTIC SEARCH")

query = "How might a wormhole connect distant places in space?"

query_embedding = model.encode([query])[0]

search_results = []

for i, sentence_embedding in enumerate(embeddings):
    score = cosine_similarity_manual(
        query_embedding, sentence_embedding
    )

    search_results.append({
        "Sentence Number": i + 1,
        "Sentence": sentences[i],
        "Cosine Similarity": round(score, 4)
    })

search_df = pd.DataFrame(search_results)

search_df = search_df.sort_values(
    "Cosine Similarity", ascending=False
)

print("Query:", query)

print("\nTOP 3 CLOSEST SENTENCES")

print(search_df.head(3).to_string(index=False))


df.to_csv("W6D4_all_pair_results.csv", index=False)

search_df.to_csv("W6D4_mini_search_results.csv", index=False)

print("\nResults saved successfully.")