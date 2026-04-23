import random
from collections import defaultdict

import pandas as pd

def transitivity_augmentation(df, limit_per_type=1000):
    if 'weight' not in df.columns:
        df['weight'] = 1.0

    existing_pairs = set()
    for _, row in df.iterrows():
        u, v = int(row['id1']), int(row['id2'])
        if u > v:
            u, v = v, u
        existing_pairs.add((u, v))

    new_rows = []
    unique_labels = df['label'].unique()

    for label in unique_labels:
        if label == 0:
            continue

        sub_df = df[df['label'] == label]

        adj = defaultdict(set)
        for _, row in sub_df.iterrows():
            u, v = int(row['id1']), int(row['id2'])
            adj[u].add(v)
            adj[v].add(u)

        added_count = 0
        nodes = list(adj.keys())
        random.shuffle(nodes)

        for b in nodes:
            neighbors = list(adj[b])
            n = len(neighbors)
            if n < 2:
                continue
            if n > 100:
                neighbors = neighbors[:100]
                n = 100

            for i in range(n):
                for j in range(i + 1, n):
                    a = neighbors[i]
                    c = neighbors[j]
                    if a == c:
                        continue
                    u, v = (a, c) if a < c else (c, a)

                    if (u, v) not in existing_pairs:
                        new_rows.append([u, v, label, 0.5])
                        existing_pairs.add((u, v))
                        added_count += 1

            if added_count >= limit_per_type:
                break

    if len(new_rows) > 0:
        new_df = pd.DataFrame(new_rows, columns=['id1', 'id2', 'label', 'weight'])
        final_df = pd.concat([df, new_df], ignore_index=True)
        print(f"Transitivity augmentation: {len(df)} -> {len(final_df)} (+{len(new_rows)})")
        return final_df
    else:
        print("No new pairs found via transitivity.")
        return df
