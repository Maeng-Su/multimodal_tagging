def get_tags(
    image_path, 
    query_text, 
    top_k=10, 
    final_k=1, 
    threshold=0.7, 
    category="text"  # "text"/ "image"
):
    if primary_by == "text":
        primary_emb = get_text_embedding(query_text)
    else:
        primary_emb = get_image_embedding(image_path)

    results = collection.query(
        query_embeddings=[primary_emb],
        n_results=top_k,
        include=["metadatas"]
    )

    if primary_by == "text" :
        rerank_emb = get_image_embedding(image_path)
    else: 
        rerank_emb = get_text_embedding(query_text)

    above_threshold = []
    below_threshold = []

    for meta in results["metadatas"][0]:
        tag_text = meta["tag"]
        tag_text_emb = get_text_embedding(tag_text)

        sim = cosine_similarity(rerank_emb, tag_text_emb)

        if sim >= threshold:
            above_threshold.append((tag_text, sim))
        else:
            below_threshold.append((tag_text, sim))

    above_threshold.sort(key=lambda x: x[1], reverse=True)
    below_threshold.sort(key=lambda x: x[1], reverse=True)

    reranked = above_threshold
    if len(reranked) < final_k:
        reranked += below_threshold[:(final_k - len(reranked))]

    return reranked[:final_k]

# get_tags(image , text, category="text")
# get_tags(image text, category="image")