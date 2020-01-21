Top_adjs looks at what descriptors music reviewers tend to use when writing about different kinds of music by calculating tf-idf scores for the adjectives in 18393 Pitchfork reviews spanning ten genres.

After cleaning the review text using SpaCy and NLTK (tokenizing, lemmatizing, removing punctuation, numbers), it keeps only those tokens labeled as adjectives. The resulting representations of the review documents are input to gensim's tf-idf model with the top-scoring tokens per genre as output.
