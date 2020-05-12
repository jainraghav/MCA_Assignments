import numpy as np
from scipy.sparse import *
from sklearn.metrics.pairwise import cosine_similarity

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    rf_sim = sim
    newqs = lil_matrix( (vec_queries.shape[0],vec_queries.shape[1]))
    iters = 3
    for it in range(iters):
        relevant=[]
        nonrelevant = []
        for i in range(vec_queries.shape[0]):
            rel = np.argsort(-1*rf_sim[:, i])[:n]
            relevant.append(rel)
            nonrel = np.argsort(rf_sim[:, i])[:vec_docs.shape[0]-n]
            nonrelevant.append(nonrel)

        for q in range(vec_queries.shape[0]):
            
            Cr = csr_matrix( (1,vec_docs.shape[1]))
            Cnr = csr_matrix( (1,vec_docs.shape[1]))
            
            for i in relevant[q]:
                Cr += vec_docs[i]
            for i in nonrelevant[q]:
                Cnr += vec_docs[i]

            Cr /= n
            Cnr /= vec_docs.shape[0]-n

            qnew = Cr - Cnr
            newqs[q] = qnew

        rf_sim = cosine_similarity(vec_docs, newqs)
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    newqs = lil_matrix( (vec_queries.shape[0],vec_queries.shape[1]))
    
    rf_sim = relevance_feedback(vec_docs, vec_queries, sim, n)
    

    iters = 3
    for it in range(iters):
        relevant=[]
        for i in range(vec_queries.shape[0]):
            rel = np.argsort(-1*rf_sim[:, i])[:n]
            relevant.append(rel)

        for q in range(vec_queries.shape[0]):
            
            Cr = csr_matrix( (1,vec_docs.shape[1]))
            for i in relevant[q]:
                Cr += vec_docs[i]

            newqs[q] = vec_queries[q] + Cr

        rf_sim = cosine_similarity(vec_docs, newqs)
    # feature_array = np.array(tfidf_model.get_feature_names())
    # for q in range(vec_queries.shape[0]):
    #     rel = np.argsort(-1*rf_sim[:, q])[:n]
    #     top_n = []
    #     for ind in rel:
    #         tfidf_sorting = np.argsort(vec_docs[ind].toarray()).flatten()[::-1]
    #         top_n = top_n + feature_array[tfidf_sorting][:n].tolist()
    #     prev_q = tfidf_model.inverse_transform(vec_queries[q])
    #     # print("++++")
    #     # print(prev_q)
    #     collectq=[]
    #     for i in range(len(prev_q[0])):
    #         collectq.append(prev_q[0][i])
    #     for j in range(len(top_n)):
    #         collectq.append(top_n[j])
    #     # print(collectq)
    #     newqs[q] = tfidf_model.transform([' '.join(collectq)])
    
    return rf_sim