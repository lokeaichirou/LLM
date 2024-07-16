# Reciprocal Rank Fusion¶¶

def rrf(documents, db_search_context_results_list, k=1):
        
    # Iterate db_search_results_list to generate the {"doc_{num}": {"text": , "rank":}} 
    # for each db_search_results in db_search_results_list, save them in all_search_results
    all_search_context_results = []
    
    for db_search_context_results in db_search_context_results_list:
        
        search_results = {"doc_"+str(documents.index(doc)): {"text": doc, "rank": i} 
                          for i, doc in enumerate(db_search_context_results)}

        all_search_context_results.append(search_results)

    # Iterate all_search_results and compute reciprocal rank fusion score
    ret = {}
    
    for rank in all_search_context_results:

        for id, val in rank.items():
            
            if id not in ret:
                ret[id] = {"score": 0, 
                           "text":  val["text"]}
            
            ret[id]["score"] += 1.0/(k+val["rank"])
    
    ret_sorted   = sorted(ret.items(), key=lambda item: item[1]["score"], reverse=True)
    sorted_texts = [item[1]["text"] for item in ret_sorted]

    return sorted_texts