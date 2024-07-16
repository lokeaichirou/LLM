import time

def get_embeddings(client, texts, model="text-embedding-ada-002", dimensions=None, delay=1):

    '''Encapsulate OpenAI's Embedding model interface'''

    embeddings = []

    if model == "text-embedding-ada-002":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    else:
        data = client.embeddings.create(input=texts, model=model).data

    embeddings.append(data[0].embedding)
    time.sleep(delay)  # Wait for 'delay' seconds before the next request

    return [x.embedding for x in data]
