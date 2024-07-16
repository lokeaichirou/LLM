# Function to generate queries using OpenAI's ChatGPT

def generate_queries_by_llm(client, original_query, num_generated_queries, model="gpt-3.5-turbo-1106"):

    response = client.chat.completions.create(model=model,
                                              messages=[{"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
                                                        {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
                                                        {"role": "user", "content": f"OUTPUT ({num_generated_queries} queries):"}]
                                              )

    generated_queries = response.choices[0].message.content.strip().split("\n")

    return generated_queries