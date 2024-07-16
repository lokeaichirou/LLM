
def get_completion(client, prompt, model="gpt-3.5-turbo-1106"):

    '''Encapsulate openai interface'''

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model = model,
                                              messages = messages,
                                              temperature = 0,  # The randomness of the model output, 0 means the least randomness
                                              )
    return response.choices[0].message.content