import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def image_qa(client, query, image_path):
    
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(model       = "gpt-4o",
                                              temperature = 0,
                                              seed        = 42,
                                              messages    = [{"role": "user",
                                                              "content": [{"type": "text", "text": query},
                                                                          {"type": "image_url",
                                                                           "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",},
                                                                           },
                                                                          ],
                                                               } 
                                                              ],
                                               )

    if response and response.choices:
        if response.choices[0]:
            return response.choices[0].message.content
    
    return None


def generate_image_description(client, image_file_paths, corpus):

    images_descriptions = []
    for image_path in image_file_paths:
        image_description = image_qa(client, "Please briefly describe the information in the image", image_path)
        if image_description:
            images_descriptions.append(image_description)

    corpus.extend(images_descriptions)

    return images_descriptions