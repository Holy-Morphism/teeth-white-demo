# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import mimetypes
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

model = "gemini-2.0-flash-preview-image-generation"


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")



def generate_images(image_bytes):
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=image_bytes  # Use the raw bytes directly
                ),
                types.Part.from_text(text="""This is an image of a person's teeth. Please generate the 4 examples of the same image but in each the teeth should be more brighter. Forexample the first image could have teeth 5/4 brightness, the second 6/4, the third 7/4 finally 8/4 for the last one which would double the brightness of the original."""),
            ],
        ),
      
    ]

    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
        response_mime_type="text/plain",
    )

    image_count = 0

    image_file_paths = []

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue


        if chunk.candidates[0].content.parts[0].inline_data:
            file_name = "image" + str(image_count)
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            file_path = f"{file_name}{file_extension}"
            image_file_paths.append(file_path)
            save_binary_file(file_path, data_buffer)
       
        image_count = image_count + 1

    return image_file_paths