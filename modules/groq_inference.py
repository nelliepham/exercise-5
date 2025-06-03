import os
import requests
import base64
from PIL import Image
import io


class GroqMultimodalProcessor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def encode_image(self, image_path):
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def generate_image_caption(self, img_path):
        """Generate caption for image using Groq Vision API"""
        try:
            if not img_path or not os.path.exists(img_path):
                return "No image provided"

            # Encode image to base64
            base64_image = self.encode_image(img_path)
            if not base64_image:
                return "Error encoding image"

            # Get image format
            image_format = img_path.lower().split('.')[-1]
            if image_format == 'jpg':
                image_format = 'jpeg'

            # Prepare the vision API request
            payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",  # Groq's vision model
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please describe this image in detail. What do you see?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_msg = response.text
                print(f"Vision API error: {error_msg}")

                # Fallback to basic image info if vision API fails
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        mode = img.mode
                        return f"Image uploaded: {os.path.basename(img_path)} (Size: {width}x{height}, Mode: {mode}). Vision analysis temporarily unavailable."
                except:
                    return f"Image uploaded: {os.path.basename(img_path)}. Vision analysis temporarily unavailable."

        except Exception as e:
            print(f"Error in generate_image_caption: {e}")
            return f"Error processing image: {str(e)}"

    def speech_to_text(self, audio_path):
        """Convert speech to text using Whisper"""
        try:
            if not audio_path or not os.path.exists(audio_path):
                return "No audio provided"

            print(f"Processing audio file: {audio_path}")

            url = f"{self.base_url}/audio/transcriptions"

            # Determine audio format
            audio_format = audio_path.lower().split('.')[-1]
            mime_type = f'audio/{audio_format}'
            if audio_format == 'mp3':
                mime_type = 'audio/mpeg'
            elif audio_format == 'wav':
                mime_type = 'audio/wav'
            elif audio_format == 'ogg':
                mime_type = 'audio/ogg'

            with open(audio_path, 'rb') as audio_file:
                files = {
                    'file': (os.path.basename(audio_path), audio_file, mime_type),
                    'model': (None, 'whisper-large-v3'),
                    'response_format': (None, 'json')
                }
                headers = {"Authorization": f"Bearer {self.api_key}"}

                response = requests.post(url, headers=headers, files=files)

            if response.status_code == 200:
                result = response.json()
                return result.get("text", "No text transcribed")
            else:
                print(f"Speech-to-text API error: {response.text}")
                return f"Speech-to-text error: {response.status_code}"

        except Exception as e:
            print(f"Error processing audio: {e}")
            return f"Error processing audio: {str(e)}"

    def query_groq_llm(self, prompt, model="llama-3.3-70b-versatile"):
        """Query Groq LLM"""
        try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"LLM API error: {response.text}")
                return f"LLM error: {response.status_code}"

        except Exception as e:
            print(f"Error with LLM: {e}")
            return f"Error with LLM: {str(e)}"

    def multimodal_chat(self, user_text, user_image_path, user_audio_path):
        """Main multimodal processing function"""
        context_parts = []

        # Process image if provided
        if user_image_path:
            print(f"Processing image: {user_image_path}")
            caption = self.generate_image_caption(user_image_path)
            context_parts.append(f"Image: {caption}")

        # Process audio if provided
        if user_audio_path:
            print(f"Processing audio: {user_audio_path}")
            spoken_text = self.speech_to_text(user_audio_path)
            context_parts.append(f"Audio: {spoken_text}")

        # Add user text if provided
        if user_text and user_text.strip():
            context_parts.append(f"User typed: {user_text}")

        # If no input provided
        if not context_parts:
            return "Please provide some input (text, image, or audio)!"

        # Combine into a single prompt
        combined_prompt = "\n".join(
            context_parts) + "\n\nPlease provide a helpful response based on the above information:"

        # Get LLM response
        response = self.query_groq_llm(combined_prompt)

        return response
