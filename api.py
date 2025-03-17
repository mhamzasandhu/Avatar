# import os
# import fitz  # PyMuPDF for PDF text extraction
# import subprocess
# import soundfile as sf
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from ttsmms import TTS
# from dotenv import load_dotenv
# from groq import Groq

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("GROQ_API_KEY")

# if not api_key:
#     raise ValueError("Error: GROQ_API_KEY is not set in .env")

# # Initialize Groq client
# client = Groq(api_key=api_key)

# app = FastAPI()

# # Paths for Lip-Syncing
# FACE_VIDEO_PATH = "/home/hamza/Desktop/translater/Wav2Lip/kashif_demo.mp4"
# CHECKPOINT_PATH = "/home/hamza/Desktop/translater/Wav2Lip/checkpoints/wav2lip_gan.pth"
# OUTPUT_VIDEO_PATH = "/home/hamza/Desktop/translater/Wav2Lip/results/output.mp4"

# # Language Configuration
# LANGUAGE_MAP = {
#     "eng_Latn": "english",
#     "urd_Arab": "urdu",
#     "arb_Arab": "arabic",
# }

# LANGUAGE_DIRS = {
#     "english": "/home/hamza/Desktop/translater/data/eng",
#     "urdu": "/home/hamza/Desktop/translater/data/urd-script_arabic",
#     "arabic": "/home/hamza/Desktop/translater/data/ara",
# }

# # Load Translation Model
# def load_translator(model_name="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="eng_Latn"):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang, max_length=500)

#     def translate(text):
#         """Translate text in chunks to avoid truncation."""
#         words = text.split()
#         chunks = [' '.join(words[i:i + 250]) for i in range(0, len(words), 250)]
#         return ' '.join([translator(chunk, truncation=True)[0]['translation_text'] for chunk in chunks])
    
#     return translate

# # Load Text-to-Speech (TTS) Model
# def load_tts_model(language):
#     mapped_language = LANGUAGE_MAP.get(language)
#     model_path = LANGUAGE_DIRS.get(mapped_language)
#     if not model_path or not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model directory '{model_path}' not found for language '{mapped_language}'.")
#     return TTS(model_path)

# # Convert Text to Speech
# def text_to_speech(text, language, output_file="output.wav"):
#     tts = load_tts_model(language)
#     wav = tts.synthesis(text)
#     if "x" in wav and "sampling_rate" in wav:
#         sf.write(output_file, wav["x"], wav["sampling_rate"])
#         return output_file
#     else:
#         raise RuntimeError("Speech synthesis failed.")

# # Extract text from PDF or TXT
# def extract_text(file):
#     """Extracts text from an uploaded PDF or TXT file."""
#     file_extension = os.path.splitext(file.filename)[-1].lower()

#     if file_extension == ".pdf":
#         text = ""
#         try:
#             doc = fitz.open(stream=file.file.read(), filetype="pdf")
#             for page in doc:
#                 text += page.get_text("text") + "\n"
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {e}")
#         return text.strip()

#     elif file_extension == ".txt":
#         try:
#             return file.file.read().decode("utf-8").strip()
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Error reading text file: {e}")

#     else:
#         raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a PDF or TXT file.")

# # Summarize extracted text
# def summarize_text(text):
#     """Summarizes text using Groq API with the Mistral model."""
#     response = client.chat.completions.create(
#         model="mixtral-8x7b-32768",
#         messages=[
#             {"role": "system",
#                             "content": (
#                                 "You are an expert AI assistant specializing in generating structured, topic-wise summaries. "
#                                 "Present the key information in a clear, logical flow as if teaching a student. "
#                                 "Use a conversational yet informative tone, making complex ideas easy to grasp. "
#                                 "Break the summary into meaningful sections with soft headings, ensuring coherence and readability."
#                             )
# },
#             {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
#         ]
#     )
#     return response.choices[0].message.content

# # Generate Lip-Synced Video
# def generate_lip_synced_video(input_text, src_lang, tgt_lang, face_video_path, checkpoint_path, output_video_path):
#     """Converts text to speech, then generates a lip-synced video."""
#     audio_path = text_to_speech(input_text, tgt_lang, output_file="output.wav")
#     cmd = [
#         "python", "inference.py",
#         "--checkpoint_path", checkpoint_path,
#         "--face", face_video_path,
#         "--audio", audio_path,
#         "--outfile", output_video_path
#     ]
#     subprocess.run(cmd, check=True)
#     return output_video_path

# # FastAPI Endpoints
# @app.post("/summarize_translate_speech_video/")
# async def process_file(file: UploadFile = File(...), src_lang: str = "eng_Latn", tgt_lang: str = "urd_Arab"):
#     try:
#         # Step 1: Extract Text
#         extracted_text = extract_text(file)
#         if not extracted_text:
#             raise HTTPException(status_code=400, detail="Failed to extract text from file.")

#         # Step 2: Summarize
#         summary = summarize_text(extracted_text)

#         # Step 3: Translate
#         translator = load_translator(src_lang=src_lang, tgt_lang=tgt_lang)
#         translated_text = translator(summary)

#         # Step 4: Convert to Speech
#         audio_path = text_to_speech(translated_text, tgt_lang, output_file="output.wav")

#         # Step 5: Generate Lip-Synced Video
#         video_path = generate_lip_synced_video(
#             translated_text, src_lang, tgt_lang, FACE_VIDEO_PATH, CHECKPOINT_PATH, OUTPUT_VIDEO_PATH
#         )

#         return {
#             "message": "Process completed successfully",
#             "summary": summary,
#             "translated_text": translated_text,
#             "audio_path": audio_path,
#             "video_path": video_path
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Run API
# # uvicorn api:app --host 0.0.0.0 --port 8000

###########################################################################################################################

import os
import fitz  # PyMuPDF for PDF text extraction
import subprocess
import soundfile as sf
import jwt
import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from ttsmms import TTS
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
SECRET_KEY = "your_secret_key"  # Change this to a secure key
ALGORITHM = "HS256"

if not api_key:
    raise ValueError("Error: GROQ_API_KEY is not set in .env")

# Initialize Groq client
client = Groq(api_key=api_key)

app = FastAPI()
security = HTTPBearer()

# Paths for Wav2Lip
UPLOAD_DIR = "/home/hamza/Desktop/translater/Wav2Lip/uploads/"
OUTPUT_VIDEO_PATH = "/home/hamza/Desktop/translater/Wav2Lip/results/output.mp4"
CHECKPOINT_PATH = "/home/hamza/Desktop/translater/Wav2Lip/checkpoints/wav2lip_gan.pth"

os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure upload directory exists

# Generate JWT Token
@app.get("/token/")
def generate_token():
    expiration_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    token = jwt.encode({"exp": expiration_time}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token}

# Token Validation
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Extract text from PDF or TXT
def extract_text(file):
    file_extension = os.path.splitext(file.filename)[-1].lower()

    if file_extension == ".pdf":
        text = ""
        try:
            doc = fitz.open(stream=file.file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text("text") + "\n"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {e}")
        return text.strip()

    elif file_extension == ".txt":
        try:
            return file.file.read().decode("utf-8").strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading text file: {e}")

    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a PDF or TXT file.")

# Summarize text using Groq API
def summarize_text(text):
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system",
                "content": "You are an AI expert at summarization. Provide structured and topic-wise summaries."},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
        ]
    )
    return response.choices[0].message.content

# Convert Text to Speech
def text_to_speech(text, output_file="output.wav"):
    model_path = "/home/hamza/Desktop/translater/data/eng"
    tts = TTS(model_path)
    wav = tts.synthesis(text)
    
    if "x" in wav and "sampling_rate" in wav:
        sf.write(output_file, wav["x"], wav["sampling_rate"])
        return output_file
    else:
        raise RuntimeError("Speech synthesis failed.")

# Generate Lip-Synced Video
def generate_lip_synced_video(audio_path, face_video_path):
    cmd = [
        "python", "inference.py",
        "--checkpoint_path", CHECKPOINT_PATH,
        "--face", face_video_path,
        "--audio", audio_path,
        "--outfile", OUTPUT_VIDEO_PATH
    ]
    subprocess.run(cmd, check=True)
    return OUTPUT_VIDEO_PATH

# API Endpoint: Summarize + Speech + Video
@app.post("/summarize_speech_video/")
async def process_file(
    file: UploadFile = File(...),
    face_video: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    try:
        # Save face video
        video_path = os.path.join(UPLOAD_DIR, face_video.filename)
        with open(video_path, "wb") as f:
            f.write(face_video.file.read())

        # Extract text
        extracted_text = extract_text(file)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from file.")

        # Summarize
        summary = summarize_text(extracted_text)

        # Convert to Speech
        audio_path = text_to_speech(summary, output_file="output.wav")

        # Generate Lip-Synced Video
        video_output_path = generate_lip_synced_video(audio_path, video_path)

        return {
            "message": "Process completed successfully",
            "summary": summary,
            "audio_path": audio_path,
            "video_path": video_output_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run API
# uvicorn api:app --host 0.0.0.0 --port 8000
