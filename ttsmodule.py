# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from ttsmms import TTS
# import subprocess
# import soundfile as sf
# import os

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
# def load_translator(model_name="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="urd_Arab"):
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

# # **Main Function: Call from Other Files or CLI**
# def text_to_voice(input_text=None, src_lang=None, tgt_lang=None, output_file="output.wav"):
#     if input_text is None:
#         input_text = input("Enter the text to translate: ")

#     if src_lang is None:
#         print("Available Languages: English (eng_Latn), Urdu (urd_Arab), Arabic (arb_Arab)")
#         src_lang = input("Enter source language code: ")

#     if tgt_lang is None:
#         tgt_lang = input("Enter target language code: ")

#     # Load Translator
#     translator = load_translator(src_lang=src_lang, tgt_lang=tgt_lang)
    
#     # Translate the text
#     translated_text = translator(input_text)
#     print(f"\n[INFO] Translated Text: {translated_text}")

#     # Convert to Speech
#     tts_output = text_to_speech(translated_text, tgt_lang, output_file)
#     print(f"\n[INFO] Voice Output Saved at: {tts_output}")
    
#     return tts_output

# # Run only when executed directly
# if __name__ == "__main__":
#     text_to_voice()

# def generate_lip_synced_video(input_text, src_lang, tgt_lang, face_video_path, checkpoint_path, output_video_path):
#     """
#     Generates a lip-synced video by first converting text to speech and then using Wav2Lip for lip-syncing.
#     """
#     # Step 1: Convert text to voice
#     audio_path = text_to_voice(input_text, src_lang, tgt_lang, output_file="output.wav")
#     print(f"Generated speech audio: {audio_path}")

#     # Step 2: Run Wav2Lip inference
#     cmd = [
#         "python", "inference.py",
#         "--checkpoint_path", checkpoint_path,
#         "--face", face_video_path,
#         "--audio", audio_path,
#         "--outfile", output_video_path
#     ]

#     subprocess.run(cmd, check=True)
#     print(f"Lip-synced video generated at: {output_video_path}")
#     return output_video_path


