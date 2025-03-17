from textspech import text_to_voice  # Assuming the file is saved as tts_module.py

output_wav = text_to_voice(
    input_text="Hello, how are you? my name is hamza sandhu from Lahore",
    src_lang="eng_Latn",
    tgt_lang="urd_Arab"
)

print(f"Generated voice file: {output_wav}")
