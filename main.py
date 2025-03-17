from ttsmodule import generate_lip_synced_video

video_path = generate_lip_synced_video(
    input_text="""Introduction

I am Jaggi, an AI and Machine Learning Developer with a strong focus on building and fine-tuning large language models (LLMs),
generative AI, and deep learning applications. My expertise lies in Python programming, model debugging, and optimizing various
AI models for real-world applications. With a passion for innovation, I work extensively with frameworks like TensorFlow, PyTorch, 
and Hugging Face to develop state-of-the-art AI solutions.""",
    src_lang="eng_Latn",
    tgt_lang="urd_Arab",
    face_video_path="/home/hamza/Desktop/translater/Wav2Lip/kashif_demo.mp4",
    checkpoint_path="/home/hamza/Desktop/translater/Wav2Lip/checkpoints/wav2lip.pth",
    output_video_path="/home/hamza/Desktop/translater/Wav2Lip/results/output.mp4"
)

print(f"Lip-synced video saved at: {video_path}")

