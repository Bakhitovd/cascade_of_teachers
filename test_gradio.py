from gradio_client import Client

client = Client("http://localhost:7860/")

result = client.predict(
    "Howdy!",               # Text Prompt
    "default,default",      # Style
    "C:/projects/cascade_of_teachers/temp/audio_samples/speach_sample_1_min_ru.wav",  # local WAV path
    True,                   # Agree
    fn_index=1,
)

print("RESULT:", result)
