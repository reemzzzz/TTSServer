import os


# def synthesize_text(text, step="900000"):
#     # Get the absolute path to the FastSpeech2 directory
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     fastspeech_dir = os.path.join(base_dir, "FastSpeech2")

#     # Ensure the FastSpeech2 directory exists
#     if not os.path.exists(fastspeech_dir):
#         raise FileNotFoundError(f"FastSpeech2 directory not found: {fastspeech_dir}")

#     # Build the command to run the synthesis script
#     command = [
#         "python", "synthesize.py",
#         "--text", text,
#         "--restore_step", step,
#         "--mode", "single",
#         "-p", "config/LJSpeech/preprocess.yaml",
#         "-m", "config/LJSpeech/model.yaml",
#         "-t", "config/LJSpeech/train.yaml"
#     ]

#     # Run the synthesis script as a subprocess
#     try:
#         subprocess.run(
#             command,
#             cwd=fastspeech_dir,  # Set the working directory for the subprocess
#             check=True  # Raise an exception if the subprocess fails
#         )
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"Synthesis failed: {e}")


from FastSpeech2.synthesize import synthesize_from_text

import os
from contextlib import contextmanager

@contextmanager
def change_dir(path):
    prev_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_dir)

def synthesize_text(text):
    fastspeech_dir = os.path.join(os.path.dirname(__file__), "FastSpeech2")
    with change_dir(fastspeech_dir):
        from  FastSpeech2.synthesize import synthesize_from_text
        return synthesize_from_text(
        text=text,
        step="900000",
        preprocess_path=os.path.join(fastspeech_dir, "config/LJSpeech/preprocess.yaml"),
        model_path=os.path.join(fastspeech_dir, "config/LJSpeech/model.yaml"),
        train_path=os.path.join(fastspeech_dir, "config/LJSpeech/train.yaml")
        )



    

# Example usage:
if __name__ == "__main__":
    synthesize_text("Hello, this is a test from wrapper.")


