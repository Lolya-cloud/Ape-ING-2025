import sys
from google import genai


def init_google_genai_client() -> genai.client.Client:
    if "google.colab" not in sys.modules:
        import subprocess

        project_id = subprocess.check_output(
            ["gcloud", "config", "get-value", "project"], text=True
        ).strip()

        location = subprocess.check_output(
            ["gcloud", "config", "get-value", "compute/region"], text=True
        ).strip()

        return genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )