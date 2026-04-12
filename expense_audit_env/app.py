# OpenEnv Space Root Shim
# This file is used for compatibility with the Hugging Face Spaces 'app_file' requirement

from server.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
