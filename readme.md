## Create a .env file for environment variables
- API_KEY="your pinecone API key"
- MODE="test or any other string"-m

## Steps
- Create a virtual env
- Install the dependencies from requirements.txt
- Run the fastapi server

```
python -m venv env
pip install -r requirements.txt
fastapi dev main.py
```

## Run the web service
- Open browser and run "http://localhost:8000/"

## Extra options
- The threshold cutoff value for cosine score can be changed using the threshold input and radio button
    - Values should be between 0 and 1