FROM ubuntu

RUN apt-get update && apt-get install -y python3 python3-pip

RUN apt-get install -y python3-venv
RUN python3 -m venv /venv

COPY . /opt/source-code

ENV PATH="/venv/bin:$PATH"

RUN pip3 install -r /opt/source-code/requirements.txt

CMD ["fastapi", "dev", "/opt/source-code/main.py","--host", "0.0.0.0", "--port", "3000"]

# docker run -e API_KEY=331578f9-2b07-4931-b966-34410ac1353f -e MODE=tes -e STATIC=/opt/source-code/static -e UPLOAD_DIR=/opt/source-code/uploads -p 8000:3000 navigator:2.0 