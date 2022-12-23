FROM python:3.8

# set a directory for the app
WORKDIR /usr/src/app

# copy all the files to the container
COPY . .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENV DEBUG=${DEBUG}

# tell the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python", "./manage.py"]
