# openAI
Code for openAI's gym environment in R

To download the code and install the requirements, you can run the following shell commands:

    git clone https://github.com/openai/gym-http-api
    cd gym-http-api
    pip install -r requirements.txt

This code is intended to be run locally by a single user. The server runs in python.

To start the server from the command line, run this:

    python gym_http_server.py
    
In R we need the `gym` package:

    library(gym)