
# HUK Coburg Technical Challenge

This project is a Flask-based REST API for making inference calls to a machine learning model predicting marketing affinity of HUK Coburg customers.
The API allows clients to send POST requests with customer data, and it responds with a prediction classifying if a customer will be interested in an offer from HUK Coburg (0 = not interested / 1 = interested).

## Features

- Poetry for dependency management
- Data preprocessing and feature engineering on top of initial costumer data
- REST API built using Flask
- Model is pre-trained and serialized using `joblib`
- Easy deployment with Docker

## Prerequisites

Before you begin, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Poetry](https://python-poetry.org/docs/)


## Quickstart

### 1. Clone the repository

```bash
git clone git@github.com:kris-fillip/HUK_Technical_Challenge.git
cd HUK_Technical_Challenge
```

### 2. Build and Run with Docker

#### Build the Docker Image

To build the Docker image with **Poetry** and all dependencies included:

```bash
docker build -t huk-challenge .
```

#### Run the Docker Container

Run the Flask API inside a Docker container:

```bash
docker run -d -p 8080:5000 huk-challenge
```

This will start the Flask app inside Docker and expose it on `http://localhost:8080`.

### 3. Test the API

You can test the API by sending a POST request with the required features using `curl` or an API client like **Postman**.

#### Example `curl` Request

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
  "id": [107727],
  "Geschlecht": ["Male"],
  "Alter": [42],
  "Interesse": [0.0],
  "Fahrerlaubnis": [1],
  "Regional_Code": [28],
  "Vorversicherung": [0],
  "Alter_Fzg": ["1–2 Year"],
  "Vorschaden": ["No"],
  "Jahresbeitrag": [27324.0],
  "Vertriebskanal": [152.0],
  "Kundentreue": [208]
}'
```

This request will return a prediction from the model in JSON format.

### 4. Stopping the Docker Container

To stop the running container:

```bash
docker ps  # Find the container ID
docker stop <container_id>
```

## Local Development (Without Docker)

New models can currently only be built in local development.

If you want to run the Flask app locally without Docker, follow these steps:

### 1. Install Dependencies

Ensure you have **Poetry** installed, then run:

```bash
poetry install
```

### 2. Run the Flask App

```bash
poetry run python app.py
```

The Flask API will be running at `http://localhost:5000`.

## API Endpoints

### POST `/predict`

This endpoint accepts JSON input with feature data and returns the prediction based on the model.

#### Request Body Example:

```json
{
  "id": [107727],
  "Geschlecht": ["Male"],
  "Alter": [42],
  "Interesse": [0.0],
  "Fahrerlaubnis": [1],
  "Regional_Code": [28],
  "Vorversicherung": [0],
  "Alter_Fzg": ["1–2 Year"],
  "Vorschaden": ["No"],
  "Jahresbeitrag": [27324.0],
  "Vertriebskanal": [152.0],
  "Kundentreue": [208]
}
```

#### Response Example:

```json
{
  "prediction": 1
}
```
