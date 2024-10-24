# Use an official Python runtime as a parent image
FROM python:3.10.15-bookworm

# Install curl and other required tools for Poetry
RUN apt-get update && apt-get install -y curl

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy only the pyproject.toml and poetry.lock first (for better Docker layer caching)
COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN poetry install --no-dev

# Copy the rest of the application files (app.py, lgbm.pkl, etc.)
COPY . /app

# Expose the port that Flask runs on
EXPOSE 5000

# Run the Flask app
CMD ["poetry", "run", "python", "app.py"]
