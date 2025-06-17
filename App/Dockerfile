# Use a minimal Conda base image
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy environment.yml first (to cache Docker layer)
COPY environment.yml .

# Create the conda environment from your file
RUN conda env create -f environment.yml

# Make sure Conda is activated in every command
SHELL ["conda", "run", "-n", "flask_env", "/bin/bash", "-c"]

# Install gunicorn explicitly in the environment (if not in .yml)
RUN conda run -n flask_env pip install gunicorn

# Copy the entire app content
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Start the app using gunicorn from inside the conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "flask_env", "gunicorn", "-b", "0.0.0.0:8080", "app:app"]
