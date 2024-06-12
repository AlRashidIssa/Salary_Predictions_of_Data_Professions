# Use an official Arch Linux base image
FROM archlinux:latest

# Install system dependencies
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm python python-pip python-virtualenv base-devel

# Create a directory for the app
WORKDIR /app
COPY app/requirements.txt /app
COPY saved_model_joblib/cat_model.joblib /app/saved_model_joblib/cat_model.joblib


# Create and activate a virtual environment
RUN python -m venv venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["/app/venv/bin/python", "app.py"]
