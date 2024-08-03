# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable
ENV NAME CALIFORNIAHousingModel

# Run model.py when the container launches
CMD ["python", "model.py"]
