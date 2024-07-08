# Use an official Python runtime as a parent image
FROM python:3.10.6

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y bash coreutils

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run sd3_infer.py when the container launches
CMD ["python", "sd3_infer.py", "--model", "models/sd3_medium.safetensors", "--prompt", "cute wallpaper art of a cat"]