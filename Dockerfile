FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir torch torchvision

CMD ["bash"]
