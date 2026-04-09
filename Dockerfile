FROM python:3.10

WORKDIR /app

COPY . .

<<<<<<< HEAD
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["python", "-m", "server.app"]
=======
RUN pip install --no-cache-dir fastapi uvicorn pydantic

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
