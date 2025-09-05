import whisper
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import AudioUploadSerializer
from django.conf import settings

# Load Whisper model once
model = whisper.load_model("base")

class TranscribeAudio(APIView):
    def post(self, request, *args, **kwargs):
        serializer = AudioUploadSerializer(data=request.data)
        if serializer.is_valid():
            audio_file = serializer.validated_data['audio']
            file_path = os.path.join(settings.BASE_DIR, 'uploaded_audio', audio_file.name)

            # Ensure upload directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save file
            with open(file_path, 'wb+') as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)

            # Transcribe
            result = model.transcribe(file_path)
            text = result['text']

            return Response({"transcription": text}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
