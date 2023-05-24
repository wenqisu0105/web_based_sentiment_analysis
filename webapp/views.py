from django.shortcuts import render
from .utils.models.model import predict_sent

def question_answer_view(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        answer = predict_sent(question)
        return render(request, 'answer.html', {'answer': answer})
    else:
        return render(request, 'question.html')