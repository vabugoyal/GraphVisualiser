# created by me
from django.http import HttpResponse
from django.shortcuts import render
from . import process


def index(request):
    return render(request, 'index.html')


def solve(request):
    # is request se data utha sakte hai form ka
    # print(request.GET.get('givenMatrix', 'default')) : jis element ka naam text hai request mai uska data utha lega
    solution = process.simulateGraph(request)[2]
    params = {"stages" : solution}
    # here I have to process the images which are formed
    return render(request, "answer.html", params)
