from django.shortcuts import render
from movie.models import Movie
from django.core.management.base import BaseCommand
import os
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

from dotenv import load_dotenv, find_dotenv

# Create your views here.
def recommendation(request):
    recommendationPrompt = request.GET.get('recommendationSearch')
    recommendedMovie = ""
    if recommendationPrompt:
        recommendationReturned= recommendationMethod(recommendationPrompt)
        recommendedMovie= Movie.objects.get(title=recommendationReturned)
    return render(request, 'recom.html', {'recommendationPrompt':recommendationPrompt, 'recommendedMovie':recommendedMovie})

def recommendationMethod(recommendationPrompt):
    _ = load_dotenv('../openAI.env')
    openai.api_key  = os.environ['openAI_api_key']
    
    items = Movie.objects.all()

    req = recommendationPrompt
    emb_req = get_embedding(req,engine='text-embedding-ada-002')

    sim = []
    for i in range(len(items)):
        emb = items[i].emb
        emb = list(np.frombuffer(emb))
        sim.append(cosine_similarity(emb,emb_req))
    sim = np.array(sim)
    idx = np.argmax(sim)
    idx = int(idx)
    return items[idx].title