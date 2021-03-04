import os
import json
import pickle
import pandas as pd
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from Machine_Learning.Processing.features_extraction import clean_and_extract

#CSRF token is sort of an authentication token, which only client and server knows,
#so its excluded here because no auth is needed. Its commonly use in HTML forms.  

@csrf_exempt
def index(request):

	LR_tfidf = pickle.load(open(settings.BASE_DIR / "Machine_Learning/Models/LR_tfidf.sav", "rb"))
	SVC_tfidf = pickle.load(open(settings.BASE_DIR / "Machine_Learning/Models/SVC_tfidf.sav", "rb"))
	NB_count = pickle.load(open(settings.BASE_DIR / "Machine_Learning/Models/NB_count.sav", "rb"))
	LR_count = pickle.load(open(settings.BASE_DIR / "Machine_Learning/Models/LR_count.sav", "rb"))
	SVC_count = pickle.load(open(settings.BASE_DIR / "Machine_Learning/Models/SVC_count.sav", "rb"))

	if request.method == 'GET':
		
		data = json.loads(request.body.decode("utf-8"))
		
		x = clean_and_extract([str(data['question_text'])])
		x_te = x[0]
		x_te1 = x[1]
		x_c = x[2]

		prob_LR_tfidf = LR_tfidf.predict_proba(x_te)[:,1][0]
		prob_SVC_tfidf = SVC_tfidf._predict_proba_lr(x_te)[:,1][0]
		prob_NB_count = NB_count.predict_proba(x_c)[:,1][0]
		prob_LR_count = LR_count.predict_proba(x_te1)[:,1][0]
		prob_SVC_count = SVC_count._predict_proba_lr(x_te1)[:,1][0]


		Type = []
		probabilities = [prob_LR_tfidf, prob_SVC_tfidf, prob_NB_count, prob_LR_count, prob_SVC_count]
		
		for prob in probabilities:
			if prob >= 0.5:
				Type.append("Insincere")
			else:
				Type.append("Sincere")
		
		print(Type)

		response_body = {
			"question_text" : str(data['question_text']),
			"type_LR_tfidf" : Type[0],
			"probability_LR_tfidf" : probabilities[0],
			"type_SVC_tfidf" : Type[1],
			"probability_SVC_tfidf" : probabilities[1],
			"type_NB_tfidf" : Type[2],
			"probability_NB_count" : probabilities[2],
			"type_LR_count" : Type[3],
			"probability_LR_count" : probabilities[3],
			"type_SVC_count" : Type[4],
			"probability_SVC_count" : probabilities[4]
		}

		response = JsonResponse(response_body)
		
		return response

	if request.method == "POST":

		data = json.loads(request.body.decode("utf-8"))

		questions = []
		for ele in data:
			questions.append(str(ele["question_text"]))

		X_te = clean_and_extract(questions)

		probabilities = LR_tfidf[0].predict_proba(X_te)[:,1]

		response_body = []
		for i in range(len(probabilities)):

			Type = ""
			if probabilities[i] >= 0.5:
				Type = "Insincere"
			else:
				Type = "Sincere"

			response = {
				"question_text" : questions[i],
				"type" : Type,
				"probability" : float(probabilities[i]),
				"model" : "Logistic Regression"
			}
			response_body.append(response)

		response = JsonResponse(response_body, safe = False)

		return response