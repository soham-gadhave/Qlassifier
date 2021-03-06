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
		
		# print(request.GET.get("question_text"))
		try:
			if request.GET.get("question_text"):
				data = {'question_text': request.GET.get("question_text")}
			else:
				data = json.loads(request.body.decode("utf-8"))
		except (TypeError, json.JSONDecodeError):
			return JsonResponse({"message": "Bad Request, check if you are not sending Empty or invalid Data"}, status=400)

		try:
			x = clean_and_extract([str(data['question_text'])])
			x_te = x[0]
			x_te1 = x[1]
			x_c = x[2]

			prob_LR_tfidf = LR_tfidf.predict_proba(x_te)[:,1][0]
			prob_LR_count = LR_count.predict_proba(x_te1)[:,1][0]
			prob_SVC_tfidf = SVC_tfidf._predict_proba_lr(x_te)[:,1][0]
			prob_SVC_count = SVC_count._predict_proba_lr(x_te1)[:,1][0]
			prob_NB_count = NB_count.predict_proba(x_c)[:,1][0]


			Type = []
			probabilities = [prob_LR_tfidf, prob_LR_count, prob_SVC_tfidf, prob_SVC_count, prob_NB_count]
			
			for prob in probabilities:
				if prob >= 0.5:
					Type.append("Insincere")
				else:
					Type.append("Sincere")

			response_body = {
				"question_text" : str(data['question_text']),
				"models": {
					"LR_TFIDF": {
						"type": Type[0],
						"probability": probabilities[0] 
					},
					"LF_COUNT": {
						"type": Type[1],
						"probability": probabilities[1] 
					},
					"SVC_TFIDF": {
						"type": Type[2],
						"probability": probabilities[2] 
					},
					"SVC_COUNT": {
						"type": Type[3],
						"probability": probabilities[3] 
					},
					"NB": {
						"type": Type[4],
						"probability": probabilities[4] 
					}
				}
			}

			response = JsonResponse(response_body, status=200)
			
			return response
		except Exception as exception:
			return JsonResponse({"message" : "Sorry something went wrong, counld not process ypur request. Try again", "error": exception}, status=500)

	if request.method == "POST":

		data = json.loads(request.body.decode("utf-8"))

		print(data)

		questions = []
		for ele in data:
			questions.append(str(ele["question_text"]))

		# X_te = clean_and_extract(questions)

		# probabilities = LR_tfidf[0].predict_proba(X_te)[:,1]

		x = clean_and_extract(questions)
		x_te = x[0]
		x_te1 = x[1]
		x_c = x[2]

		prob_LR_tfidf = LR_tfidf.predict_proba(x_te)[:,1][0]
		prob_SVC_tfidf = SVC_tfidf._predict_proba_lr(x_te)[:,1][0]
		prob_NB_count = NB_count.predict_proba(x_c)[:,1][0]
		prob_LR_count = LR_count.predict_proba(x_te1)[:,1][0]
		prob_SVC_count = SVC_count._predict_proba_lr(x_te1)[:,1][0]

		probabilities = [prob_LR_tfidf, prob_SVC_tfidf, prob_NB_count, prob_LR_count, prob_SVC_count]	

		response = []
		for i in range(len(questions)):

			Type = []
			for prob in probabilities:
				if prob >= 0.5:
					Type.append("Insincere")
				else:
					Type.append("Sincere")

			response_body = {
				"question_text" : questions[i],
				"models": {
					"LR_TFIDF": {
						"type": Type[0],
						"probability": probabilities[0] 
					},
					"LF_COUNT": {
						"type": Type[1],
						"probability": probabilities[1] 
					},
					"SVC_TFIDF": {
						"type": Type[2],
						"probability": probabilities[2] 
					},
					"SVC_COUNT": {
						"type": Type[3],
						"probability": probabilities[3] 
					},
					"NB": {
						"type": Type[4],
						"probability": probabilities[4] 
					}
				}
			}

			# Type = ""
			# if probabilities[i] >= 0.5:
			# 	Type = "Insincere"
			# else:
			# 	Type = "Sincere"

			# response = {
			# 	"question_text" : questions[i],
			# 	"type" : Type,
			# 	"probability" : float(probabilities[i]),
			# 	"model" : "Logistic Regression"
			# }
			response.append(response_body)

		response = JsonResponse(response, safe = False)

		return response