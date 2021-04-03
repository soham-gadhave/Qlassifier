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
		
		# print(request.GET.get("text"))
		try:
			if request.GET.get("text"):
				data = {'text': request.GET.get("text")}
			else:
				data = json.loads(request.body.decode("utf-8"))
		except (TypeError, json.JSONDecodeError):
			return JsonResponse({"message": "Bad Request, check if you are not sending Empty or invalid Data"}, status=400)

		try:
			x = clean_and_extract([str(data['text'])])
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
			threshold = [0.18, 0.17, 0.4, 0.4, 0.12]
			
			for i in range(len(probabilities)):
				if probabilities[i] >= threshold[i]:
					Type.append("Insincere")
				else:
					Type.append("Sincere")

			response_body = {
				"text" : str(data['text']),
				"models": {
					"LR_TFIDF": {
						"type": Type[0],
						"probability": probabilities[0] 
					},
					"LR_COUNT": {
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
			return JsonResponse({"message" : "Sorry something went wrong, counld not process your request. Try again", "error": str(exception)}, status=500)

	if request.method == "POST":

		try:
			data = json.loads(request.body.decode("utf-8"))
		except Exception as e:
			return JsonResponse({"message" : "Bad Data", "error": str(e)}, status=422)
		if data == []:
			return JsonResponse({"message" : "Empty data", "error": "No Data sent for processing"}, status=400)
		if type(data) != type([]): 
			return JsonResponse({"message" : "Bad Data", "error": "Data is not in JSON array/list format"}, status=422)
				
		try:
			questions = []
			try:
				for ele in data:
					questions.append(str(ele["text"]))
			except KeyError as ke:
				return JsonResponse({"message" : """"text" attribute not found. Try again""", "error": str(ke)}, status=422)

			x = clean_and_extract(questions)
			x_te = x[0]
			x_te1 = x[1]
			x_c = x[2]

			print(x_te)

			prob_LR_tfidf = LR_tfidf.predict_proba(x_te)[:,1]
			prob_SVC_tfidf = SVC_tfidf._predict_proba_lr(x_te)[:,1]
			prob_NB_count = NB_count.predict_proba(x_c)[:,1]
			prob_LR_count = LR_count.predict_proba(x_te1)[:,1]
			prob_SVC_count = SVC_count._predict_proba_lr(x_te1)[:,1]

			probabilities = [prob_LR_tfidf, prob_LR_count, prob_SVC_tfidf, prob_SVC_count, prob_NB_count]
			Type = [[], [], [], [], []]
			response = []
			threshold = [0.18, 0.17, 0.4, 0.4, 0.12]

			print(probabilities)

			for i in range(len(probabilities)):
				for j in range(len(probabilities[i])):
					if probabilities[i][j] >= threshold[i]:
						Type[i].append("Insincere")
					else:
						Type[i].append("Sincere")

			for i in range(len(questions)):			
				
				response_body = {
					"text" : questions[i],
					"models": {
						"LR_TFIDF": {
							"type": Type[0][i],
							"probability": probabilities[0][i] 
						},
						"LR_COUNT": {
							"type": Type[1][i],
							"probability": probabilities[1][i] 
						},
						"SVC_TFIDF": {
							"type": Type[2][i],
							"probability": probabilities[2][i] 
						},
						"SVC_COUNT": {
							"type": Type[3][i],
							"probability": probabilities[3][i]
						},
						"NB": {
							"type": Type[4][i],
							"probability": probabilities[4][i] 
						}
					}
				}

				response.append(response_body)

			response = JsonResponse(response, safe = False)

			return response
		except Exception as exception:
			return JsonResponse({"message" : "Sorry something went wrong, counld not process your request. Try again", "error": str(exception)}, status=500)