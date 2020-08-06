# Run the test harness for evaluating a model

from dataset import load
from evaluate_model import evaluate_model
from prepare_pixel_data import prepare_pixel_data
from evaluate_model import summarize_diagnostics
from evaluate_model import summarize_performance

def run_test_harness():
	trainX, trainY, testX, testY = load()
	trainX, testX = prepare_pixel_data(trainX, testX)
	evaluate_model(trainX, trainY)
	scores, histories = evaluate_model(trainX, trainY)
	summarize_diagnostics(histories)
	summarize_performance(scores)

run_test_harness()