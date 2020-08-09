# Run the test harness for evaluating a model

from keras.models import load_model
from dataset import load
from evaluate_model import evaluate_model
from prepare_pixel_data import prepare_pixel_data
from evaluate_model import summarize_diagnostics
from evaluate_model import summarize_performance
from neural_network import define_model

# Evaluate model

def run_eval_model():
	trainX, trainY, testX, testY = load()
	trainX, testX = prepare_pixel_data(trainX, testX)
	evaluate_model(trainX, trainY)
	scores, histories = evaluate_model(trainX, trainY)
	summarize_diagnostics(histories)
	summarize_performance(scores)

# Save final model

def run_save_model():
	trainX, trainY, testX, testY = load()
	trainX, testX = prepare_pixel_data(trainX, testX)
	model = define_model()
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
	model.save('final_model.h5')

# Evaluate final model

def run_eval_final_model():
	trainX, trainY, testX, testY = load()
	trainX, testX = prepare_pixel_data(trainX, testX)
	model = define_model()
	model = load_model('final_model.h5')
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))