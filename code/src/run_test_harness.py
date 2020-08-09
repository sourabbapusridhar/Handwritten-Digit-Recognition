# Run the test harness for evaluating a model

from keras.models import load_model
from src import dataset, evaluate_model, neural_network, prepare_pixel_data

# Evaluate model

def run_eval_model():
	trainX, trainY, testX, testY = dataset.load()
	trainX, testX = prepare_pixel_data.prepare_pixel_data(trainX, testX)
	evaluate_model.evaluate_model(trainX, trainY)
	scores, histories = evaluate_model.evaluate_model(trainX, trainY)
	evaluate_model.summarize_diagnostics(histories)
	evaluate_model.summarize_performance(scores)

# Save final model

def run_save_model():
	trainX, trainY, testX, testY = dataset.load()
	trainX, testX = prepare_pixel_data.prepare_pixel_data(trainX, testX)
	model = neural_network.define_model()
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
	model.save('final_model.h5')

# Evaluate final model

def run_eval_final_model():
	trainX, trainY, testX, testY = dataset.load()
	trainX, testX = prepare_pixel_data.prepare_pixel_data(trainX, testX)
	model = neural_network.define_model()
	model = load_model('final_model.h5')
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))