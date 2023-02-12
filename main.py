from train import Train
from model import Model

def main():
	dataset_path = './dataset'
	save_path = './weights'
	num_classes = 2

	model = Model(num_classes).model()
	train = Train(dataset_path,save_path)

	train.train(model)


if __name__ == '__main__':
	main()