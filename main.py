from Model import POSModel
model = POSModel()
model.read(file_name='./berp-POS-training.txt')
model.train(1000)
