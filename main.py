from Model import POSModel
model = POSModel(smoothing= 'laplace')
model.read(file_name='./berp-POS-training.txt')
model._get_prob_tag_transition_dict()
