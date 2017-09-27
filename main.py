from Model import POSModel
d = 0.80
print "\nKN Smoothing", d
model = POSModel(smoothing = 'kn', d_kn = d)
model.predict_on_test(train_file_name='./berp-POS-training.txt', test_file_name='./assgn2-test-set.txt')