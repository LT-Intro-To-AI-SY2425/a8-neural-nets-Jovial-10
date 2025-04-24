from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

xorn = NeuralNet(2, 1, 1)
xorn.train(xor_training_data)
print(xorn.test_with_expected(xor_training_data))


party_training_data = [([0.9, 0.6,0.8,0.3,0.1], [1.0]), ([0.8 , 0.8, 0.4, 0.6, 0.4], [1.0]), ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]), ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]), ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]), ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0])]

party_testing_data = [([1.0, 1.0, 1.0, 0.1, 0.1], []), ([0.5, 0.2, 0.1, 0.7, 0.7], []), ([0.8,0.3,0.3,0.3,0.8], []), ([0.8,0.3,0.3,0.8,0.3], []), ([0.9,0.8,0.8,0.3,0.6], [])]

PoliticalNet = NeuralNet(5,2,1)
PoliticalNet.train(party_training_data, iters= 10000, print_interval = 100)
print(PoliticalNet.test_with_expected(party_testing_data))