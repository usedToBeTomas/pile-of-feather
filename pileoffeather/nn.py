from . import engine

def create(**options):
    model = engine.neuralNetworkModel()
    model.name = options.get("name")
    model.layers = options.get("layers")
    model.initializeWeightsAndBiases()
    return model

def load(**options):
    model = engine.neuralNetworkModel()
    model.name = options.get("name")
    model.load()
    return model

def backpropagation(model, data_input, data_output, **options):
    batch_size, epoch_number, learning_rate = options.get("batch_size"), options.get("epoch_number"), options.get("rate")
    for epoch in range(epoch_number):
        t_loss = 0
        for batch_start in range(0, len(data_input), batch_size):
            batch_end =  min(batch_start + batch_size,len(data_input))
            t_loss += model.computeBatch(data_input[batch_start:batch_end], data_output[batch_start:batch_end], learning_rate)
        print("Epoch = " + str(epoch) + " Loss = " + str(t_loss/(len(data_input)/batch_size)), end='\r')
    model.save()
    print("\nTraining finished, model saved.")
