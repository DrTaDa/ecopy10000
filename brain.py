import numpy
from scipy.special import softmax


def step(x):
    return numpy.heaviside(x, 1)


def double_relu(x):
    return numpy.clip(x, -1, 1)
 
 
def glorot_uniform(n_in, n_out):

    limit = numpy.sqrt(6 / (n_in + n_out))
    weights = limit * ((numpy.random.random((n_in, n_out)) * 2.) - 1)
    biases = limit * ((numpy.random.random(n_out) * 2.) - 1)

    return weights, biases


class Brain():
    
    layer_names = [
        "layer_hidden", "biases_hidden", "layer_mouv", 
        "biases_mouv", "layer_act", "biases_act", 
        #"layer_mem", "biases_mem"
    ]
 
    def __init__(
        self,
        n_vision_rays=13,
        length_vision_vector=5,
        n_other_inputs=5,
        n_actions=1,
        n_memory=10,
        n_hidden=15,
        model=None
    ):

        self.input_size = (length_vision_vector * n_vision_rays) + n_other_inputs #+ n_memory

        self.n_hidden = n_hidden
        self.n_memory = n_memory
        self.n_actions = n_actions

        self.layer_hidden, self.biases_hidden = glorot_uniform(self.input_size, self.n_hidden)
        self.activ_hidden = step

        self.layer_mouv, self.biases_mouv = glorot_uniform(self.n_hidden, 2)
        self.activ_mouv = double_relu

        self.layer_act, self.biases_act = glorot_uniform(self.n_hidden, self.n_actions)
        self.activ_act = step

        #self.layer_mem, self.biases_mem = glorot_uniform(self.n_hidden, self.n_memory)
        #self.activ_mem = step

        self.flatten = numpy.concatenate([getattr(self, layer_name).flatten() for layer_name in self.layer_names], axis=0)

    def evaluate(self, input):

        tmp = self.activ_hidden(input.dot(self.layer_hidden) + self.biases_hidden)
        output_mov = self.activ_mouv(tmp.dot(self.layer_mouv) + self.biases_mouv)
        output_act = self.activ_act(tmp.dot(self.layer_act) + self.biases_act)
        #output_mem = self.activ_mem(tmp.dot(self.layer_mem) + self.biases_mem)

        #return [output_mov, output_act, output_mem]
        return [output_mov, output_act]

    def copy_weights_and_mutate_vertical(self, brain_a, brain_b=None):

        for layer_name in self.layer_names:

            if brain_b is not None:

                new_weights_a = getattr(brain_a, layer_name)
                new_weights_b = getattr(brain_b, layer_name)

                split_idx = numpy.random.randint(0, len(new_weights_a))
                if numpy.random.random() > 0.5:
                    new_weights = numpy.concatenate((new_weights_a[:split_idx], new_weights_b[split_idx:]), axis=0)
                else:
                    new_weights = numpy.concatenate((new_weights_b[:split_idx], new_weights_a[split_idx:]), axis=0)

            else:
                new_weights = getattr(brain_a, layer_name)

            mutation_rate = numpy.random.random()
            if mutation_rate > 0.99:
                new_weights = new_weights + numpy.random.normal(loc=0.0, scale=0.1, size=new_weights.shape)
            elif mutation_rate > 0.9:
                new_weights = new_weights + numpy.random.normal(loc=0.0, scale=0.05, size=new_weights.shape)
            elif mutation_rate > 0.6:
                new_weights = new_weights + numpy.random.normal(loc=0.0, scale=0.005, size=new_weights.shape)
            else:
                new_weights = new_weights + numpy.random.normal(loc=0.0, scale=0.001, size=new_weights.shape)

            setattr(self, layer_name, new_weights)
        
        self.flatten = numpy.concatenate([getattr(self, layer_name).flatten() for layer_name in self.layer_names], axis=0)

    def copy_weights_and_mutate_puctual(self, brain_a, brain_b=None):

        for layer_name in self.layer_names:

            if brain_b is not None:

                new_weights_a = getattr(brain_a, layer_name)
                new_weights_b = getattr(brain_b, layer_name)

                split_idx = numpy.random.randint(0, len(new_weights_a))
                if numpy.random.random() > 0.5:
                    new_weights = numpy.concatenate((new_weights_a[:split_idx], new_weights_b[split_idx:]), axis=0)
                else:
                    new_weights = numpy.concatenate((new_weights_b[:split_idx], new_weights_a[split_idx:]), axis=0)

            else:
                new_weights = getattr(brain_a, layer_name)

            for i, w in enumerate(new_weights):
                mutation_rate = numpy.random.random()
                if mutation_rate > 0.999:
                    new_weights[i] += (0.2 * numpy.random.random()) - 0.1
                elif mutation_rate > 0.99:
                    new_weights[i] += (0.02 * numpy.random.random()) - 0.01
                elif mutation_rate > 0.9:
                    new_weights[i] += (0.002 * numpy.random.random()) - 0.001
                """if mutation_rate > 0.999:
                    new_weights[i] += (0.02 * numpy.random.random()) - 0.01
                elif mutation_rate > 0.99:
                    new_weights[i] += (0.002 * numpy.random.random()) - 0.001"""

            setattr(self, layer_name, new_weights)
        
        self.flatten = numpy.concatenate([getattr(self, layer_name).flatten() for layer_name in self.layer_names], axis=0)

    def save(self, blob_name, t):

        to_save = []
        for layer_name in self.layer_names:
            to_save.append(getattr(self, layer_name))
        numpy.save(f"./models/{blob_name}__{t}.npy", to_save)

    def load(self, path):

        new_weights = numpy.load(path, allow_pickle=True)
        for i, layer_name in enumerate(self.layer_names):
            setattr(self, layer_name, new_weights[i])
