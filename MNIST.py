# Import necessary libraries
from torchvision import datasets, transforms
import torch.utils.data as data
from sklearn import decomposition
from torch.autograd import Variable

# Compose transforms. Here we only convert images to tensor format.
tf = transforms.Compose([transforms.ToTensor()])  

# Load the MNIST dataset  
mnist_dataset = datasets.MNIST('../data', train=True, download=True, transform=tf)

# Create a loader for the entire dataset
mnist_full_loader = data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=False)

# Get the first batch of the entire dataset
all_digits, all_labels = next(iter(mnist_full_loader))

# Reshape the data to 2D
all_digits_np, all_labels_np = all_digits.reshape(-1, 784).data.cpu().numpy(), all_labels.data.cpu().numpy()

# Perform PCA on the data
estimator = decomposition.PCA(n_components=4, svd_solver='randomized', whiten=True)

# Fit the PCA on the data
all_projected_cordinates = estimator.fit_transform(all_digits_np)

# Print the shape of the projected cordinates
print(all_projected_cordinates.shape)

# Reconstruct the digits from the projected cordinates
reconstructed_digits = estimator.inverse_transform(all_projected_cordinates)

logging.info("Finished the PCA on MNIST \n")

class DataGenerator(object):
    "superclass of all data. WARNING: doesn't raise StopIteration so it loops forever!"

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def get_batch(self):
        raise NotImplementedError()

    def float_tensor(self, batch):
        return torch.from_numpy(batch).type(torch.FloatTensor)

class RealDataGeneratorDummy(DataGenerator):
    """samples from real data"""
    def __init__(self, loader):
        self.loader = loader
        self.generator = iter(self.loader)
        self.data_len = len(self.loader)
        self.count = 0

    def get_batch(self):
        if (((self.count + 1) % self.data_len) == 0):
            del self.generator
            self.generator = iter(self.loader)
        self.count += 1
        return next(self.generator)

first_five_digit_cordinates_torch = torch.from_numpy(reconstructed_digits[all_labels_np < 5]).float()
first_five_digit_cordinates_generator = RealDataGeneratorDummy(torch.utils.data.DataLoader(first_five_digit_cordinates_torch, batch_size=100, shuffle=True))

def to_var(x, requires_grad=False):
    """Converts numpy to variable."""
    return Variable(x, requires_grad=requires_grad)

def get_data(real_data_generator=first_five_digit_cordinates_generator):
    real_data = to_var(next(real_data_generator))
    return real_data

real_data = get_data()
real_data.size()

print("Content of real_data:\n", real_data[1].size())