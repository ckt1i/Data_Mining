import numpy as np

# Generate a sinusoid with noise
def generate_sinusoid(start, end, num_points):
    x = np.linspace(start, end, num_points)
    y = np.sin(np.pi * x / end * 4)
    return x, y

def sampling(samp_rate , x , y , distubance): 
    x = x[::samp_rate]
    y = y[::samp_rate]
    noise = distubance * np.random.normal(size=y.size)
    y = y + noise
    return  x , y

def write_to_file(x , y , filename):
#    np.savetxt(filename , np.column_stack((x , y)))
    pass