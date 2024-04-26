
# Get the current script's directory
current_dir = dirname(@__FILE__)

# Create the path for the data folder one level above
data_folder_path = joinpath(dirname(current_dir), "data/nn")

include("npy_to_layers.jl")
include("nn_to_lp.jl")
include("nn_vs_lp.jl")

# Load the layers from the .npy files
layers = npy_to_layers(data_folder_path, "nn")

# generate the linear programming model
lp_model = icnn_to_lp(layers)
lp_model = nn_to_lp(layers)

# compare the output of the input convex network and the output of the linear programming model
nn_vs_lp(layers, lp_model)


