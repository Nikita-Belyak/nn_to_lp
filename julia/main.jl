
# Get the current script's directory
current_dir = dirname(@__FILE__)

# Create the path for the data folder one level above
data_folder_path = joinpath(dirname(current_dir), "data")

include("npy_to_layers.jl")
include("nn_to_lp.jl")
include("nn_vs_lp.jl")

# Load the layers from the .npy files
nn_layers = npy_to_layers(data_folder_path*"/nn", "nn")
icnn_layers = npy_to_layers(data_folder_path*"/icnn", "icnn")

# generate the linear programming model
icnn_lp_model = icnn_to_lp(icnn_layers)

for i in eachindex(icnn_lp_model[:y])
    JuMP.fix(icnn_lp_model[:y][i], 1.5)
end
optimize!(icnn_lp_model)
value.(icnn_lp_model[:z][3,:])

# compare the output of the input convex network and the output of the linear programming model
icnn_vs_lp(icnn_layers, icnn_lp_model)


nn_lp_model = nn_to_lp(nn_layers)
for i in eachindex(nn_lp_model[:y])
    JuMP.fix(nn_lp_model[:y][i], 4)
end
optimize!(nn_lp_model)
print("NN LP Model\n")
print(nn_lp_model)

