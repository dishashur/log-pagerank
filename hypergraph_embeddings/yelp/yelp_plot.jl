using MAT
using Plots, Colors
using DataStructures

data_path = homedir()*"/log-pagerank/hypergraph_embeddings/data/yelp_dataset"
code_path = homedir()*"/log-pagerank/hypergraph_embeddings/yelp"

M = matread(data_path*"/yelp_restaurant_hypergraph.mat")
D = matread(data_path*"/yelp_restaurant_trial_$trials.mat")

U = D["U"]
locations = M["locations"]
states = locations[:, 3]
statesz = counter(states)
state_int2name = M["state_int2name"]

big_states = []
for state in unique(states)
    if statesz[state] > 10
        push!(big_states, state)
    end
end

nstate = length(big_states) 
d = Dict(sort(big_states) .=> 1:nstate)
rvd = Dict(1:nstate .=> sort(big_states))

@show nstate

color_palette = distinguishable_colors(nstate);
fig = Plots.plot()

for i = 1:nstate
    st = rvd[i]
    ids = findall(locations[:, 3] .== st)
    #@show state_int2name[st]
    #@show color_palette[i]
    Plots.scatter!(fig, U[ids,2],U[ids,3],color=color_palette[i], label=state_int2name[st], 
    axis = :off, alpha=0.3,markersize=4,colorbar=false, markerstrokewidth=0)
end
Plots.plot!(fig, size=(800,800), legend=false, framestyle=:none, dpi = 300)
Plots.savefig(code_path*"/yelp_restaurant_H.png")
