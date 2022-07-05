using MAT
using Plots, Colors
using DataStructures

data_path = homedir()*"/log-pagerank/hypergraph_embeddings/data/walmart-trips"
code_path = homedir()*"/log-pagerank/hypergraph_embeddings/walmart"

trials = 10000

M = matread(data_path*"/walmart.mat")
D = matread(data_path*"/walmart_trial_$trials.mat")

U = D["U"]
H = M["H"]
labels = M["labels"]
labelname = M["labelname"]

cnt = counter(labels)
big_label = []
for label in unique(labels)
    if cnt[label] > 10
        push!(big_label, label)
    end
end
nlabel = length(labelname)
color_palette = distinguishable_colors(nlabel);
d = Dict(sort(big_label) .=> 1:nlabel)
rvd = Dict(1:nlabel .=> sort(big_label))

fig = Plots.plot()

for i = 1:nlabel
    st = rvd[i]
    ids = findall(labels .== st)
    #@show labelname[st]
    #@show color_palette[i]
    Plots.scatter!(fig, U[ids,2],U[ids,3],color=color_palette[i], label=labelname[st], 
    axis = :off, alpha=0.3,markersize=4,colorbar=false, markerstrokewidth=0)
end
Plots.plot!(size=(800,800), legend=false, framestyle=:none, dpi = 300)
Plots.savefig(code_path*"/walmart_H.png")