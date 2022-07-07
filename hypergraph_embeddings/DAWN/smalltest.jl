include("../common.jl")
include("../local-hyper.jl")
using SparseArrays
using MatrixNetworks
using Plots
using LinearAlgebra
using Statistics
using Statistics
using Random
using LazySets
using JLD2, FileIO
using PyCall
@pyimport collections as py_collections

using MAT

min_size = 3
max_size = 1000000

M = matread("dawn.mat")
H = M["H"]
EdgeLabels = vec(M["EdgeColors"])
NodeNames = M["NodeNames"]
Labels = M["Labels"]
num_nodes = M["n"]
order = vec(round.(Int64,sum(H,dims=2)))
good = findall(x->x > 0, order)
H = H[good,:]
order = vec(round.(Int64,sum(H,dims=2)))
G = LH.graph(H,1.0) # a hypergraph object with delta=1.0 in its cut function
volA = sum(G.deg)
 @show size(G.deg)
 @show volA
 @show size(G.order)
 edge_labels = py_collections.Counter(EdgeLabels)
 edge_labels = [(key,edge_labels[key]) for key in keys(edge_labels)]
 edge_labels = sort(filter(x->(x[2]>min_size && x[2]<max_size),edge_labels),by=x->x[2])
 labels = [x[1] for x in edge_labels]
 #indx = labels .<= length(LabelNames)
 #labels = labels[findall(x->x==1, indx)]
 clusters = Dict()
 for i = 1:length(labels)
       label = labels[i]
       temp = findall(x->x ==label,EdgeLabels)
       T = []
       for he in temp
           nd = findnz(H[he,:])[1]
           for a in nd
               push!(T,a)
           end
       end
       T = unique(T)
       nT =length(T)
       clusters[label] = (label,T,Labels[label])
 end
#this clusters for each condition show all the medicines that were
#involved across patients (that is hyperedges)
#color according to the condition



kappa = 0.01 # value of kappa (sparsity regularization)
rho = 0.5 # value of rho (KKT apprx-val)

function prmatrix(G,clusters;ratio=0.01,delta=0.0,max_iters=10000, x_eps=1.0e-8,aux_eps=1.0e-8,kappa=0.0025,gamma=0.1,rho=0.5,q=2.0)
    X = []
    L = LH.loss_type(q)
    for label in keys(clusters)
    T = clusters[label][2]
    @show label
    @show n = size(T,1)
    @show seednum = max(round(Int64,ratio*n),5)
    Random.seed!(0)
    for indx=1:seednum
    seednode = T[rand(1:n)]
    x,r,iter = LH.lh_diffusion(G,seednode,gamma,kappa,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
    push!(X,x)
    end
    end
 return X
end

alpha = 0.999
gamma = (1/alpha)-1
@time X = prmatrix(G, clusters,ratio=0.04,gamma = gamma)


@show("We have X")

#post processing X for embedding
m = length(X)
n = size(G.H,2)
ppr = spzeros(n,m)
for i=1:m
ppr[:,i] = X[i]
end
X = ppr

using SparseArrays
u,v,vals = findnz(X)
minele = log(minimum(vals))
lX = minele*ones(size(X,1),size(X,2))
for i=1:length(u)
    lX[u[i],v[i]] = log(max(vals[i],0))
end
using Arpack
temp = svds(lX)[1]
U = temp.U


#making an edge list from incidence matrix
m = size(H, 1)
edgelist = []
for i = 1:m
    edge = findall(x -> x > 0, H[i, :])
    push!(edgelist, edge)
end


gr()

x = U[:,2]
y = U[:,3]


labels = Int.(EdgeLabels)
ulabels = unique(labels)
color_palette = distinguishable_colors(length(ulabels),[RGB(1,1,1), RGB(0,0,0)], dropseed=true);
colorstouse = map(i->color_palette[labels[i]],1:length(labels))

#labels are from 1:10
lab = 6
ids = clusters[lab][2]
scatter!(x[ids],y[ids],markercolor="blue",
alpha=0.5,markersize=4,legend=:bottomright,
colorbar=false, markerstrokewidth=0,framestyle=:none)
#label = "$(clusters[7][3])")



plot!(size=(800,800),dpi = 300, legend = false)

Plots.savefig("dawn.png")

@show clusters

