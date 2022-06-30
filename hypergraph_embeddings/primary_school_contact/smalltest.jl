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
max_size = 10000

M = matread("primary_school.mat")
H = M["H"]
NodeLabels = vec(M["nodelabels"])
LabelNames = M["labelnames"]
order = vec(round.(Int64,sum(H,dims=2)))
good = findall(x->x > 0, order)
H = H[good,:]
order = vec(round.(Int64,sum(H,dims=2)))
G = LH.graph(H,1.0) # a hypergraph object with delta=1.0 in its cut function
volA = sum(G.deg)
 @show size(G.deg)
 @show volA
 @show size(G.order)
 node_labels = py_collections.Counter(NodeLabels)
 node_labels = [(key,node_labels[key]) for key in keys(node_labels)]
 node_labels = sort(filter(x->(x[2]>min_size && x[2]<max_size),node_labels),by=x->x[2])
 labels = [x[1] for x in node_labels]
 clusters = Dict()
 for i = 1:length(labels)
       label = labels[i]
       T = findall(x->x ==label,NodeLabels)
       nT =length(T)
       condT, volT, cutT = tl_cond(H,T,G.deg,1.0,volA,G.order)
       println("$label \t $nT \t $condT \t $(LabelNames[label])")
       clusters[label] = (label,T,LabelNames[label],condT)
 end






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
    seednode = [T[rand(1:n)]]
    @time x,r,iter = LH.lh_diffusion(G,seednode,gamma,kappa,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
    push!(X,x)
    end
    x = mean(X)
    lcond,new_cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order,nseeds=seednum)
    end
 return X
end

kappa = 0.0025 # value of kappa (sparsity regularization)
alpha = 0.999
gamma = (1/alpha)-1
@time X = prmatrix(G,clusters,ratio=0.03,gamma = gamma,kappa = kappa)



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


#plotting

gr()

labels = Int.(NodeLabels)
ulabels = unique(labels)
d = Dict(sort(ulabels) .=> 1:length(ulabels))
color_palette = distinguishable_colors(length(ulabels));color_palette[1] = RGB(1,1,1);
colorstouse = map(i->color_palette[d[labels[i]]],1:length(labels))

#plot for 135,5,6
ids = findall(labels .== 135)
scatter!(U[:,2],U[:,3],color=colorstouse[ids],
    alpha=0.6,markersize=8,
    colorbar=false, markerstrokewidth=0,framestyle=:none)
plot!(size=(800,800), dpi = 300, legend = :false)

savefig("primary_school_lab.png")


