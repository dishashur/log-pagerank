#code for visualizing the distance effect creating by log and
#matrix powers of the normalized adjacency

include("diffusion-tools.jl")
using SparseArrays
using MatrixNetworks
using Plots
using LinearAlgebra
using Statistics
using DelimitedFiles
using Statistics
using Random



tp = 0.999
# Pick a random seed
function make_pagerank_samples_bigalpha(G;N::Int=1000,alpha::Float64=0.999)
  F = lu(I-alpha*G*Diagonal(1.0./vec(sum(G,dims=2))))
  X = zeros(size(G,1),N)
  for i=1:N
    si = 3 #rand(1:size(G,1))
    @show si
    v = zeros(size(G,1))
    v[si] = 1
    X[:,i] = (1-alpha)*(F\v)
  end
  return X
end
Random.seed!(0)
X = make_pagerank_samples_bigalpha(G;N=1,alpha=tp)

#making a different colour for seed
X[3] = minimum(X)
DiffusionTools.draw_graph(G,xy[:,1:2]; size=(1000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3,legend=false)
scatter!(xy[:,1],xy[:,2],marker_z=log10.(X),markersize=8,markerstrokewidth=0)
plot!(dpi=300)

tp = 0.999
z = zeros(size(G,1))
Dihalf = Diagonal(1.0./sqrt.(vec(sum(G,dims=1))))
Di = Diagonal(1.0./vec(sum(G,dims=1)))
z[3] = 1
for i=1:1000
  z = normalize!(tp*Dihalf*G*Dihalf*z) #tp*Dihalf*G*Dihalf*z
end
colr = z
colr[3] = minimum(colr)
DiffusionTools.draw_graph(G,Z[:,2:3]; size=(1000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3,legend=false)
scatter!(xy[:,1],xy[:,2],marker_z=colr,markersize=8,markerstrokewidth=0)
plot!(dpi=300)
