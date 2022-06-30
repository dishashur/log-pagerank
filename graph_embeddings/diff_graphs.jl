#code for generating the different graph used in the paper
include("diffusion-tools.jl")
using SparseArrays
using MatrixNetworks
using Plots
using LinearAlgebra
using DelimitedFiles
using Statistics
using Random
using LinearAlgebra
using MAT

#minnesota network
M = load_matrix_network_metadata("minnesota")
A = M[1]
xy = M[2]
G,temp = largest_component(A)
k=0
newxy = zeros(sum(temp),2)
for i=1:size(xy,1)
  if temp[i]==1
    k=k+1
    newxy[k,:]=xy[i,:]
  end
end


#tapir
G=load_matrix_network("tapir")


#nearest neighbour graphs
using Random, NearestNeighbors, SparseArrays
function gnk(n,k;dims::Integer=2)
  xy = rand(dims,n)
  T = BallTree(xy)
  idxs = knn(T, xy, k)[1]
  # form the edges for sparse
  ei = Int[]
  ej = Int[]
  for i=1:n
    for j=idxs[i]
      if i != j
        push!(ei,i)
        push!(ej,j)
      end
    end
  end
  return copy(xy'), sparse(ei,ej,1,n,n) |> A -> max.(A,A')
end
Random.seed!(0)
xy, G = gnk(10000,6;dims=2)


#topo-paper-logPR text
#letter.smat can be geenrated using introduction.jl
G = MatrixNetworks.readSMAT("letter.smat")
xy = readdlm("letter.xy")


#sbm
function sbm(m,k,p,q)
  n = m*k
  A = sprand(Bool,n,n,q)
  offset = 1
  sets = Vector{Vector{Int}}()
  for i=1:k
    A[offset:(offset+m-1), offset:(offset+m-1)] = sprand(Bool,m,m,p)
    push!(sets, offset:(offset+m-1))
    offset += m
  end
  A = dropzeros!(triu(A,1))
  A = float(A + A')
  return A
end
Random.seed!(0)
G = sbm(60,50,0.25,0.001)  


