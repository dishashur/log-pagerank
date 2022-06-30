#code for recreating the error variance in Table1

include("diffusion-tools.jl")
using SparseArrays
using MatrixNetworks
using Plots
using LinearAlgebra
using Statistics
using DelimitedFiles
using Statistics
using Random

diffr = zeros(8,3,50)
cols = [0.04, 0.07, 0.1]
function make_pagerank_samples_bigalpha(G;N::Int=1000,alpha::Float64=0.999)
  F = lu(I-alpha*G*Diagonal(1.0./vec(sum(G,dims=2))))
  X = zeros(size(G,1),N)
  seed = zeros(N,1)
  for i=1:N
    si = rand(1:size(G,1))
    seed[i] = si
    v = zeros(size(G,1))
    v[si] = 1
    X[:,i] = (1-alpha)*(F\v)
  end
  return X, seed
end


for ntwrk = 1:8
  if ntwrk == 1
    Random.seed!(0)
    xy, G = gnk(30,6;dims=2)
  elseif ntwrk == 2
    Random.seed!(0)
    xy, G = gnk(3000,6;dims=2)
  elseif ntwrk == 3
    Random.seed!(0)
    xy, G = gnk(10000,6;dims=2)
  elseif ntwrk == 4
    n = 30;
    G = sparse(1:n-1,  2:n, 1, n, n) |> G -> max.(G,G')
  elseif ntwrk == 5
    n = 3000;
    G = sparse(1:n-1,  2:n, 1, n, n) |> G -> max.(G,G')
  elseif ntwrk == 6
    inp = matread("/home/disha/Documents/MS_stuff/MS_Thesis/codes/minnesota.mat")
    A = inp["Problem"]["A"]
    xy = inp["Problem"]["aux"]["coord"]
    G,temp = largest_component(A)
    k=0
    newxy = zeros(sum(temp),2)
    for i=1:size(xy,1)
      if temp[i]==1
        k=k+1
        newxy[k,:]=xy[i,:]
      end
    end
  elseif ntwrk == 7
    G=load_matrix_network("tapir")
    xy = readdlm("/home/disha/Documents/MS_stuff/MS_Thesis/codes/orig_svd/tapir.xy")
  else
    G = MatrixNetworks.readSMAT("/home/disha/Documents/topo_paper/letter.smat")
    xy = readdlm("/home/disha/Documents/topo_paper/letter.xy")
  end
  Z,lams = DiffusionTools.spectral_embedding(G, 7)
  for colsize = 1:3
    Random.seed!(0)
    for err=1:50
      X, seed = make_pagerank_samples_bigalpha(G;N=max(Int(round(cols[colsize]*size(G,1))),3),alpha=0.99)
      X = max.(0,X)
      for i=1:size(X,2)
        X[:,i] = X[:,i] ./ sum(X[:,i])
      end
      lX = log.(X)
      minelem =  minimum(filter(isfinite,lX))
      lX[.!isfinite.(lX)] .= minelem
      U,S,V = svd(lX)
      function align_signs(X,Y)
         # figure out signs
         bestdiff = norm(X .- Y)
         bestsign = [1,1]
         for signs = [[-1,-1],[-1,1],[1,-1]]
           if norm(repeat(signs',size(X,1),1).*X - Y) < bestdiff
             bestsign = signs
           end
         end
         return  repeat(bestsign',size(X,1),1).*X
      end
      U[:,2:3] = align_signs(U[:,2:3],Z[:,2:3])

      #sim_score
      Dih = Diagonal(1.0 ./ sqrt.(vec(sum(G,dims=2))))
      L = I-(Dih*G*Dih)
      p = (U[:,2]'*L*U[:,2])/(U[:,2]'*U[:,2])
      s = (Z[:,2]'*L*Z[:,2])/(Z[:,2]'*Z[:,2])
      diffr[ntwrk,colsize,err] = abs(s-p)/s
    end
  end
end



using JLD
save("err_analysis.jld", "errors", diffr)



@show var(diff)
@show minimum(diff)
@show maximum(diff)
