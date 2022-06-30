#code for producing the low dimensional graph embeddings in figures 4, 5, 6 and 7
include("diffusion-tools.jl")
using SparseArrays
using MatrixNetworks
using Plots
using LinearAlgebra
using Statistics
using DelimitedFiles
using Statistics
using Random


Z,lams = DiffusionTools.spectral_embedding(G, 7)
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
  return X
end
using Random

Random.seed!(0)
X = make_pagerank_samples_bigalpha(G;N=200,alpha=0.99999)
X = max.(0,X)
for i=1:size(X,2)
  X[:,i] = X[:,i] ./ sum(X[:,i])
end
lX = log.(X)
minelem =  minimum(filter(isfinite,lX))
lX[.!isfinite.(lX)] .= minelem
U,S,V = svd(lX)
P,Q,R = svd(X)
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
P[:,2:3] = align_signs(P[:,2:3],Z[:,2:3])

DiffusionTools.draw_graph(G,Z[:,2:3]; size=(2000,1000), linewidth=5.5,
  framestyle=:none, linecolor=:black, linealpha=0.9, legend=false,
  axis_buffer=0.02)
plot!(dpi=300, legend=:false)


DiffusionTools.draw_graph(G,P[:,2:3]; size=(2000,1000), linewidth=5.5,
    framestyle=:none, linecolor=:black, linealpha=0.9, legend=false,
    axis_buffer=0.02)
  plot!(dpi=300, legend=:false)

DiffusionTools.draw_graph(G,U[:,2:3]; size=(2000,1000), linewidth=5.5,
    framestyle=:none, linecolor=:black, linealpha=0.9, legend=false,
    axis_buffer=0.02)
plot!(dpi=300, legend=:false)

scatter(Z[:,2],U[:,2],xlabel="Spectral2",ylabel="PageRank2",
xguidefontsize=25, yguidefontsize=25, ticks = false, size=(400,400))
plot!(dpi=300, legend=:false)

scatter(Z[:,3],U[:,3],xlabel="Spectral3",ylabel="PageRank3",
xguidefontsize=25, yguidefontsize=25, ticks = false, size=(400,400))
plot!(dpi=300, legend=:false)

#sim_score
Dih = Diagonal(1.0 ./ sqrt.(vec(sum(G,dims=2))))
L = I-(Dih*G*Dih)
p = (U[:,2]'*L*U[:,2])/(U[:,2]'*U[:,2])
s = (Z[:,2]'*L*Z[:,2])/(Z[:,2]'*Z[:,2])
@show diff = abs(s-p)/s





#a has the 2nd smallest eigenvalue from
#chain graphs of sizes 30, 300, 3k
a = [0.005862, 5.5198e-5, 5.48672e-7]
epsi = 1 .- a
x = [0.99, 0.999, 0.9999, 0.99999, 0.999999]
y = (1 .- x[4]) ./ (1 .- (x[4].*epsi))
plot(epsi,y,label="4",xlabel="epsi",
 ylabel="(1-alpha)/1-(alpha*eps)",legend=:topleft,dpi=300)

##experiment 5 - log helps in better reconstruction
Z,lams = DiffusionTools.spectral_embedding(G, 5)
Di = Diagonal(1.0./vec(sum(G,dims=2)))
D = Diagonal(vec(sum(G,dims=2)))
symL = I-(Di.^(1/2))*G*(Di.^(1/2))

tp = 0.99
function make_pagerank_samples_bigalpha(G;N::Int=1000,alpha::Float64=0.999)
  F = lu(I-alpha*Diagonal(1.0./vec(sum(G,dims=2)))*G)
  X = zeros(size(G,1),N)
  for i=1:N
    si = rand(1:size(G,1))
    v = zeros(size(G,1))
    v[si] = 1
    X[:,i] = (1-alpha)*(F\v)
  end
  return X
end
Random.seed!(0)
Xactual = make_pagerank_samples_bigalpha(G;N=200,alpha=tp)
u,s,v = svd(Xactual)

##experiment 6 - other functions like log
Z,lams = DiffusionTools.spectral_embedding(G, 5)

tp = 0.99
function make_pagerank_samples_bigalpha(G;N::Int=1000,alpha::Float64=0.999)
  F = lu(I-alpha*G*Diagonal(1.0./vec(sum(G,dims=2))))
  X = zeros(size(G,1),N)
  for i=1:N
    si = rand(1:size(G,1))
    v = zeros(size(G,1))
    v[si] = 1
    X[:,i] = (1-alpha)*(F\v)
  end
  return X
end
Random.seed!(0)
Xactual = make_pagerank_samples_bigalpha(G;N=200,alpha=tp)
u,s,v = svd(Xactual)

lXact = -exp.(-Xactual)# take the log
minelem =  minimum(filter(isfinite,lXact))
lXact[.!isfinite.(lXact)] .= minelem # replace all “-Inf” with ~-142.15
Ut,St,Vt = svd(lXact)

##experiment 7 - low powers of Z imply low power of X with low alpha
alphas = [0.99, 0.999, 0.9999, 0.99999]
tp = alphas[1]
Z,lams = DiffusionTools.spectral_embedding(G, 5)
Di = Diagonal(1.0./vec(sum(G,dims=2)))
D = Diagonal(vec(sum(G,dims=2)))
m = D*Z*(tp*(I-Diagonal(lams)))*Z'
ei = zeros(size(L,1))
ei[5]=1
ele = m*ei
for k=1:1000
  ele = (m*ele)
  ele = ele ./ norm(ele)
end

##plots for chain formulation explains it
#plot 1
tp = 0.99
plus = (1+sqrt(1-tp^2))/tp
i = [30, 300, 3000, 5000]
y = sqrt((1-tp)/(1+tp))*(plus.^(-i))
plot(i,log.(y),xlabel = "num nodes", ylabel="log.(chain)",legend=false)
title!("Chain vs node size for alpha=0.99")

#plot 2
tp = [0.99, 0.999, 0.9999, 0.99999]
plus = (1 .+ sqrt.(1 .- (tp.^2))) ./ tp
y = sqrt.((1 .- tp)./(1 .+ tp)).*(plus.^(-3000))
plot(tp, log.(y),xlabel = "alpha", ylabel="log.(chain)",legend=false)
title!("3000 node Chain vs alpha")

##expt to verify large powers
tp = 0.9
function make_pagerank_samples_bigalpha(G;N::Int=1000,alpha::Float64=0.999)
  F = lu(I-alpha*G*Diagonal(1.0./vec(sum(G,dims=2))))
  X = zeros(size(G,1),N)
  for i=1:N
    si = i #rand(1:size(G,1))
    v = zeros(size(G,1))
    v[si] = 1
    X[:,i] = (1-alpha)*(F\v)
  end
  return X
end
Random.seed!(0)
X = make_pagerank_samples_bigalpha(G;N=200,alpha=tp)
u,s,v = svd(X);
Z,lams = DiffusionTools.spectral_embedding(G,5)
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
u[:,2:3] = align_signs(u[:,2:3],Z[:,2:3])
epsi = 1 .- lams
M = Diagonal((1-tp) ./ (1 .- (tp*epsi)))
D = Diagonal(vec(sum(G,dims=2)))


Xapr = D*Z*M*Z'
Xapr = Xapr[:,1:200]

Xsv = D*Z*M*Z'*Z*M*Z'*D
mkm = M*Z'*Z*M


#verif
DiffusionTools.draw_graph(G,u[:,2:3]; size=(2000,1000), linewidth=4.0,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)


DiffusionTools.draw_graph(G,Z[:,2:3]; size=(2000,1000), linewidth=4.0,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)

#doesn't explain with normalization
#explains with normalizing
tp = 0.9
Di = Diagonal(1.0./vec(sum(G,dims=2)))
P = G*Di
vk = zeros(size(G,1))
vk[5] = 1.0
temp = P*vk
for i=1:1000
  temp = P*temp
  normalize!(temp)
end
@show temp[1792]

temp[iszero.(temp)] .= 1000.0
@show minimum(temp),argmin(temp)

##cycle graph appx

tp = 0.999999
function make_pagerank_samples_bigalpha(G;N::Int=1000,alpha::Float64=0.999)
  F = lu(I-alpha*Diagonal(1.0./vec(sum(G,dims=2)))*G)
  X = zeros(size(G,1),N)
  seed_nodes = zeros(N)
  for i=1:N
    si = rand(1:size(G,1))
    @show i,si
    v = zeros(size(G,1))
    v[si] = 1
    X[:,i] = (1-alpha)*(F\v)
  end
  return X
end
using Random
Random.seed!(0)
X = make_pagerank_samples_bigalpha(G;N=1,alpha=tp)
#normalizing X
for i =1:size(X,2)
  X[:,i] = X[:,i]./sum(X[:,i])
end
lX = log.(X)

y = zeros(size(X)) #w the approx
plus = (1+sqrt(1-tp^2))/tp
minus = (1-sqrt(1-tp^2))/tp

function closedform(tp,seed_nodes,n)
  for j=1:size(X,2)
    k = Int(seed_nodes[j])
    y[1,j] = sqrt((1-tp)/(1+tp))/(plus^(k-1))
    y[n,j] = sqrt((1-tp)/(1+tp))/(plus^(n-k))
    y[k,j] = sqrt((1-tp)/(1+tp))*tp/plus
    for i=2:k-1
      y[i,j] = sqrt((1-tp)/(1+tp))/(plus^min(abs(n-(k-i)),abs(i-k)))
    end
    for i=k+1:n-1
      y[i,j] = sqrt((1-tp)/(1+tp))/(plus^min(abs(n-(k-i)),abs(i-k)))
    end
  end
  return y
end

y = closedform(tp,[4],n)
for i =1:size(y,2)
  y[:,i] = y[:,i]./sum(y[:,i])
end
ly = log.(y)

#plotting for comparision
case = 3 #max N=100 #check what si was at i=3
toplot=hcat(lX[:,case],ly[:,case])
plot(toplot,label=["original" "w approx"])
title!("Comparision of the 4 PPRs seeded at 275")
