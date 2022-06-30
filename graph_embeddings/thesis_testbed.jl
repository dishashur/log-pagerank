#testbed for spectral section
## dependencies
include("diffusion-tools.jl")
using SparseArrays
using MatrixNetworks
using Plots
using LinearAlgebra
using Statistics
using DelimitedFiles
using Statistics
using Random


## singular values vs alpha
S=zeros(4,5)
alphas = [0.9, 0.999, 0.9999, 0.99999]
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
using Random

for i=1:size(alphas,1)
  tp = alphas[i]
  Random.seed!(0)
  X = make_pagerank_samples_bigalpha(G;N=200,alpha=tp)
  lX = log.(X)# take the log
  minelem =  minimum(filter(isfinite,lX))
  lX[.!isfinite.(lX)] .= minelem # replace all “-Inf” with ~-142.15
  u,s,v = svd(lX)
  S[i,:]=s[1:5]
end
plot()
for i =1:5
  plot!(log.(1 .- alphas),log.(S[:,i]),label=i,linewidth=2.0,dpi=300,legend=:bottomright)
end
xlabel!("alpha")
ylabel!("log(singular values)")
title!("Tapir")

##showing the rotation
Z,lams = DiffusionTools.spectral_embedding(G, 7)
@show lams[3]-lams[2]
@show 1/size(G,1)
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

plot(scatter(Z[:,2],Z[:,3], marker_z=Z[:,2], title="SPECTRAL"),
scatter(U[:,2]-U[:,3],U[:,2]+U[:,3], marker_z=Z[:,2], title = "PAGERANK"),
dpi=300, size=(1000,500), legend=:false,
framestyle=:none, colorbar=:false)

##intro figures
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
#max(Int(round(0.07*size(G,1))),3)
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
##error variance
using Random
using Statistics
using LinearAlgebra

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
##spectral figures
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
pow_m = normalize!(tp*Dihalf*G*Dihalf)
for i=1:1000
  pow_m = normalize!(tp*Dihalf*G*Dihalf) #tp*Dihalf*G*Dihalf*z
end
z = pow_m*z
colr = z
colr[3] = minimum(colr)
DiffusionTools.draw_graph(G,Z[:,2:3]; size=(1000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3,legend=false)
scatter!(xy[:,1],xy[:,2],marker_z=colr,markersize=8,markerstrokewidth=0)
plot!(dpi=300)
#savefig("100power.png")
##showing how embeddings change
Z,lams = DiffusionTools.spectral_embedding(G, 5)
plot(Z[:,2],Z[:,3],label="spectral")


alphas = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]

tp = alphas[6]
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

lXact = log.(Xactual)# take the log
minelem =  minimum(filter(isfinite,lXact))
lXact[.!isfinite.(lXact)] .= minelem # replace all “-Inf” with ~-142.15
Ut,St,Vt = svd(lXact)

plot(u[:,2],u[:,3],label="u at $(tp)",title=("at 50 nodes"))
plot(Ut[:,2],Ut[:,3],label="log $(tp)",title=("at 50 nodes"))

##experiment number 2 -
alphas = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
tp = alphas[2]

Z,lams = DiffusionTools.spectral_embedding(G, 5)
Di = Diagonal(1.0./vec(sum(G,dims=2)))
D = Diagonal(vec(sum(G,dims=2)))
P = G*Di
ei = zeros(size(L,1),1)
ei[5]=1
ele = L*ei
K=[]
for k=1:100000
  ele = (L*ele)
  ele = ele ./ norm(ele)
  #push!(K,ele)
end
ele = abs.(ele)


#the next three steps might hang your system - mind the risk
#K = hcat(K...)
#K = K[:,49001:50000]
#heatmap(K)

 #heatmaps of PageRank
function make_pagerank_samples_bigalpha(G;N::Int=1000,alpha::Float64=0.999)
  F = lu(I-alpha*Diagonal(1.0./vec(sum(G,dims=2)))*G)
  X = zeros(size(G,1),N)
  for i=1:N
    si = 550 #rand(1:size(G,1))
    v = zeros(size(G,1))
    v[si] = 1
    X[:,i] = (1-alpha)*(F\v)
  end
  return X
end
Random.seed!(0)
Xactual = make_pagerank_samples_bigalpha(G;N=1,alpha=0.999999)

lXact = log.(Xactual)# take the log
minelem =  minimum(filter(isfinite,lXact))
lXact[.!isfinite.(lXact)] .= minelem # replace all “-Inf” with ~-142.15

heatmap(lXact);title!("log(X) at 0.999999")
plot!(dpi=300, legend=:false)
heatmap(Xactual);title!("X at 0.999999")
plot!(dpi=300, legend=:false)
## experiment 3
using LinearAlgebra
Z,lams = DiffusionTools.spectral_embedding(G, 5)
Di = Diagonal(1.0./vec(sum(G,dims=2)))
D = Diagonal(vec(sum(G,dims=2)))
symL = I-(Di.^(1/2))*G*(Di.^(1/2))
Q = eigvecs(Matrix(symL))


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
Xactual = make_pagerank_samples_bigalpha(G;N=200,alpha=0.999999)
u,s,v = svd(Xactual)

lXact = log.(Xactual)# take the log
minelem =  minimum(filter(isfinite,lXact))
lXact[.!isfinite.(lXact)] .= minelem # replace all “-Inf” with ~-142.15
Ut,St,Vt = svd(lXact)

## experiment 4 - M vs alpha for diff graphs

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
