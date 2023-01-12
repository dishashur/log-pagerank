include("../common.jl")
include("../local-hyper.jl")

using MAT
using SparseArrays

data_path = homedir()*"/log-pagerank/data/yelp_dataset/"
M = matread(data_path*"yelp_restaurant_hypergraph.mat")
H = M["H"]
Ht = sparse(H')
order = round.(Int64,vec(sum(H,dims=2)))
d = vec(sum(H,dims=1))
volA = sum(d)
m,n = size(H)

# LH parameters
ratio = 0.01
max_iters = 1000000
x_eps=1.0e-8
aux_eps=1.0e-8
rho=0.5
q= 2.0
kappa_lh = 0.000025

# shared parameters
delta = 5000.0
gamma = 1.0
G = LH.HyperGraphAndDegrees(H,Ht,delta,d,order)

# Run LH

trials = 1000 
seednum = 1
ord = randperm(n)

P = spzeros(n, trials)

for i = 1:trials
    seeds = [ord[i]] 
    L = LH.loss_type(q,delta)
    tic = time()
    x,r,iter = LH.lh_diffusion(G,seeds,gamma,kappa_lh,rho,L,max_iters=max_iters,x_eps=x_eps,aux_eps=aux_eps)
    cond,cluster = hyper_sweepcut(G.H,x,G.deg,G.delta,0.0,G.order)
    toc = time()-tic
    println("time")
    println(toc)
    println("nnz=$(sum( x.> 1e-9))")
    P[:, i] = x
end

#elementwise log
minelem = minimum(nonzeros(P))
lP = log(minelem) .* sparse(ones(size(P,1),size(P,2)));
(u,v,vals) = findnz(P)
for i=1:length(u)
    lP[u[i],v[i]]=log(vals[i])
end

U, S, V = svd(Matrix(lP))

#save results
matwrite(data_path*"yelp_restaurant_trial_$trials.mat", Dict("P"=>P, "U"=>U))