using DelimitedFiles
using DataStructures
using SparseArrays
using MatrixNetworks
using MAT

data_path = homedir()*"/log-pagerank/hypergraph_embeddings/data/walmart-trips"
code_path = homedir()*"/log-pagerank/hypergraph_embeddings/walmart"

#read node label
V = readdlm(data_path*"/node-labels-walmart-trips.txt", ',', Int64)[:, 1]
n = size(V, 1)

#read edges
Elist = open(data_path*"/hyperedges-walmart-trips.txt", "r") do datafile
    [parse.(Int64, split(line, ',')) for line in eachline(datafile)]
end
m = length(Elist)

#create node-edge matrix
H = spzeros(m, n) 

eid = 0
for e in Elist
    global eid += 1
    for v in e
        H[eid, v] = 1
    end
end 

#read label names
labels = open(data_path*"/label-names-walmart-trips.txt", "r") do datafile
    [line for line in eachline(datafile)]
end

d = vec(sum(H,dims=1))
order = round.(Int64,vec(sum(H,dims=2)))
volA = sum(d)
m,n = size(H)

## remove trivial edges
edges = findall(x->x>1,order)
H = H[edges,:]
order = order[edges]
d = vec(sum(H,dims=1))
volA = sum(d)
m,n = size(H)

## Form bipartite expansion to find largest connected component

A = [spzeros(m,m) H; sparse(H') spzeros(n,n)]
lcc, pcc = largest_component(A)
p_nodes = pcc[m+1:end]
p_edges = pcc[1:m]
H = H[p_edges,p_nodes]

V = V[p_nodes]  

m,n = size(H)
@show m, n

matwrite(data_path*"/walmart.mat", Dict("H"=>H, "labels"=> V, "labelname"=> labels))


