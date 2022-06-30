#code to reproduce figure 1
include("all-together-for-you.jl")
include("diffusion-tools.jl")


## Generate the graph
using Random, CairoMakie
using Plots
Random.seed!(0)
edgefun = (x,y) -> nearest_neighbor_edges(x, y; relradius=0.015, k=25)
#P = make_graph("log\nPR", 4500, edgefun)
P = make_graph("\$\\begin{array}{c} \\mbox{log} \\\\ \\mbox{PR} \\end{array}\$", 5000, edges_from_tess_of_xy)
#P = (P..., x=P.y, y=P.x)
fig = draw_picture(P)

##
using MatrixNetworks, Plots
G = P.G
xy = [P.y -P.x]
@assert(MatrixNetworks.is_connected(G))
function writeSMAT(filename::AbstractString, A::SparseMatrixCSC{T,Int};
    values::Bool=true) where T
    open(filename, "w") do outfile
        write(outfile, join((size(A,1), size(A,2), nnz(A)), " "), "\n")

        rows = rowvals(A)
        vals = nonzeros(A)
        m, n = size(A)
        for j = 1:n
           for nzi in nzrange(A, j)
              row = rows[nzi]
              val = vals[nzi]
              if values
                write(outfile, join((row-1, j-1, val), " "), "\n")
              else
                write(outfile, join((row-1, j-1, 1), " "), "\n")
              end
           end
        end
    end
end
using DelimitedFiles
writeSMAT("letter.smat",G)
writedlm("letter.xy", xy)


#for Figure 1a
DiffusionTools.draw_graph(G,Z[:,2:3]; size=(2000,2000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)
  Plots.scatter!(P0, Z[:,2], Z[:,3], marker_z=xy[:,1].+xy[:,2],
  markerstrokewidth=0, linewidth=0)
Plots.plot!(size=(1000,1000), dpi=300, legend=:false)
savefig("1a.png")


#for Figure 1b
Z,lams = DiffusionTools.spectral_embedding(G, 4)

DiffusionTools.draw_graph(G,Z[:,2:3]; size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)
  
  
 #for Figures 1c,d,e
 
 function make_pagerank_samples_bigalpha(G;N::Int=1000,alpha::Float64=0.999)
  F = lu(I-alpha*G*Diagonal(1.0./vec(sum(G,dims=2))))
  X = zeros(size(G,1),N)
  for i=1:N
    si = rand(1:size(G,1))
    v = zeros(size(G,1))
    v[si] = 1
    X[:,i] = (F\v)
  end
  return X
end

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

Random.seed!(0)
X = make_pagerank_samples_bigalpha(G;N=50,alpha=0.999)
U,S,V = svd(log.(X))
U = align_signs(U[:,2:3], Z[:,2:3])
U = U[:,1:2]*Diagonal([-1,-1])
P = DiffusionTools.draw_graph(G,-U[:,1:2]; size=(1000,1000), linewidth=1.5,
    framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
    axis_buffer=0.02)
  Plots.scatter!(P, -U[:,1], -U[:,2], marker_z=xy[:,1].+xy[:,2],
  markerstrokewidth=0, linewidth=0)
 Plots.plot!(size=(1000,1000), dpi=300, legend=:false)
savefig("1e.png")
