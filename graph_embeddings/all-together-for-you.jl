
## Keep a list of packages we will need.
#=
using Pkg; Pkg.add.(["Images","FileIO","Downloads","HTTP", "CairoMakie",
    "Statistics", "StatsBase", "NearestNeighbors",
    "GLMakie", "MatrixNetworks",
    "VoronoiDelaunay", "GeometricalPredicates"]);
=#
using Images, FileIO, Downloads, HTTP
function get_image(words::AbstractString)
    io = IOBuffer()
    words = HTTP.escapeuri(words)
    url = "https://latex.codecogs.com/png.latex?%5Cdpi%7B600%7D%20%5Cfn_phv%20%5Chuge%20%5Cmbox%7B%5Csffamily%5Cbfseries%20$(words)%7D"
    Downloads.download(url, io) # download into an in-memory buffer IO
    return load(Stream{format"PNG"}(io)) # load the buffer as a PNG
end 

using Statistics, StatsBase
"""Generate a series of x,y points based on the values in a matrix.
If the values are high, we will generate those x,y values more often then
when the values are low. The return value is the set of x,y points."""
function sample_from_matrix(A::Matrix, n::Integer)
    xs = zeros(0)
    ys = zeros(0)
    w = weights(A)
    inds = zeros(Int,0)
    map = CartesianIndices(A) # this converts linear to cartesian indices!
    for i=1:n
        p = sample(w) # these are linear coordinates, not x,y values :(
        push!(inds, p) # push the linear indices
        p2d = map[p]
        px = p2d[1]
        py = p2d[2]
        push!(xs, px)
        push!(ys, py)
    end
    return xs, ys, inds
end

## This looks good, but ... it'd be nice to have the "image go outside" the text...
function pad_and_center(A::Matrix; xpad::Real=0.25, ypad::Real=0.25, fillval=zero(eltype(A)))
  # remember x and y are flipped with respect to rows and columns...
  newxsize = ceil(Int,size(A,2)*(1+xpad))
  newysize = ceil(Int,size(A,1)*(1+ypad))
  B = fill!(similar(A, newysize, newxsize), fillval)
  offx = div(newxsize - size(A,2),2) # use integer division
  offy = div(newysize - size(A,1),2)
  B[offy:offy+size(A,1)-1,offx:offx+size(A,2)-1] .= A
  return B
end

using NearestNeighbors
""" Build a list of edges based on a set of x, y points. Two points will be connected
if they are within the `k` nearest neighbors or within relative radius `relradius`.
    The relative radius is just relradius*(max(max(xcoord)-min(xcoord),max(ycoord)-min(ycoord)))
"""
function nearest_neighbor_edges(x, y; k::Int = 15, relradius::Float64 = 0.05)
    xmin,xmax = extrema(x)
    ymin,ymax = extrema(y)
    r = max(ymax-ymin, xmax-xmin)*relradius
    points = copy([x y]') # make a hard copy to avoid adjoint
    T = BallTree(points)

    # drop duplicates
    edges = Set{Tuple{Int,Int}}()

    # edges ...
    src = zeros(Int,0)
    dst = zeros(Int,0)

    if k > 0 # find the nearest neighbors
        idxs, dists = knn(T, points, k+1)
        for i=1:length(x) # for each point...
            neighs = idxs[i] # the points are indexed the same way, so i is shared...
            for j in neighs # for each neighbor
                if i > j # only include one end-point
                    push!(src, i)
                    push!(dst, j)
                    push!(edges, (i,j))
                end
            end
        end
    end

    if r > 0 # find radius points
        idxs = inrange(T, points, r, false)
        for i=1:length(x) # for each point...
            neighs = idxs[i] # the points are indexed the same way, so i is shared...
            for j in neighs # for each neighbor
                if i > j # only include one end-point
                    if !((i,j) in edges)
                        push!(src, i)
                        push!(dst, j)
                    end
                end
            end
        end
    end
    return src, dst
end


function data_to_plot_lines(x, y, src, dst)
  px = zeros(eltype(x),0)
  py = zeros(eltype(y),0)

  for (s,d) in zip(src,dst)
      push!(px, x[s])
      push!(px, x[d])
      push!(px, NaN)

      push!(py, y[s])
      push!(py, y[d])
      push!(py, NaN)
  end
  return px, py
end

using Makie
empty_theme = Theme(
    Axis = (
        backgroundcolor = :transparent,
        leftspinevisible = false,
        rightspinevisible = false,
        bottomspinevisible = false,
        topspinevisible = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
        xminorticksvisible = false,
        yminorticksvisible = false,
        xticksvisible = false,
        yticksvisible = false,
        xautolimitmargin = (0.0,0.0),
        yautolimitmargin = (0.0,0.0),
    )
)


using SparseArrays
function edges_to_network(src, dst)
  n = max(maximum(src),maximum(dst))
  A = sparse(src, dst, 1.0, n, n)
  A = max.(A,A')  # make undirected graph
  fill!(A.nzval, 1.0) # set all to zero
  return A
end
##
using GeometricalPredicates, VoronoiDelaunay

struct IndexedPoint2D <: GeometricalPredicates.AbstractPoint2D
    _x::Float64
    _y::Float64
    _ind::Int
end
GeometricalPredicates.getx(p::IndexedPoint2D) = p._x
GeometricalPredicates.gety(p::IndexedPoint2D) = p._y
IndexedPoint2D(x, y) = IndexedPoint2D(x, y, 0)

function edges_from_tess_of_xy(x, y)
    # write a rescale to scale coordinates to [1,2]
    xmin,xmax = extrema(x)
    ymin,ymax = extrema(y)
    newscale = max((xmax-xmin),(ymax-ymin))
    rescale = p -> begin # think of p as a coordinate x, or y
        pmin, pmax = extrema(p)
        return ( p .- pmin ) ./(1.01*newscale) .+ 1.001
    end
    x = rescale(x)
    y = rescale(y)
    #@show extrema(x), extrema(y)
    pts = IndexedPoint2D[ IndexedPoint2D(px,py,i) for
                                (i,(px,py)) in enumerate(zip(x,y)) ]
    tess = DelaunayTessellation2D{IndexedPoint2D}(length(pts))
    push!(tess, pts)

    src,dst = Int[], Int[]
    for e in delaunayedges(tess)
        srcp = geta(e) # these are the IndexedPoint2D...
        dstp = getb(e)
        push!(src, srcp._ind)
        push!(dst, dstp._ind)
    end
    return src,dst, tess
end

##
function make_graph(words::AbstractString,
            npts::Integer,
            edgefunction::Function;
            xpad::Real = 0.25,
            ypad::Real = 0.25,
            random_point_param::Real = 0.1, # increase this to make random points more likely
            point_randomness::Real = 0.25, # needs to be non-zero for DelaunayTessellation
            )
  img = get_image(words)
  R = Float64.(alpha.(img))
  M = pad_and_center(R;xpad,ypad)
  x,y,inds = sample_from_matrix(M .+ random_point_param, npts)
  x .+= point_randomness*randn(size(x)...)
  y .+= point_randomness*randn(size(y)...)
  grps = round.(Int, M[inds]) # index into the matrix M to get 0, 1 values...
  src,dst = edgefunction(x, y)
  G = edges_to_network(src, dst)
  return (x=x, y=y, groups=grps, src=src, dst=dst, G=G, M=M)
end


##
function draw_picture(G::NamedTuple; kwargs...)
  lx,ly = data_to_plot_lines(G.x, G.y, G.src, G.dst)
  with_theme(empty_theme) do
    plt = lines(ly,-lx,
        axis=(aspect=DataAspect(),), linewidth=0.75, color=RGBA(0,0,0,0.5),
        figure=(resolution=(1920,1080),))
    Plots.scatter!(G.y, -G.x, markersize=2, dpi=300)
    return plt
  end
end
