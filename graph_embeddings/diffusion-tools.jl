module DiffusionTools

using SparseArrays
import MatrixNetworks._symeigs_smallest_arpack
using Printf
using LinearAlgebra
import LinearAlgebra.checksquare
using Plots

function _compute_eigen(A::SparseMatrixCSC,d::Vector,nev::Int,
        tol=1e-12,maxiter=1000,dense::Int=96)

    n = checksquare(A)
    if n == 1
        X = zeros(typeof(one(eltype(d))/one(eltype(d))),1,nev)
        lams = [0.0]
    elseif n <= dense
        ai,aj,av = findnz(A)
        L = sparse(ai,aj,-av./((d[ai].*d[aj])),n,n) # applied sqrt above
        L = Matrix(L) + 2I
        F = eigen!(Symmetric(L))
        lams = F.values.-1.0
        X = F.vectors
    else # modifying this branch 1-11-2019
        ai,aj,av = findnz(A)
        L = sparse(ai,aj,-av./((d[ai].*d[aj])),n,n) # applied sqrt above
        L = L + sparse(2.0I,n,n) # Rich Lehoucq suggested this idea to
                                 # make ARPACKs internal tolerance better scaled.

        (lams,X,nconv) = _symeigs_smallest_arpack(L,nev,tol,maxiter,d)
        lams = lams.-1.0
    end
    X = X[:,sortperm(lams)] # sort in ascending order
    return X, lams
end


""" `spectral_embedding(A,k)`

Get a spectral embedding of a sparse matrix A that represents a graph.

This handles small matrices by using LAPACK and large sparse matrices with
ARPACK. Given a sparse matrix \$ A \$, this returns the smallest eigenspace
of the generalized eigenvalue problem min x'*L*x/x'*D*x where L is the
Laplacian and D is the degree matrix.  The sign of the eigenspace
is based on the vertex with maximum degree.

## Inputs
- `A::SparseMatrix` A sparse matrix that represents the graph data.
- `k::Int` the number of eigenvectors to compute
## Outputs
- `X::Matrix` A k-column marix where each column are the eigenvectors and X[:,1]
  is the standard null-space vector. The sign of each column is picked by choosing
  nodes with maximum degree to have positive signs.
## Optional inputs
- `normalize::Bool=true` produce the degree-normalized generalized eigenvectors of
      D^{-1} L (normalize=true) instead of the normalized
      Laplacian D^{-1/2} L D^{-1/2} (normalize=false)
## Uncommon Inputs (where defaults should be handled with care.)
- `dense::Int` the threshold for a dense (LAPACK) computation
- `checksym::Bool` A switch to turn off symmetry checking (don't do this)
- `tol::Real` A real-valued tolerance to give to ARPACK
- `maxiter::Int` The maximum number of iterations for ARPACK
"""
function spectral_embedding(A::SparseMatrixCSC{V,Int},k::Int;
        tol=1e-12,maxiter=10000,dense::Int=96,checksym=true,
        normalize::Bool=true) where {V <: Real}
# ### History
# This code is from the GLANCE package, based on code from the
# MatrixNetworks.jl package, based on a Matlab spectral clustering code.

    n = checksquare(A)
    if checksym
        if !issymmetric(A)
            throw(ArgumentError("The input matrix must be symmetric."))
        end
    end

    nev = k
    d = vec(sum(A,dims=1))
    d = sqrt.(d)

    @show maxiter
    X,lams = _compute_eigen(A,d,nev,tol,maxiter,dense)

    x1err = norm(X[:,1]*sign(X[1,1]) - d/norm(d))
    if x1err >= sqrt(tol)
        s = @sprintf("""
        the null-space vector associated with the normalized Laplacian
        was computed inaccurately (diff=%.3e); the Fiedler vector is
        probably wrong or the graph is disconnected""",x1err)
        @warn s
    end

    vdmax = argmax(d)  # the vertex of maximum degree
    X ./= repeat(d,1,size(X,2))
    X .*= repeat(sign.(X[vdmax,:])',size(X,1),1)
    return X, lams
end


""" `local_spectral_embedding(A,S,k)`

Get a local spectral embedding of a sparse matrix A that represents a graph.

A local spectral embedding is spectral embedding of a subset of the Laplacian
matrix corresponding to vertices in a set S. This does not have a null-vector
like a standard Laplacian vector does.  For a large sparse matrix,
this does not form the entire graph.

### Reference

@Article{Chung-2007-local-cuts,
  author    = {Fan Chung},
  title     = {Random walks and local cuts in graphs},
  journal   = {Linear Algebra and its Applications},
  year      = {2007},
  volume    = {423},
  number    = {1},
  pages     = {22 - 32},
  doi       = {10.1016/j.laa.2006.07.018},
}

This handles small matrices by using LAPACK and large sparse matrices with
ARPACK. Given a sparse matrix \$ A \$, this returns the smallest eigenspace
of the generalized eigenvalue problem min x'*L*x/x'*D*x where L is the
Laplacian and D is the degree matrix.  The sign of the eigenspace
is based on the vertex with maximum degree.

## Inputs
- `A::SparseMatrix` A sparse matrix that represents the graph data.
- `S::Vector` A set of vertices for the local computation.
- `k::Int` the number of eigenvectors to compute
## Outputs
- `X::Matrix` A k-column marix where each column are the eigenvectors and X[:,1]
  is the standard null-space vector. The sign of each column is picked by choosing
  nodes with maximum degree to have positive signs.
## Optional inputs
- `normalize::Bool=true` produce the degree-normalized generalized eigenvectors of
      D^{-1} L (normalize=true) instead of the normalized
      Laplacian D^{-1/2} L D^{-1/2} (normalize=false)
## Uncommon Inputs (where defaults should be handled with care.)
- `dense::Int` the threshold for a dense (LAPACK) computation
- `checksym::Bool` A switch to turn off symmetry checking (don't do this)
- `tol::Real` A real-valued tolerance to give to ARPACK
- `maxiter::Int` The maximum number of iterations for ARPACK
"""
function local_spectral_embedding(Af::SparseMatrixCSC{V,Int},S::Vector{Int},k::Int;
        tol=1e-12,maxiter=300,dense::Int=96,checksym=true,
        normalize::Bool=true) where V <: Real
# ### History
# This code is from the GLANCE package, based on code from the
# MatrixNetworks.jl package, based on a Matlab spectral clustering code.

    n = checksquare(Af)
    ns = length(S)
    Ac = Af[:,S] #
    d = vec(sum(Ac,dims=1)) # get the full degrees...
    A = Ac[S,:] # extract the submatrix

    n = checksquare(A)

    if checksym
        if !issymmetric(A)
            throw(ArgumentError("The input matrix must be symmetric."))
        end
    end

    nev = k
    d = sqrt.(d)

    X,lams = _compute_eigen(A,d,nev,tol,maxiter,dense)

    vdmax = argmax(d)  # the vertex of amximum degree
    X ./= repeat(d,1,size(X,2))
    X .*= repeat(sign.(X[vdmax,:])',size(X,1),1)

    return X,lams
end

function cycle_graph(n::Int)
  A = sparse(1:n-1,2:n,1,n,n)
  A[1,end] = 1
  A = max.(A,A')
  return A
end

function draw_graph(A::SparseMatrixCSC, xy; kwargs...)
    ei,ej = findnz(triu(A,1))[1:2]
    # find the line segments
    lx = zeros(0)
    ly = zeros(0)
    for nz=1:length(ei)
        src = ei[nz]
        dst = ej[nz]
        push!(lx, xy[src,1])
        push!(lx, xy[dst,1])
        push!(lx, Inf)

        push!(ly, xy[src,2])
        push!(ly, xy[dst,2])
        push!(ly, Inf)
    end
    plot(lx,ly;
        kwargs...)
end


export cycle_graph, spectral_embedding, local_spectral_embedding

end # end module

using Test
@testset "DiffusionTools" begin
    n = 200
    A = DiffusionTools.cycle_graph(n)
    @test_nowarn X,lams = DiffusionTools.spectral_embedding(A,2)

    @test_nowarn X,lams = DiffusionTools.local_spectral_embedding(A,collect(1:2:n),2)
end
