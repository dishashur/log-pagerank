function read_hypergraph_label_names()
    names = String[]
    open("hyperedge-label-identities.txt") do f
        for line in eachline(f)
            push!(names, line)
        end
    end
    return names
end

function read_hypergraph_labels()
    labels = Int64[]
    open("hyperedge-labels.txt") do f
        for line in eachline(f)
            push!(labels, parse(Int64, line))
        end
    end
    return labels
end


using SparseArrays
function read_hypergraph(maxsize::Int64=10000, minsize::Int64=3)
    edgelist = []
    edgelabels = read_hypergraph_labels()
    labelnames = read_hypergraph_label_names()
    n = 0
    open("hyperedges.txt") do f
        for line in eachline(f)
            edge = [parse(Int64, v) for v in split(line, '\t')]
            push!(edgelist,edge)
                if maximum(edge) > n
                    n = maximum(edge)
                end
        end
     end
     num_e = length(edgelist)
     H = spzeros(num_e,n)
     for i=1:num_e
      for nd in edgelist[i]
        H[i,nd] = 1
      end
     end
    return H
end

labelnames = read_hypergraph_label_names()

edgelabels = read_hypergraph_labels()

H = read_hypergraph()

using MAT
matwrite("dawn.mat", Dict("labelnames" => labelnames,"edgelabels" => nodelabels, "H" => H))
