function read_hypergraph_label_names()
    names = String[]
    open("label-names-contact-primary-school-classes.txt") do f
        for line in eachline(f)
            push!(names, line)
        end
    end
    return names
end

function read_hypergraph_labels()
    labels = Int64[]
    open("node-labels-contact-primary-school-classes.txt") do f
        for line in eachline(f)
            push!(labels, parse(Int64, line))
        end
    end
    return labels
end


using SparseArrays

function read_hypergraph(maxsize::Int64=25, minsize::Int64=2)
    edgelist = []
    nodelabels = read_hypergraph_labels()
    labelnames = read_hypergraph_label_names()
    n = length(nodelabels)
    open("hyperedges-contact-primary-school-classes.txt") do f
        for line in eachline(f)
            edge = [parse(Int64, v) for v in split(line, ',')]
            push!(edgelist,edge)
        end
     end
     H = spzeros(length(edgelist),n)
     for i=1:length(edgelist)
      for nd in edgelist[i]
        H[i,nd] = 1
      end
     end
    return H
end

labelnames = read_hypergraph_label_names()

nodelabels = read_hypergraph_labels()

H = read_hypergraph()

using MAT
matwrite("primary_school.mat", Dict("labelnames" => labelnames,"nodelabels" => nodelabels, "H" => H))
