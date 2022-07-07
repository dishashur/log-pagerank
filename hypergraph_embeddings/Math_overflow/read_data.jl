function read_hypergraph_label_names()
    names = String[]
    open("label-names-mathoverflow-answers.txt") do f
        for line in eachline(f)
            push!(names, line)
        end
    end
    return names
end

function read_hypergraph_labels()
    labels = Int64[]
    open("node-labels-mathoverflow-answers.txt") do f
        for line in eachline(f)
            push!(labels, parse(Int64, line))
        end
    end
    return labels
end


using SparseArrays
function read_hypergraphmaxsize::Int64=25, minsize::Int64=2)
    edgelist = []
    nodelabels = read_hypergraph_labels()
    labelnames = read_hypergraph_label_names()
    n = length(nodelabels)
    num_e = 0
    open("hyperedges-mathoverflow-answers.txt") do f
        for line in eachline(f)
            edge = [parse(Int64, v) for v in split(line, ',')] 
            global num_e = num_e + 1
            push!(edgelist,edge)
        end
     end
     H = spzeros(num_e,n)
     for i=1:num_e
      if nd in edgelist[i]
        H[i,nd] = 1
      end
     end
    return H
end



using MAT
matwrite("mathoverflow.mat", Dict("labelnames" => labelnames,"nodelabels" => nodelabels, "H" => H))
