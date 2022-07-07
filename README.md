# Log-PageRank
This repository offers the codes for the following paper, 

>A flexible PageRank-based graph embedding framework closely related to spectral eigenvector embeddings

## Experiments on Graphs
The graph embeddings folder hosts the codes for all the experiments on graphs. The various graphs used in the paper can be generated by running 
`diff_graphs.jl` and the "log PR" graph in the introduction can be obtained by running introduction.jl
Similarly, all the codes in this folder are self explanatory.

## Experiments on Hypergraphs
<!--- For the hypergraph emebddings, we offer the codes for contact-primary-school dataset and the yelp dataset. The necessary data is hosted at the
references mentioned in the paper. After obtaining the data, the file read_data.jl should be run first which will produce the .mat file. This file 
is used by smalltest.jl to produce the pictures in the paper. --->

- Yelp: follow [LHQD/yelp](https://github.com/MengLiuPurdue/LHQD/tree/main/yelp_local_algorithms) to download the data, run `read_data.jl` and `yelp_restaurant_hypergraph.jl` to generate the hypergraph and store it at `yelp_restaurant_hypergraph.mat`, run `yelp_experiment.jl` to get log-PageRank embedding and `yelp_plot.jl` to plot the embedding.
- Walmart: download data from [Walmart-Trips](https://www.cs.cornell.edu/~arb/data/walmart-trips/), run `read_data.jl` to preprocess the data and store it at `walmart.mat` format, run `walmart_experiment.jl` to get log-PageRank embedding and `walmart_plot.jl` to plot the embedding.



