# Log-PageRank
This repository offers the codes for the following paper, 

>A flexible PageRank-based graph embedding framework closely related to spectral eigenvector embeddings

## Experiments on Graphs
The graph embeddings folder hosts the codes for all the experiments on graphs. 
- The various graphs used in the paper can be generated by running 
`diff_graphs.jl` and the "log PR" graph in the introduction can be obtained by running `introduction.jl` 
- The distance effect showed in `figure 2` in the paper can be generated using `distance_effect.jl` 
- `Figures 4-7` can be generated using `graph_embeddings.jl` 
- The error variance table in `figure 8` can be generated using `error_variance.jl`


## Experiments on Hypergraphs

- Primary-school-contact: Raw data was obtained from [nveldt/HyperModularity.jl](https://github.com/nveldt/HyperModularity.jl/tree/master/data/contact-primary-school-classes); `read_data.jl` was used to generate the `primary_school.mat` file which was used in `smalltest.jl` to generate `figure9a`
- Yelp: Instructions in [LHQD/yelp](https://github.com/MengLiuPurdue/LHQD/tree/main/yelp_local_algorithms) were followed to download the data; `read_data.jl` and `yelp_restaurant_hypergraph.jl` was used to generate the hypergraph and store it at `yelp_restaurant_hypergraph.mat` which was used in `yelp_experiment.jl` to generate `figure 9b`
- Drug Abuse Network (DAWN): Raw data was obtained from [DAWN dataset](https://github.com/nveldt/CategoricalEdgeClustering/tree/master/data/DAWN); `read_data.jl` was used to generate the `dawn.mat` file which was used in `smalltest.jl` to generate `figure 10a`
- Walmart: Raw data was obtained from [Walmart-Trips](https://www.cs.cornell.edu/~arb/data/walmart-trips/); `read_data.jl` was used to generate the `walmart.mat` file which was used in `walmart_experiment.jl` to generate `figure 10b`
- Math Overflow: Raw data was obtained from [mathoverflow-answers](https://github.com/nveldt/HyperModularity.jl/tree/master/data/mathoverflow-answers); `read-data.jl` was used to generate the `mathoverflow.mat` file which was used in `smalltest.jl` to generate `figure 10c`





