from net import GraphEncoder

if __name__ == "__main__":
    model = GraphEncoder.GAT(
        num_of_layers=2, num_heads_per_layer=1, num_features_per_layer=1
    )
