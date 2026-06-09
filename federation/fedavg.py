import copy
import torch

class FedAvgServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def broadcast(self, clients):
        for client in clients:
            client.model.load_state_dict(
                copy.deepcopy(self.global_model.state_dict())
            )

    def aggregate(self, clients, client_sizes):
        total = sum(client_sizes)
        global_sd = self.global_model.state_dict()
        aggregated = copy.deepcopy(global_sd)
        for key in aggregated:
            aggregated[key] = torch.zeros_like(global_sd[key], dtype=torch.float32)
        for client, n_k in zip(clients, client_sizes):
            w = n_k / total
            sd = client.model.state_dict()
            for key in aggregated:
                aggregated[key] += w * sd[key].float()
        for key in aggregated:
            orig_dtype = global_sd[key].dtype
            if orig_dtype != torch.float32:
                aggregated[key] = aggregated[key].to(orig_dtype)
        self.global_model.load_state_dict(aggregated)

    def get_global_model(self):
        return self.global_model