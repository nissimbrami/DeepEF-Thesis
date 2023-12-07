import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn.functional import softplus

from supervised_model.utils import load_config

# COL_WT_FOLDED = 0
# COL_WT_UNFOLDED = 1
# COL_MUT_FOLDED = 2
# COL_MUT_UNFOLDED = 3

config = load_config()

COL_FOLDED = 0
COL_UNFOLDED = 1


def mse_folded_unfolded(energies, experiment_delta_gs, structure):
    return nn.MSELoss()(energies[COL_UNFOLDED] - energies[COL_FOLDED], experiment_delta_gs)


def energy_by_structure(energies, experiment_delta_gs, structure):
    energy_mutated_folded = energies[COL_FOLDED]
    grad_energy_wrt_structure = torch.autograd.grad(outputs=energy_mutated_folded, inputs=structure,
                                                    grad_outputs=torch.ones_like(energy_mutated_folded),
                                                    create_graph=True)[0]
    grad_energy_wrt_structure_normalized = 0.5 * torch.norm(grad_energy_wrt_structure, p=2) ** 2

    return 2 / (1 + torch.exp(-grad_energy_wrt_structure_normalized)) - 1


loss_dict = {
    'mse_folded_unfolded': mse_folded_unfolded,
    # 'thermodynamic_cycle': thermodynamic_cycle,
    'energy_by_structure': energy_by_structure
}


def criterion(energies, experiment_delta_gs, structure):
    losses = {}
    for crit in config.criteria:
        losses[crit] = loss_dict[crit](energies, experiment_delta_gs, structure)
    return losses


if __name__ == '__main__':
    print(criterion(torch.rand([100, 2], requires_grad=True),
                    torch.rand([58, 4, 3], requires_grad=True))
          )
