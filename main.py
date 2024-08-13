from TorchRL.config import config_dict
from TorchRL.Trainer import Trainer

trainer = Trainer(config_dict)
trainer.train()
trainer.plot_logs()
