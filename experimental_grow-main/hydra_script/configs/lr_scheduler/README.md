# Learning Rate Scheduler Configuration (`lr_scheduler/`)

Defines the learning rate schedule applied during training steps. By default `.step` is called after every epoch, but this can be changed to after every batch by setting `scheduler_step_granularity: batch` in the configuration.

