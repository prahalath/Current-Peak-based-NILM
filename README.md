# Current-Peak-based-NILM

Developed a classification model for Non Intrusive Load Monitoring (NILM) suitable for running on a low-cost
system. The model was developed and evaluated using BLUED dataset and
Extremely Randomized Trees was used as the classification algorithm with
current peaks as features. The model was finally run on a Raspberry Pi Zero, enabling a low-cost solution for NILM.

The BLUED dataset contains the aggregate voltage and
current measurements sampled at a frequency of 12 kHz
for Phases A and B of a residential household in the US for
a week. Around fifty appliances were monitored and each
state transition in any appliance in the household is termed
as an event.
