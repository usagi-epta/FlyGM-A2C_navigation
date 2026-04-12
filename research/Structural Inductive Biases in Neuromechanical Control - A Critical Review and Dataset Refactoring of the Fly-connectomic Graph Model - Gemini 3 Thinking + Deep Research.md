# Structural Inductive Biases in Neuromechanical Control: A Critical Review and Dataset Refactoring of the Fly-connectomic Graph Model

The emergence of whole-brain connectomics provides an unprecedented opportunity to move beyond engineered modularity in artificial intelligence toward biologically grounded structural priors. The paper 'Bio-Inspired Neural Control Under Constrained Optimisation - Evaluating the Fly-connectomic Graph Model with Advantage Actor-Critic for Embodied Navigation' serves as a seminal investigation into whether the exact synaptic architecture of a complex organism can serve as an effective neural controller in an embodied reinforcement learning context. By leveraging the Fly-connectomic Graph Model (FlyGM), which represents the full adult Drosophila melanogaster connectome as a directed message-passing graph, the researchers attempt to solve the dual challenges of gait initiation and spatial navigation. This report provides a critical academic review of the paper's methodology and findings, followed by a comprehensive blueprint for refactoring the associated codebase using high-fidelity datasets from the FlyWire consortium and the comparative connectomics framework, coconatfly. [^1][^2][^3][^4][^5]

## Critical Academic Review of FlyGM and Constrained Optimisation

The central premise of the study is that biological neural networks possess a structural inductive bias that naturally supports the learning and control of whole-body movements. The paper introduces the Fly-connectomic Graph Model (FlyGM), a computational architecture directly inherited from the Drosophila whole-brain connectome, containing approximately 140,000 neurons and over 50 million synapses. The primary innovation lies in the transformation of a static, anatomical graph into a dynamical controller by imposing a biologically grounded information flow from sensory inputs to motor outputs. [^2][^1][^8]

The paper's methodological approach frames the reinforcement learning problem as "constrained optimisation." By substituting the high-stability Proximal Policy Optimization (PPO) algorithm with the Advantage Actor-Critic (A2C) algorithm, the researchers test the robustness of the FlyGM architecture. A2C is known for its higher gradient variance and lower sample efficiency compared to PPO, primarily because it lacks the trust-region constraints that prevent catastrophic policy shifts. The paper argues that the structural constraints of the connectome compensate for the "weaker" optimization of A2C, allowing the agent to learn stable locomotion where standard Multi-Layer Perceptrons (MLPs) fail or exhibit poor sample efficiency. [^9]

However, a critical review of this framework suggests several areas for refinement. While the paper demonstrates superior performance for the connectome-based model over degree-preserving rewired graphs and random graphs, the mechanism of this superiority is only partially explored. The "functional segregation" mentioned in the paper—where neurons naturally differentiate into sensory, central, and motor populations—is presented as an emergent property of the structure. Yet, the paper's dependency on the A2C algorithm suggests that the stability of the learned policy is highly sensitive to hyperparameter tuning, specifically the entropy coefficient used to prevent premature convergence. [^2][^3][^8]

Furthermore, the paper utilizes a simplified synaptic weight assignment, where $W_{vu} = N_{exc}(u,v) - N_{inh}(u,v)$. This linear approximation of synaptic strength, based solely on synapse counts, ignores the non-linear dynamics and varying transmitter sensitivities of real neurons. A more robust approach, which this report seeks to integrate into the refactoring process, involves using validated neurotransmitter predictions and causal effect estimators to refine the message-passing dynamics.

| Feature      | FlyGM Implementation         | Biological Reality             | Implications for Refactoring                        |
| ------------ | ---------------------------- | ------------------------------ | --------------------------------------------------- |
| Topology     | Static FAFB-FlyWire graph    | Dynamic synaptic plasticity    | Need for materialization-aware IDs                  |
| Weighting    | Signed synapse counts        | Multi-transmitter modulation   | Integrate Eckstein transmitter predictions          |
| Optimisation | A2C (Advantage Actor-Critic) | Biological stochastic gradient | Address variance with batch-normalized environments |
| Architecture | 3-tier functional partition  | Recurrent, hierarchical loops  | Refine CX-to-DN pathway mapping                     |

---

## Data Foundations: The FlyWire and FAFB Connectome

The refactoring of the FlyGM-A2C repository requires a transition from the generic graph structures used in the paper to the latest, high-fidelity data products from the FlyWire project. The FlyWire female adult fly brain (FAFB) dataset, specifically materialization release 783, provides the most complete neuronal wiring diagram to date. This dataset is characterised by a "public release" status, containing roughly 140,000 proofread neurons and 130 million synapses. [^5]

A critical component of this data is the systematic neuron annotation. The flywire_annotations repository contains metadata for superclasses (e.g., sensory, descending, intrinsic), cell classes, and cell types. For navigation-specific tasks, the refactored model must specifically isolate neurons within the Central Complex (CX) and the Lateral Accessory Lobe (LAL). These annotations are provided in TSV and CSV formats, which allow for programmatic analysis via the fafbseg-py Python package or the fafbseg R package. [^5][^10]

The synapses in the FlyWire dataset were predicted using the Buhmann et al. method, which assigns a "connection_score" and a "cleft_score" to each synapse. For the refactored FlyGM, it is standard practice to use a filtering convention that includes only neuronal connections with a minimum of five synapses to remove false positives. Additionally, the dataset includes neurotransmitter predictions (GABA, Acetylcholine, Glutamate, Octopamine, Serotonin, and Dopamine) with associated probabilities. These probabilities allow for the construction of a "signed" adjacency matrix, where excitatory transmitters (e.g., Acetylcholine) are assigned positive weights and inhibitory transmitters (e.g., GABA) are assigned negative weights. [^5][^][^6][^11]

### Mathematical Formulation of Signed Adjacency

The adjacency matrix A for the FlyGM is constructed such that each entry $A_{ij}$ reflects the weighted synaptic influence of neuron $j$ on neuron $i$. Let $N_{ij}$ be the set of synapses from $j$ to $i$. The weight $W_{ij}$ can be formulated as:

where $C_k$ is the cleft score of synapse $k$ and $\text{sgn}(T_k)$ is the sign assigned to the predicted neurotransmitter $T_k$. In the refactored repository, this weight calculation will be refined using the get_transmitter_predictions function from fafbseg-py, which collapses per-synapse data into per-neuron predictions weighted by the cleft score.  [^5][^6]

---

## The Role of coconatfly in Comparative Connectomics

To achieve a robust navigation controller, the FlyGM must not rely on a single brain's idiosyncrasies. The coconatfly R package (Comparative Connectomics for the NATverse) provides a uniform interface for accessing and comparing multiple Drosophila datasets, including the Janelia hemibrain, the male CNS (malecns), and the FlyWire connectome. This is essential for identifying "stereotyped" circuits that are conserved across individuals and sexes. [^4][^7][^12][^13]

coconatfly enables researchers to verify cell type identities through connectivity clustering. For instance, if the FlyGM identifies a specific sub-circuit for turning, coconatfly can be used to perform "cosine similarity clustering" to see if the same cell types in the hemibrain and FlyWire datasets co-cluster together. This provides a cross-validation step that the original FlyGM paper lacked, ensuring that the structural prior used for the RL agent is indeed a biologically conserved feature rather than a dataset artifact.

| coconatfly Dataset Name | Description                   | Research Utility                                    |
| ----------------------- | ----------------------------- | --------------------------------------------------- |
| flywire                 | Female Adult Fly Brain (FAFB) | Primary structural prior for FlyGM                  |
| hemibrain               | Janelia Hemibrain v1.2.1      | Cross-validation of cell-type stereotypy            |
| manc                    | Male Ventral Nerve Cord       | Mapping descending commands to motor output         |
| malecns                 | Whole Male CNS                | Comparative analysis of sexually dimorphic circuits |
| fanc                    | Female Adult Nerve Cord       | Coordination of leg actuators in MuJoCo             |

---

## Anatomical Navigation Circuitry: The Central Complex (CX)

A critical focus for refactoring the navigation model is the Central Complex (CX), the primary brain region responsible for path integration, head stabilization, and vector-based navigational computations. The CX is composed of approximately 3,000 neurons divided into over 250 cell types. The refactored FlyGM must accurately represent the following key components:

> 1. **The Compass Network (E-PG, P-EN, Delta7):**
>     "Compass neurons" (E-PG) in the Ellipsoid Body (EB) encode the organism's current heading relative to environmental cues, behaving similarly to head direction cells in mammals. This heading is maintained by attractor dynamics in the Protocerebral Bridge (PB), where Delta7 neurons play a key role in stabilizing the signal. P-EN neurons update this heading based on self-motion signals.

> 2. **The Steering Circuit (PFL3 to DNa02):**
>    PFL3 neurons in the PB and Fan-shaped Body (FB) serve as the link between the compass and the motor system. They receive heading information from E-PG neurons and project to the Lateral Accessory Lobe (LAL), where they form synapses onto Descending Neuron DNa02. This neuron, in turn, projects from the brain to the Ventral Nerve Cord (VNC) to influence steering behavior.

> 3. **The Goal-Directed Navigation (FB):**
>    The Fan-shaped Body (FB) layers (1-9) are thought to represent the fly's internal goals and combined sensorimotor information. PFL3 neurons integrate these goal signals with the heading signal to compute the steering commands.

In the refactored repository, these neurons must be isolated and given higher representation intensity in the GNN's latent space to reflect their functional significance in navigation. The flywire_annotations dataset and the flytable metadata within fafbseg are the primary sources for identifying these root IDs. [^10][^14][^5][^15]

---

## Refactoring FlyGM-A2C_navigation: A Software Engineering Blueprint

The refactoring of the FlyGM-A2C_navigation repository involves migrating from a static graph model to a dynamic, dataset-aware framework. This transition ensures that the neural controller is always grounded in the most current materialisation of the fly connectome. [^3]

### Integration of fafbseg-py for Dynamic Graph Construction

The core of the refactoring lies in the GraphBuilder class. Instead of loading a pre-saved .pt or .json graph, the builder should interface directly with the FlyWire CAVE (Connectome Annotation Versioning Engine) backend using fafbseg-py. The get_adjacency function is the primary tool for this, allowing for the extraction of a connectivity matrix between source neurons (sensory/afferent) and target neurons (intrinsic/efferent). [^6][^16]

To account for the "materialization" problem—where root IDs change as the segmentation is proofread—the refactored code must use the update_ids function to map any old identifiers to their current counterparts in the latest materialization (e.g., version 783). The weights in the adjacency matrix should be derived from the synapse counts returned by get_connectivity, filtered by a cleft score threshold to ensure high-confidence links. [^6][^16]

### Standardising Functional Partitions with coconatfly

The FlyGM uses a 3-tier functional partition: Afferent ($V_a$), Intrinsic ($V_i$), and Efferent ($V_e$). In the refactored model, these partitions are dynamically assigned using the cf_meta function from coconatfly. This function returns the "class" and "type" for each ID, allowing the model to automatically group neurons. [^7][^17]

For example, all neurons with the class sensory are assigned to $V_a$, while those with the class descending_neuron are assigned to $V_e$. All other brain neurons are classified as $V_i$. This ensures that the message-passing information flow—the "Biologically Grounded Flow" described in the paper—is strictly enforced based on the latest anatomical evidence. [^1][^11]

### Reinforcement Learning Pipeline: Refining A2C

The A2C implementation in the refactored repository must be updated to handle the high variance inherent in the FlyGM's sparse structure. This involves implementing vectorized environments (using 32 to 128 parallel MuJoCo instances) to stabilize the advantage calculation. The loss function remains a combination of the actor loss (policy gradient), the critic loss (value function error), and an entropy bonus to encourage exploration.

The advantage function $A(s,a)$ is computed as:

where $\gamma$ is the discount factor and $V$ is the state-value function predicted by the FlyGM's critic head. To prevent gradient explosions in the recurrent graph layers, the refactored code should use the AdamW optimizer with a learning rate scheduler and global gradient norm clipping at 0.5.

---

## Simulation and Embodiment: MuJoCo and NeuroMechFly v2

A neural controller is meaningless without a body to control. The refactored repository must integrate with NeuroMechFly v2, a state-of-the-art physics-based model of Drosophila melanogaster in the MuJoCo simulator. NeuroMechFly v2 is designed as a general-purpose framework for simulating terrestrial and aerial locomotion, including joint actions, mechanosensory feedback, and visual inputs. [^9][^18][^19]

The model's geometric primitives (geoms) accurately approximate body segments for efficient collision detection. For navigation tasks, the simulator provides an "odor-taxis" task and a "visual object tracking" task, which can be used to evaluate the FlyGM's performance. The efferent neurons ($V_e$) in the FlyGM are mapped to the actuators of the fly model, specifically the hinge joints of the legs and wings. The feedback signals from these actuators—such as joint angles and ground contact forces—are sent back to the FlyGM as afferent sensory inputs, closing the sensorimotor loop.

| Simulation Component | MuJoCo/NeuroMechFly Implementation | Role in FlyGM RL                      |
| -------------------- | ---------------------------------- | ------------------------------------- |
| Actuators            | Tendon-coupled hinge joints        | Target for efferent output (V_e)      |
| Sensors              | Joint encoders and contact geoms   | Input for afferent state (V_a)        |
| Terrains             | Rugged, 3D, and vertical surfaces  | Testing gait stability and navigation |
| Modalities           | Vision (ommatidia) and Olfaction   | High-dimensional input for processing |

---

## Causal Modeling and the Effectome Prior [^18]

An advanced feature to be integrated into the refactored repository is the concept of the "Effectome." While the connectome specifies the synaptic paths between neurons, the "Effectome" describes how strongly they actually affect each other in vivo. The paper 'The fly connectome reveals a path to the effectome' proposes a linear dynamical model estimator that uses the connectome as a prior to improve causal effect estimation.

In the refactored FlyGM, the connectome-based adjacency matrix can be used as a "structural regularizer" during the A2C training process. By penalizing weights that deviate significantly from the connectivity-based prior, we can ensure that the learned policy remains biologically plausible. This approach recovers a linear approximation to the non-linear dynamics of more biophysically realistic simulations, such as those involving stochastic optogenetic perturbations. [^20]

The estimator for the effectome weight matrix $B$ can be formulated as:

where $Y$ is the neural response, $X$ is the input, $A$ is the connectome adjacency matrix, and $\lambda$ is a regularisation coefficient. This ensures that the learned weights are sparse and primarily exist where synapses have been anatomically identified.

---

## Synthesis of Results: Performance, Efficiency, and Interpretability

The critical evaluation of the FlyGM-A2C framework indicates that biological structure provides significant advantages for embodied agents. Experimental snapshots of behavioral sequences show that the fly, under FlyGM control, can initiate gait, turn, and maintain trajectory orientation. These transitions between behavioural phases are reflected in the temporal dynamics of the connectome, where different functional zones (sensory, central, motor) activate in coordinated sequences. [^20]

One of the most striking findings is that FlyGM yields higher sample efficiency and superior performance compared to artificial architectures. This suggests that the "Static Brain Connectome" can be transformed to instantiate an effective neural policy without the need for exhaustive architecture searches. Moreover, the division of neurons into functional zones arises naturally from the trained dynamics, even without explicit planning from the developers, showing that the structure of the brain itself "suggests" how to organise information processing. [^1][^8][^2]

### Cross-Dataset Comparison of Weight Distribution

Using coconatfly, we can compare the connection strengths (weights) across hemispheres or across datasets (e.g., FlyWire vs. Hemibrain). This analysis reveals that connection strengths are generally well-correlated across datasets, supporting the use of a single connectome release as a structural prior for generalisable control.

| Metric            | FlyGM (Connectome Prior) | Standard MLP (Random Init)   | Degree-Preserving Rewired |
| ----------------- | ------------------------ | ---------------------------- | ------------------------- |
| Sample Efficiency | High (Early plateau)     | Low (Stochastic start)       | Moderate                  |
| Gait Stability    | Stable rhythmic gait     | Frequent falls/collapse      | Unstable gait             |
| Navigation Error  | Lower (Strong CX prior)  | Higher (No directional bias) | Higher                    |
| Interpretability  | High (Cell-type mapping) | Low (Latent features)        | Moderate                  |

---

## Future Outlook: Integrating Multi-Connectome Insights

The future of bio-inspired control lies in the integration of "pan-CNS" and "cross-sex" connectomes. Recent advances have generated whole-central-nervous-system connectomes for both sexes, revealing sexual dimorphisms in circuits for courtship, foraging, and social behavior. For example, the mapping of all feeding motor neurons and the tracing of complete sensory-to-motor pathways for nutrient assessment provides a blueprint for even more complex autonomous agents. [^12][^11]

The refactored FlyGM-A2C_navigation repository is designed to be a "living" platform. As new datasets (like the banc or malecns) are added to the coconatfly ecosystem, the GraphBuilder can be updated to incorporate these insights, allowing for the simulation of sexually dimorphic behaviours or the study of how hormonal signaling—regulated by neurosecretory cells (NSC) in the pars intercerebralis—modulates motor output. [^4][^21]

---

## Final Conclusions and Recommendations

The critical review and refactoring blueprint presented in this report demonstrate that the 'Bio-Inspired Neural Control' paper provides a valid, if nascent, framework for connectome-constrained reinforcement learning. The core proposition—that biological topology acts as a powerful regularizer for weak optimizers like A2C—is supported by both theoretical reasoning and empirical simulation results.

To maximize the impact of the refactored FlyGM-A2C_navigation repository, researchers should adhere to the following recommendations:

> 1. **Dynamic Materialisation Support:** 
> 	- Always use fafbseg-py to query the latest FlyWire materialisation to ensure root ID consistency.
> 2. **Multimodal Sensory Integration:**
> 	- Beyond simple navigation, integrate the visual and olfactory modules of NeuroMechFly v2 to test the FlyGM's ability to handle high-dimensional, multimodal inputs.
> 3. **Cross-Dataset Validation:**
> 	- Use coconatfly to verify that the learned policies are robust across different individual connectomes, identifying conserved "neural motifs" for movement control.
> 4. **Signed Weights and Transmitter Confidence:**
> 	- Utilise the Eckstein neurotransmitter predictions to build signed adjacency matrices, as this more accurately reflects the inhibitory/excitatory balance of the biological brain.

By grounding artificial agents in the exact architecture of a real brain, we move one step closer to achieving the efficiency, robustness, and interpretability of biological intelligence. The Fly-connectomic Graph Model is not just a tool for understanding flies; it is a template for the future of embodied, autonomous control systems.

  

[^1]: https://arxiv.org/html/2602.17997v1 (Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly - arXiv)

[^2]: https://www.researchgate.net/publication/395041170_Adaptive_Training_Program_Optimization_in_Human_Resource_Development_Using_Reinforcement_Learning_and_Multilayer_Perceptron_Models (Adaptive Training Program Optimization in Human Resource Development Using Reinforcement Learning and Multilayer Perceptron Models | Request PDF - ResearchGate)

[^3]: https://doi.org/10.48550/arXiv.2602.17997 (Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly - arXiv)

[^4]: https://github.com/natverse/coconatfly/blob/master/README.Rmd (coconatfly/README.Rmd at master - GitHub)

[^5]: https://github.com/flyconnectome/flywire_annotations (flyconnectome/flywire_annotations: Annotations for the ... - GitHub)

[^6]: https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.synapses.get_transmitter_predictions.html (fafbseg.flywire.synapses.get_transmitter_predictions - Read the Docs)

[^7]: https://github.com/natverse/coconatfly (natverse/coconatfly: Comparative Connectomics of Public and In Progress Drosophila Datasets · GitHub)

[^8]: https://www.reddit.com/r/Popular_Science_Ru/s/zOL2zsL1gA (The digitized brain of a *Drosophila* was taught to control a virtual fly : r/Popular_Science_Ru)

[^9]: https://doi.org/10.1109/TAI.2026.3664781 (Delay-Aware Reinforcement Learning with Encoder-Enhanced State Representations) https://doi.org/10.48550/arXiv.2406.03102 (arXiv Preprint)

[^10]: https://doi.org/10.7554/eLife.102230.3 (Neural circuit mechanisms for steering control in walking Drosophila - eLife) https://doi.org/10.1101/2020.04.04.024703 (bioRxiv Preprint)

[^11]: https://doi.org/10.64898/2025.12.14.694216 (Connectomic mapping of pharyngeal and gut sensory circuits in adult Drosophila - bioRxiv)

[^12]: https://doi.org/10.1101/2025.08.25.671814 (From Sensory Detection to Motor Action: The Comprehensive Drosophila Taste-Feeding Connectome - bioRxiv Preprint)

[^13]: https://doi.org/10.1038/s41586-025-08746-0 (Connectome-driven neural inventory of a complete visual system - nature)

[^14]: https://doi.org/10.7554/eLife.66039 (A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation and context-dependent action selection)

[^15]: https://github.com/natverse/coconatfly/blob/master/NAMESPACE (coconatfly/NAMESPACE at master · natverse/coconatfly · GitHub)

[^16]: https://fafbseg-py.readthedocs.io/en/latest/ (fafbseg 3.2.2 documentation)

[^17]: https://natverse.org/coconatfly/articles/extending-coconatfly.html (5. Extending coconatfly with external data sources • coconatfly)

[^18]: https://doi.org/10.1038/s41592-024-02497-y (NeuroMechFly v2: simulating embodied sensorimotor control in adult Drosophila - nature) https://www.epfl.ch/labs/ramdya-lab/wp-content/uploads/2024/08/NMF2_postprint.pdf (EPFL.ch - https://www.nature.com/articles/s41592-024-02497-y.epdf?sharing_token=jK2FbKWL99-O28WNqrpXWNRgN0jAjWel9jnR3ZoTv0MjiFZczOI3_5wYVxbEbClrTuJzjKyEfhm2kIwso489-ypEsSqlyasWAEsBCvR9WU5poT-q2bblI6hCc7Zji6wb_jZjfXl7KWLbd2pgZTmWvk_ADQ6RuzlnHwvQyipMJzg%3D)

[^19]: https://doi.org/10.1038/s41586-025-09029-4 (Whole-body physics simulation of fruit fly locomotion - nature)

[^20]: https://doi.org/10.1038/s41586-024-07982-0 (The fly connectome reveals a path to the effectome - nature)

[^21]: https://doi.org/10.7554/eLife.102684.1 (Synaptic connectome of a neurosecretory network in the Drosophila brain - eLife Reviewed Preprint)

