# ExOSITO


*Zongliang (Jerry) Ji, Andre Amaral, Anna Goldenberg, Rahul G. Krishnan* <br>

## Abstract
Ordering a minimal subset of lab tests for patients in the intensive care unit (ICU) can be challenging. Care teams must balance between ensuring the availability of the right information and reducing the clinical burden and costs associated with each lab test order. Most in-patient settings experience frequent over-ordering of lab tests, but are now aiming to reduce this burden on both hospital resources and the environment.
This paper develops a novel method that combines off-policy learning with privileged information to identify the optimal set of ICU lab tests to order. Our approach, EXplainable Off-policy learning with Side
Information for ICU blood Test Orders (ExOSITO) creates an interpretable assistive tool for clinicians to order lab tests by considering both the observed and predicted future status of each patient. 
We pose this problem as a causal bandit trained using offline data and a reward function derived from clinically-approved rules; we introduce a novel learning framework that integrates clinical knowledge with observational data to bridge the gap between the optimal and logging policies. 
The learned policy function provides interpretable clinical information and reduces costs without omitting any vital lab orders, outperforming both a physician's policy and prior approaches to this practical problem.


<p align="center">
  <img src="assets/labfig1.png" width="600px">
  <br>
</p>

## Usage

- `src/lto1_get_labval_pred_model.py`: Script for learning patient status forcasting prediction model $\phi$
- `src/lto2_get_gpscnf_model.py`: Script for learning global propensity score fuction $f_\psi$
- `src/lto3_get_laborder_policy.py`: Script for learning labtest order policy $\pi_\theta$ 


## Citation

```bibtex
@article{ji2025exosito,
  title={ExOSITO: Explainable Off-Policy Learning with Side Information for Intensive Care Unit Blood Test Orders},
  author={Ji, Zongliang and Amaral, Andre Carlos Kajdacsy-Balla and Goldenberg, Anna and Krishnan, Rahul G},
  journal={arXiv preprint arXiv:2504.17277},
  year={2025}
}
```