## Adversarial-example-security
### A demo for 'How secure are the adversarial examples themselves ?' 
    H. Zeng, K. Deng, B.Chen, and A. Peng, "How secure are the adversarial examples themselves?" to appear in ICASSP2022
    Abstract: Existing adversarial example generation algorithms mainly consider the success rate of spoofing target model, but pay little attention to its own security. 
    In this paper, we propose the concept of adversarial example security as how unlikely themselves can be detected. A two-step test is proposed to deal with the adversarial
    attacks of different strengths. Game theory is introduced to model the interplay between the attacker and the investigator. By solving Nash equilibrium, the optimal 
    strategies of both parties are obtained, and the security of the attacks is evaluated. Five typical attacks are compared on the ImageNet. The results show that a rational
    attacker tends to use a relatively weak strength. By comparing the ROC curves under Nash equilibrium, it is observed that the constrained perturbation attacks are more 
    secure than the optimized perturbation attacks in face of the two-step test. The proposed framework can be used to evaluate the security of various potential attacks and
    further the research of adversarial example generation/detection.
    
See 'Supplementary material_Security of AE.pdf' for more results. The demo shows how the proposed two-step test works. Test1 utilyzes the spatial instability nature of adversarial examples, whereas test2 is based on the fact that adversarial perturbation destroys local correlation in natural images.

### Usage
For test1, run Test1_AddDe.py in the /Test1 folder  
For test2, run Demo_test2.m 

