# Euler Yet another proof Engine

<img align="left" src="https://josd.github.io/images/eye.png" alt="EYE"/> EYE is a reasoning engine supporting the [Semantic Web layers](http://www.w3.org/DesignIssues/diagrams/sweb-stack/2006a).  
It performs semibackward reasoning and it supports Euler paths.  
Via [N3](http://www.w3.org/TeamSubmission/n3/) it interoperable with [Cwm](http://www.w3.org/2000/10/swap/doc/cwm).  

Semibackward reasoning is backward reasoning for rules using <= in [N3](http://www.w3.org/TeamSubmission/n3/)  
and forward reasoning for rules using => in [N3](http://www.w3.org/TeamSubmission/n3/).  
This can be seen in [EYE The running Code](https://github.com/josd/etc).  

Euler paths are roughly _"don't step in your own steps"_ which is inspired by  
what [Leonhard Euler](https://en.wikipedia.org/wiki/Leonhard_Euler) discovered in 1736 for the [Königsberg Bridge Problem](http://mathworld.wolfram.com/KoenigsbergBridgeProblem.html).  
EYE sees the rule P => C as P & NOT(C) => C.  

EYE can be [installed manually](https://github.com/josd/eye/blob/master/INSTALL) on Linux, Windows and MacOSX.  
EYE is also available in a [Docker container for command line use](https://registry.hub.docker.com/u/bdevloed/eye/)  
and in a [Docker container for HTTP client use](https://registry.hub.docker.com/u/bdevloed/eyeserver/).  

## Architecture and design

The main building blocks of EYE are:  
<img src="https://josd.github.io/images/eye-reasoning-engine.png" width="50%" height="50%" alt="eye-reasoning-engine"/>  

The detailed design of EYE comprises:
1. [N3](http://www.w3.org/TeamSubmission/n3/) parser specified as Prolog rules  
2. [N3Logic](http://www.w3.org/DesignIssues/N3Logic) to N3P (N3 P-code) compiler  
3. EAM (Euler Abstract Machine) supporting Euler paths  
4. proof construction using the [vocabulary for proofs](http://www.w3.org/2000/10/swap/reason.n3)  
5. built-ins and support predicates for the above functionalities  

This is what the basic EAM (Euler Abstract Machine) does in a nutshell:
1. Select rule P => C  
2. Prove P & NOT(C) (backward chaining) and if it fails backtrack to 1.  
3. If P & NOT(C) assert C (forward chaining) and remove brake  
4. If C = answer(A) and tactic limited-answer stop, else backtrack to 2.  
5. If brake or tactic linear-select stop, else start again at 1.  

### Design issues

Implicit Quantification in N3
* See https://lists.w3.org/Archives/Public/public-cwm-talk/2015JanMar/0000  
* In [ETC](https://github.com/josd/etc) the scope of implicit universals is the top level and the scope  
  of implicit existentials is the direct formula in which they occur.  

Proof output without bindings
* See https://josd.github.io/etc/witch/witch-proof.n3  
* In [ETC](https://github.com/josd/etc) the variable substitutions naturally follow from the proof.  

## See also

EYE paper
* [Drawing Conclusions from Linked Data on the Web: The EYE Reasoner](http://online.qmags.com/ISW0515?cid=3244717&eid=19361&pg=25#pg25&mode2)

EYE tutorial
* [Semantic Web Reasoning With EYE](http://n3.restdesc.org/)

EYE talk
* [EYE looking through N3 glasses](http://www.agfa.com/w3c/Talks/2012/04swig/)
