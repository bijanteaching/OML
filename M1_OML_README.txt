-------------------------------------
BLOG of OPTIMIZATION and MACHINE LEARNING M1 COURSE

ZOOM:
https://umontpellier-fr.zoom.us/j/9550154894


Course contents:
https://imag.umontpellier.fr/~mohamadi/index.html

-> Optimization and Machine Learning

We will use cours_opt_bm.pdf , POLY_BM.pdf , POLY_OPT_ALGO.pdf materials.

Bring your laptop
install python / spyder (anaconda)
and check it runs

----------------------------------------------------------

NO COURSE ON OCTOBER 12 & 13.

Exams: 1.5h final exam + second chance, from cours_opt_bm.pdf 

-----------------------------------------------------------
Program:
-----------------------------------------------------------
- Unconstrained optimization, convexity, quadratic functionals, matrix form
- method of descent, gradient, optimal descent step size definition  , gradient conjugate, Newton, Quasi-Newton
- Optimization under equality and inequality constraints
-Lagrangian L = J + <p, E>
-Interpretation of Lagrange multipliers: dJ / dEi = -pi
-Problem Primal (x) -Dual (p): first solve the Dual, then the primal
-Application to the fundamental problem: quadratic minimization under linear constraint
- (p, x): saddle-point Lagrangian
-Uzawa algorithm
-Comparison of penalization / Primal-Dual / Uzawa methods
-Optimization under inequality constraint: min J (x), E (x) \ leq 0: m constraints
- KKT Conditions 
- Complementarity Conditions: <E, p> = 0 <=> (pi> 0 if Ei = 0 ie active constraint, otherwise pi = 0)
- Projected gradient algorithms (where you need to know how to project in x), 
- Uzawa projection (unlike projecting x on the admissible space, projecting on p> 0 is easy), 

- penalty formulation (this is what is very much used in machine learning for regularization and to avoid over-fitting):
see Ridge and Lasso regularizations for instance
J(x)+ p |C(x)| , p>>1

POLY_BM: chap 2, 14-16
-----------------------------------------------------------

Recall: Optimality conditions
Grad J(x)  = 0
H(J(x)) SPD or SND...

-----------------------------------------------------------

First contacts with gradproj_L2

-----------------------------------------------------------
--------------------------------------------------------
Discretization : order of accuracy 
Finite differences.

2nd order derivative:
u_i+1 = u_i + ux_i h + u2x_i h^2/2 + u3x h^3/3! + u4x h^4/4!   
u_i-1 = u_i - ux_i h + u2x_i h^2/2 - u3x h^3/3! + u4x h^4/4! 
Adding the  2 lines:
u_i+1 -2 u_i + u_i-1 = u2x_i h^2 + 2 u4x h^4/4!
Divide by h^2:
(u_i+1 - 2 u_i + u_i-1)/h^2 = u2x_i  (+ 2 u4x h^2/4!)  the rest gives the accuracy of the scheme
Only inside the domain: on boundaries the order degrades.

1st order derivative:
u_i+1 = u_i + ux_i h + u2x_i h^2/2 + u3x h^3/3! + u4x h^4/4!   
u_i-1 = u_i - ux_i h + u2x_i h^2/2 - u3x h^3/3! + u4x h^4/4! 
Substract the 2 lines:
u_i+1 - u_i-1 = 2 ux_i h + 2 u3x h^3/3!
Divide by h:
(u_i+1 - u_i-1)/h = ux_i (+ 2 u3x h^2/3! )  => 2nd order.
On boundaries:
Left:
u_i+1 - u_i  =  ux_i h + u2x_i h^2/2 + u3x h^3/3! + u4x h^4/4!   
Divide by h:
(u_i+1 - u_i)/h  =  ux_i (+ u2x_i h/2 ) order 1.
Right:
(u_i - u_i-1)/h  =  ux_i (+ u2x_i h/2 ) order 1.

----------------------------------------------------------
-------------------------------------------------------------
Gradient method builds a minimizing sequence:
x0=given
loop :
xn+1 = xn - rho Grad J(xn), rho>0
till ||Grad J(xn)||< epsilon

minimizing sequence because:
J(xn+1) = J(xn) + <Grad J(xn) , (xn+1 - xn)> =  J(xn) - rho <Grad J(xn) , Grad J(xn)>
        <= J(xn)  as  <Grad J(xn) ,  Grad J(xn)> =  ||Grad J(xn)|| >= 0 
		
QUASI-NEWTON Methods use a symetric definite positive matrix M close to the inverse of the Hessian 
to change the metric: <Grad J(xn) ,  Grad J(xn)>_M   M:SDP  

----------------------------------------------------------

PROG: run gradproj_L2.py and understand the code

-Use the code to find the solutions to question 5, compare with the theoretical solution.

-Identify the following in the code:
     1.the fixed-step gradient method
     2. the calculation of the gradient of J (by finite differences and by exact calculation)

-Solve problem 3/ in dimension ndim=3 and with y=(1,1,1). Compare to your theoretical solution. 
-Plot J(x) in R^2
-Show that two local minima exists for this problem :  xopt = +/- y / (sqrt(2) ||y||)
-You need to define the functional and its gradient in the code.
start from xinit=xmin and then xinit=xmax: notice that a gradient method fails if one starts far from the solution because gradient is close to zero everywhere.
-check your gradient using finite differences: vector in R^ndim
dJ/dxi = (J(x+eps ei) - J(x-eps ei))/(2 eps)  ei ieme vector of canonic basis of R^ndim.

-Calculate the Hessian at optimum by finite differences: what is the calculation complexity ?
-Check second order optimality condition: Hessian Sym Pos Def ?  lambda1=3 , lambda2=0.6 ?
-check the condition number K=5
-Change functional to degrade K using an anisotropic metric : see the impact on the convergence.
----------------------------------------------------------

Ex. 30/  
xn+1 = xn -rho_opt grad j(xn)   ( x \in R^n , Grad j(x) \in R^n)
rho_opt = argmin_rho J(x-rho grad j(xn)) linesearch (optim 1d)
Then <grad j(xn), grad j(xn+1)>=0  two successive directions are orthogonal.
see poly_opt_algo.pdf page 11.

J(xn - rho grad Jn) function in rho.
Finding rho_opt means finding rho s.t. J'(xn - rho grad Jn)=0  derivative of J in rho
J'(xn - rho grad Jn) = J'(xn+1(rho)) = grad J(xn+1) xn+1' = -<grad J(xn+1) , grad J(xn)> =0

----------------------------------------------------------

Equivalence between strong form / minimization / weak form (ex 31):

Ax=b / min J(x) =1/2 (A x,x) - (b,x)  / (Ax,y)=(b,y) forall y

In matrix form J(x) = 1/2 (A x,x) - (b,x) 
Show Grad J = Ax-b 
Therefore, minimizing J(x) <=> Grad J = Ax-b = 0 <=> Ax=b

Grad J = 1/h (J(x+h) - J(x)) = ...

Use gradproj to solve Ax=b with A : Hilbert Matrix and xopt=1 => b=A xopt,  xn -> xopt ?  inverse problem.

#############################################
def func(ndim, x):   #J(x)=1/2 (Ax,x) - (b,x)
    A=hilbert(ndim)
    xcible=np.ones((ndim))
    b=A.dot(xcible)
    f=0.5*(A.dot(x)).dot(x)-b.dot(x)
    return f
#############################################
def funcp(ndim, x):    #Ax-b
    A=hilbert(ndim)
    xcible=np.ones((ndim))
    b=A.dot(xcible)
    fp=[]
    fp=A.dot(x)-b
    return fp
############################################
----------------------------------------------------------
POLY 2.7 : numerical linear algebra, direct and iterative methods
2.7.7 : steepest descent to solve Ax=b by minimization of energy (J(x)=1/2(Ax,x)-(b,x))
----------------------------------------------------------

Ex. 32. If J(xk)=<Axk,xk>-<b,xk> (+c) with Gk= A xk - b,  Hessian=A (SDP)
rho_opt = <Gk,Gk>/<A Gk,Gk>  ?
J(xk-rho Gk) = J(xk) - rho <Gk,Gk> + 1/2 <-rho Gk A , -rho Gk>        (scd order dev taylor 2)
J(xk-rho Gk) = J(xk) - rho <Gk,Gk> + 1/2 rho^2 <Gk A,Gk> 
J(xk-rho Gk) is an equation in rho. To minimize we look for  J'=0 ==> rho_opt= <Gk,Gk> / <AGk, Gk>

This can also be found using the orthogonality of two successive directions for rho to be optimal:

<Gk+1 , Gk> = <A xk+1 - b , A xk - b> = <A (xk - rho Gk) - b , A xk - b>
= <A xk - b - rho A Gk , A xk - b > = <Gk - rho A Gk , Gk > = <Gk, Gk> - rho < A Gk , Gk > =0
=> rho_opt= <Gk, Gk> / < A Gk , Gk >

--------------------------------

matrice_gradpy.py is the adaptation of gradproj for the solution of linear systems through 
the minimization of a quadratic functional. In this case everything can be expressed exactly,
 in particular the optimal steepest descent size (thm 14.3.1 - exo 31)

- start from x=0, xcible=list(range(0, ndim)), A=Hilbert,  b=A xcible
-Compare solution by matricegrad with gradproj for Hilbert matrix inversion in dim=5.
-check that successive descent directions are orthogonal (exo 30, POLY_OPT_ALGO.pdf(p. 11) 
-identify and plot the optimal step size (exo 32)
-How many iterations in each case to reach relative accuracy of 1.e-3. 
(1470 with grad exact and Finite Difference, vs. 155 with matrice_grad)
Optional:
-Make the code parametric:  min J(x,N) using a nested loop on N.
Plot the condition number of A (Log10(Cond)) and the L2 norm of the error for 10 < N < 100: 
err = || alpha_opt Grad J(xopt) || = || xopt - xtarget || (discrete L2 norm)

-----------------------------------------------------------
optim_rosenbrock.py shows optimization problem with build in python libraries
-----------------------------------------------------------

Express in expression f(λ) = J(u + λv) = a λ**2 + b λ + c , a, b, c for J(u)=1/2 (A u, u) - (b,u)

Convexity: read chapter 16  --> 16.4
In particular, projection on closed convex sets (projection with angle > 90):
( x-x* , y-x*) <= 0    x*=Proj_K(y) y\in H Hilbert, K convex set \subset H
H needs to be Hilbert for us to be able to define angles (Banach would'nt be enough).
Difference between Banach and Hilbert is that one can only define distances in a Banach, not angles.
All vector subspaces are convex (show it).
Projection on Vector subspaces is orthogonal:
( x-x* , y-x*) = 0    x*=Proj_K(y), y\in H Hilbert, K Vector subspace \subset H

Illustrate inequalities of 16.3.1 by two pictures in 1D.

--------------------

Ex. 7, 18, 19, 20

----------------------
TO DO
Use gradproj to solve ex. 20.
project on R2[X] the following functions:  x**3, sin(x), exp(x) defined on [-1, 1]

Size of the optimization problem: 3

P_F(f) projection of F of f    with F: R2[X]

We have seen 2 basis of F:  (1, X, X**2) and (Q0, Q1, Q2) orthonormed

Define the optimization variables:
V=(a,b,c) to express  P_F(f) = a Q0  + b Q1 + c Q2   or P_F(f)= a + b X + c X**2
  
Define the functional to minimize:
J(V)= dist(f, P_F(f)) = || f - P_F(f) ||**2 = || f - (a Q0 + b Q1 + c Q2) ||**2
J(V)= \int_[-1}^1 (f(x) - (a Q0(x) + b Q1(x) + c Q2(x))**2 dx   see ex. 18
Use numerical integration (Riemann measure)

Validation : check if J(Vopt)= 4/175  for f=x**3 and Vopt=(0, 3/5, 0)

Define the gradient:
-start with finite differences
-define funcp : gradient of J wrt V = (J,a ; J,b ; J,c)
This requires the linearization of J wrt V
Compare the gradient with FD


--------------------
POLY 354
Optimization under equality constraint

See iso-perimetric problems: Dido ex. 26

We will use these concepts to introduce Lagrange multipliers in the presence of equality constraints.

Grad_x J = <p, Grad_x E >    p in R^m if E in R^m :  we have m equality constraints

L(x,p) = J(x) + <p , E(x) > is called the Lagrangian

Lagrange Thm: At optimum Grad J is linear combination of Grad Ci.

-----------------------------------------------------------
minimize surface of a package, under constraints on its volume and basis surface.

J(a,b,c)=2(ab+bc+ac) = 2(S + S/a V/S + a V/S),  J'=2(-V/a^2 + V/S) = 0 => a=sqrt(S)
C1(a,b,c)=abc-V=0 => a S/a c = V => c=V/S
C2(a,b,c)=ab-S=0 => b=S/a

Find the solution using Lagrange thm.

Same question for J quadratic under a linear constraint (page 355):
(x,y)=( 2/3 , 1/3 ) et lambda = - 4/3.

-----------------------------------------------------------

Ex 10, 11 et 12  : 

-first find the solution after removing the constraint through variable elimination 
then 
-find the solution as zero of Grad L(x,p) = 0 using zero_vect_fct.py

The system is often nonlinear and we need a minimizer (Newton) to find its zeros.

With m equality constraints (E_j(x)=0, j=1,...,m), the Lagrangian L(x,p)=J(x)+ <p,E(x)> with p in R^m

Check of Grad(J(x)) and Grad(E(x)) are parallel more generally if:
Grad(J(x)) in Span{Grad(E_i(x)), i=1,...,m}

Difference between Jacobian and Hessian :
J(x) : Rn -> R , Grad J(x) in Rn, Hess(J(x)) in Rnxn (symmetric matrix) and Jac(Grad(J(x)) = Hess(J(x)) 
But in gal: x in Rn, F(x) in Rm, Jac(F(x)) in Rmxn (not rectangular)


----------------------------------------------------------
----------------------------------------------------------

43. (POLY p. 355)
  
Minimize J(x)=1/2(ax,x)-(b,x) under constraint Bx=c
is equivalent to minimize (a(x-xb), x-xb) on K (vect. space defined by Bx=c)

Existence and uniquness : fct strictly convexe (quadratic) on a convex ensemble has a unique solution.

a xb = b is sol of min J(x) on Rn
Consider:
1/2 <a(x-xb), x-xb> = 1/2( <a x,x>+<a xb, xb>-2<a xb,x> )
=1/2 <a x,x>+ 1/2 <a xb, xb>  - <b,x>
=J(x)+ 1/2 <a xb, xb> = J(x) + Cste    (Cste>=0)
in other words:
J(x)= 1/2 < a(x-xb), x-xb > - Cste
Therefore, minimizing J on K is equivalent to minimizing 1/2 < a(x-xb) , x-xb > 

We are looking for the closest point on K to xb.
 
We have B xs = c (as xs is in K)
Therefore, B x = B xs =>  B (x - xs) = 0
<=> x - xs \in Ker(B) <=>  x - xs Orthogonal to Bi in Rn s.t.
B(m,n) = (B1, ..., Bm)^t  :  (Bi , x-xs) =0  i=1,...,m 
 bi column vectors of B^t
Recall linear algebra equality:  Ker(B)^\orth = Im(B^t)

Recall:  Bi are linear indept as  rang(B)=m.  Bi are row of B.

And from projection  on convex (vector subspace) :
<a (xb - xs) , (x - xs)>  =  0 (proj. orthogonal)  ( <= 0 is just convex and not subspace)
<=> a (xb - xs)= b - a xs  is orthogonal to (x - xs) and then is linear combination of 
column vectors of B^t from previous point.
We therefore have m real variables s.t.:
b - a xs = \sum_{i=1}^m p_i Bi
 p_i are called Lagrange multiplier.

We recover the Lagrangian:
L(x,p)= J(x) + <P, B x - c> = 1/2 <A x-b,x> + <B^t P, x> - <P, c>
with P=(p1,...,pm)^t
Grad_x L = Ax - b + \sum_{i=1}^m p_i Bi

----------------------------------------------------------
-------------------------------------------------------------

17. q(x)=b(x,x) with b(x,y)=x1 y1 + x2 y2 + x3 y3 +1/2(x1 y2 + x2 y1)
b is symmetric bilinear, therefore, q(x) is quadratic
b is definite as b=0 <=> x=0 and positive as it is a sum of square.
A=(1, 1/2, 0)(1/2, 1 , 0) (0,0,1)   b(x,y)=x^t A y
[alpha,x]=J(x) => alpha_1=alpha_2=2/3  alpha_3=1

Continuous function reaches its bounds on a compact ensemble in Rn.
Cauchy-Schwartz
-sqrt(q(alpha)) sqrt(q(x)) <= [x,alpha] <= sqrt(q(alpha)) sqrt(q(x))
q(x)=1  =>  -sqrt(q(alpha)) <= [x,alpha]=J(x)=x1+x2+x3 <= sqrt(q(alpha))

q(alpha)= 4/9 + 4/9 + 9/9 + 1/2 (4/9 + 4/9) = 2.33

Cauchy-Schwartz equality [x,alpha] = cste =>  x* = lambda alpha with lambda in R
x* is such that q(x*)= lambda^2 q(alpha)=1  => lambda = 1/sqrt(q(alpha)) => x* = alpha/sqrt(q(alpha))
x* = +/- sqrt(7/3)
 
----

read 16.6.3 

Ex. 12 with two constraints:
consider an additional constraint: E_2(x)= x_1 * ( x_1 - 4 - x_2) -1 =0 
L=J(x) + p1 E1(x) + p2 E2(x)
fun:  Grad L
Jac: Grad Grad L 

1. reduce to a single variable minimization problem: find the solution without constraint, 
as E2 is nonlinear, this is perhaps impossible to achieve easily.
2. use zero_vect_fct.py to solve the constrained problem
3. Which constraint is more active at the optimum : 
analyze the sensitivity of the functional with respect to each of the constraints.

('solution=', array([-0.20376823,  0.70376823,  0.5       , -0.5       , -1.        ]))
('TEST Gradient Lagrangien NUL =', [1.375521918589584e-11, 1.0274003869881199e-12, -1.8640644583456378e-13, 0.0, 7.97961696719085e
-12])
('J(x)=', 1.2500000000079796)
('TEST E1(x) constraint=', 0.0)
('TEST E2(x) constraint=', 7.97961696719085e-12)

Sensitivity analysis : J'_E1=1/2  , J'_E2=1

Check that at optimum we have:  Grad J \in Span{Grad E1, Grad E2}
without using information : Grad L(x,p)=0

QR Python --> Gram-Schmidt:                                            
Grad J - <Grad E1/||Grad E1||,Grad J> (Grad E1/||Grad E1||)  - <Grad E2/||Grad E2||, Grad J> / (Grad E2/||Grad E2||) ?

-----------------------------------------------------------

Quadratic minimization under linear constraint
Read POLY p. 358, and consider Ex. 44 to 46 

---
start with simple example:
J(x, y) = x**2 + 2 y**2  under constraint x + y = 1.

First eliminating a variable : y=1-x, j(x)=x**2 + 2 (1-x)**2 = 3 x**2 - 4x + 2 
j'(x)=0 => x=4/6 , y=1-4/6=1/3
Check the solution with Primal-Dual approach
Express: A, B, b, c in dual-primal approach
find that p=-4/3

Same question with an addition constraint: x+2y=-1 
=> intersection of the two constraints is the solution.

---

Then, solve the portfolio optimization problem in the Markovitz sense (ex. 46)
primal-dual (POLY p. 358) for a portfolio with 3 minimum 3 assets. 

min_x Risk=J(x)=1/2(Ax,x)   -->  x_opt=(x_1,x_2,x_3) + p_1, p_2 + J_opt for R given 2 contraints:
C_1(x)=x0+x1+x2-1=0
C_2(x)=r0*x0+r1*x1+r2*x2-R=0 

at optimum, dJ/dC_i = -p_i 

+ Box contraints : xmax(i)=1, xmin(i)=0

Plot the (return R / Risk J) front with N=20 points.
This requires the solution of N optimization problem with R \in [R_min, R_max] 
with R_min=0.01, R_max=0.07 --> save corresponding (R,J)_i=1,...,N and plot.

r_i and A come from historical data (expertise of the bank).

https://fr.wikipedia.org/wiki/Th%C3%A9orie_moderne_du_portefeuille#:~:text=La%20th%C3%A9orie%20moderne%20du%20portefeuille,au%20risque%20moyen%20du%20march%C3%A9.

http://opcvm.info/theorie-moderne-du-portefeuille/


Start with
ndim=3
A[i][j]=exp(-0.05*(i-j)**2)
ri=0.01, 0.02, 0.06
R=0.03

B=[[1,...,1] [r1,...,rn]] 2 lines as two constraints
c=[1,R]^t


Then consider true values from:
https://www.centralcharts.com/fr/gm/1-apprendre/3-bourse/5-gestion-portefeuille/191-volatilite-d-un-portefeuille-boursier

Eliminating 2 variables, then using primal-dual approach (using zero_vect_fct.py or optim_rosenbrock.py)

Uzawa algorithm to avoid inversion of A: A-1
Iterative minimization in x and maximization in p.
See uzawa_simple.py

Adapt gradproj to solve general minimization under equality constraint (min in x / max in p)

-----------------------------------------------------------
----------------------------------------------------------
Optimization under inequality constraint 
----------------------------------------------------------
RMQ: equality constraint = 2 inequality contraints A=B  <=>  A<B & B>A
E(x)=0   <=>  E(x)< 0  &  -E(x)<0

Def: E(x) <= 0  =>  1/ inactive E(x)<0   2/ E(x)=0 active (on the boundary of the constraint ensemble)

----------------------------------------------------------
Lagrange multipliers for inequality constraints are in R+ and not in R as for equality cstr.

KKT:
L(x,p)=J(x)+ \sum_i=1^m p_i E_i(x) =  J(x)+ <P,E>, avec p_i \in R^+  
grad_x L(x,p)  = grad_x J + \sum_i=1^m p_i grad_x E_i(x)  
p_i E_i  = 0 for i=1,...,m  complementarity cdt
Either p_i=0 when E_i<0, or p_i>0 when E_i=0
<P,E>=0  P=(p_1,...,p_m)  E=(E_1,...,E_m)

before starting always rewrite  E>=0 form as -E<=0 
ex: x>=0 => -x=<0

------------------------------

Ex 47 and check with optim_ci.py

Ex 48 and check with optim_ci.py

------------------------------

-24. parallepipede defined (X=a/sqrt(3), Y=b/sqrt(3), Z=c/sqrt(3)) has volum=abc/(3 sqrt(3))

min -XYZ  under contraints: -X,-Y,-Z <= 0  et X**2/a**2 + Y**2/b**2 + Z**2/c**2 - 1 <= 0
L = -XYZ - p1 X -p2 Y -p3 Z + p4 (X**2/a**2 + Y**2/b**2 + Z**2/c**2 - 1)

Grad_x L => 3 eq.
-Y Z + p4 2 x/a^2 -p1 = 0  *X  et add:   -3 XYZ + 2p4 =0  => p4=3/2 xyz
-X Z + p4 2 y/b^2 -p2 = 0  *Y
-Y X + p4 2 z/c^2 -p3 = 0  *Z
p4 (x^2/a^2 + y^2/b^2 + z^2/c^2 - 1) = 0 4eme eq.
p1 X=0  5eme eq
p2 Y=0  6eme eq
p3 Z=0  7eme eq

-YZ + 3 XYZ X/a^2 = 0 => X^2 = a^2/3  => X = a/sqrt(3), idem pour Y et Z et p4= abc/(2 sqrt(3)), 
and p1, p2, p3=0  as -X,-Y,-Z < 0


https://www.wolframalpha.com/calculators/system-equation-calculator

------------------------------

-25. maximiser x1 x2 (x2 - x1) sous contraintes x1 + x2 = 8, x1>0 x2>0
<=>
minimiser -x1 x2 (x2 - x1) sous contraintes x1 + x2 = 8, -x1<0 -x2<0

solution: x_{1,2} = 4 (+/-) 4/sqrt(3)

L=-x1 x2 (x2 - x1) + p1 (x1+x2-8) - p2 x1 -p3 x2
=-x1 x2**2 + x1**2 x2 + p1 (x1+x2-8) - p2 x1 -p3 x2

Grad_x L=0 :
 -x2**2 + 2 x1 x2 + p1 = 0   
 -2 x1 x2 + x1**2 + p1 = 0   
 p1 (x1+x2-8)=0                    
 
 x1,x2 = 4 (1 +/- 1/sqrt(3) )  >= 0  et x1+x2=8  avec wolfram 

otherwise, use the equality constraint to eliminate x2:
 -x2**2 + 2 x1 x2 + 2 x1 x2 - x1**2 = 0   
 -(8-x1)**2 + 2 x1 (8-x1) + 2 x1 (8-x1) - x1**2 = 0   ==> x1 sol eq 2nd ordre,  x2=8-x1, p1=x2**2 - 2 x1 x2

------------------------------

-27. link between optimization and spectral analysis : eigen values and vectors. 

L(x,p) = -<x , A x> + p (<x , x> - 1) 
Grad_x L = - 2 A x + 2 p x  = 0    <=>   A x = p x  <=> p is eigen value of A and x associated normalized eigen vector.

------------------------------

-15. Link between convex set and the convex fct defining its boundary (parabole first then general situation with g convex).
Let z=t x + (1-t) y, with x and y in Cg (therefore g(x)<=0 and g(y)<=0), 0 <= t <= 1
g(z) = g(t x + (1-t) y) <= t g(x) + (1-t) g(y) <= 0  as t and (1-t)>=0 and g(x) and g(y)<=0 
Therefore z belongs to Cg =>  Cg is convex.

If g(x1, x2) = x1**2 - x2,  g(x1,x2) is convex as the eigen values of the Hessian matrix 
(H=diag(2,0)) are positives (or zero). g is convex, not strictly. 

The projection on convex thm permits to look for the solution minimizing the distance under inequality constraint. 

y in R2, we look for x in R2 minimizing J(x)=||x-y||=(x1 - y1)**2+(x2 - y2)**2 
s.t. x in Cg : (x1**2 - x2 <=0) 

A single Lagrange multiplier as there is only one constraint.
KKT cdt in each of the two cases give the solution in each case: 
Lagrange multiplier zero (inactive constraint, y in C and x=y is solution), 
Lagrange multiplier strictly positive (active constraint, projection on the boundary of C)

Geometry recall:
.Normal unit exterior vector to the boundary of C : y=x**2  

t=(tx, ty)/xno    tx=1, ty=2*x  xno=sqrt(tx**2+ty**2)
n=(nx, ny) = (ty, -tx)/xno with  <t,n>=(tx*ty-ty*tx)/xno**2=0    


--------------------------

Ex. 50

Adapt gradproj to solve the constrained minimization problem.

Accounting for xi <= xi+1 constraint:
1. dynamic box constraint: xmin(i+1) = x^n(i) at each gradient step
2. introduce n-1 artificial variable (like in simplex method) y_i>=0 s.t.: x2=x1+y2, x3=x2+y3...  with fixed box constraints.
The final solution is then : (x1, x2=x1+x2, x3=x2+x3,...)

Test on target vector a s.t. a_i=(i+1)**2 for ndim=50 starting from x0=0

Modifier le code pour illustrer le comportement en pas fixe et pas optimal 
(sous-optimal avec heuristic type Armijo implementee).

-----------------------------------------------------------
Ex. 57
FD1, FD2, CVM (AutoDiff direct and reverse modes)
Make your code parametric wrt epsilon for FD1,2
Plot the errors wrt epsilon, find optimal epsilon in FD

-----------------------------------------------------------

Ex. 59
Having two functionals to minimize : J1(x1,x2), J2(x1,x2)
Multi-criteria optimization using the following approaches:

-Build a composite function by convex combination : J= a1 J1 + a2 J2, a1+a2=1

-Using constrained minimization : min J1 under constraint J2 < J2(x10,x20)
making sure that one does not degrade the other functional

-Alternate constrained minimization and Nash equilibrium: 
x1n=argmin(J1(x1,x2n-1) under constraint J2(x1n,x2n-1) < J2(x1n-1,x2n-1)
x2n=argmin(J2(x1n,x2)) under constraint J1(x1n,x2n) < J1(x1n-1,x2n-1)
converges toward the iterative Nash equilibrium. 
2 KKT systems at each iteration, or penalty

Use optim_ci47.py 
-----------------------------------------------------------

Machine Learning with Python Handbook

https://jakevdp.github.io/PythonDataScienceHandbook/

-----------------------------------------------------------
Scikit-learn platform
-----------------------------------------------------------
We will use this python ensemble to discover data manipulation and machine learning.
Install scikit-learn modules when necessary.

2 main classes of machine learning: 
Supervised Learning: we have a labelled dataset 
Unsupervised Learning: clustering

We will see classification (outputs are integer) and regression (outputs are real variables) problems.

The aim is to be able to reproduce the following chain:

dataset (dirty) --> analysis of DB --> reduction of dimension *
--> dataset splitting for learning, validation and inference on the test DB (unlabelled)
--> trials of different ML algorithm and scoring 
--> crossfold 

-----------------------------------------------------------
Database 
-----------------------------------------------------------
DB comes often in csv format.
Personal machine learning project :

Consider DB_Iris.csv and iris_classif.py

What are the size of input and output spaces ?
How many scenarios are present in the DB ?

Split the data in two subset for learning and test 
https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/

-----------------------------------------------------------

Y(150) = X(150,4) L(4) , Linear model: find L solving (XXt) L = Xt Y

-Linear Models (ex 40) for regression/classification.
-Read and plot csv data
-Learn and Test with linear model and different regularizations (Ridge (L2), Lasso (L1), elasticNet).
-what are MAE, R2 scores
-Use a 80%-20% ratio for the learning-testing databases
-plot y_pred vs. y_test
-How Lasso can be used to identify most important features selection (looking at larger coefficients) ?
https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b
https://en.wikipedia.org/wiki/Lasso_(statistics)

-Proceed with cross-validation (present results for 10-fold cross-validation)

----------------------------------------------------

Adapt the classifier to these medical data (DB_diabetes):

https://www.youtube.com/watch?v=U1JIo8JSuYo

Compare the methods (LM, Logistic regression, KNN, RF and SVM) for a medical classification 
problem where one would like to estimate the probability for an individual to develop diabetes.
The learning data base is pima-indians-diabetes.csv from R^8 to {0,1}
Discuss the different choices of the parameters in the models and report the best
choices.
Present the confusion matrices for each method alongside with a table comparing different methods specificity, sensitivity and error.
And conclude which method performs best on this problem.

see a good discussion site for this problem with a few ML methods:
https://www.andreagrandi.it/2018/04/14/machine-learning-pima-indians-diabetes/ 

----------------------------------------------------

SVD/SVM/SVR/SVC (39) : compression et structuration de l'information (image 1024 x 1024) representee par un vecteur de 
taille n_svd (200 plus grand valeurs singulieres)

repondre aux questions de l'exo 39.

image -> b&w ->  matrice ->  svd -> compression -> image reconstitue : trouver le bon niveau de compression
et comparaison avec distance cosinus entre 2 images via leurs svd

-SVM (Ex. 51).
https://machinelearningmastery.com/multi-output-regression-models-with-python/

For SVM it seems that we have convergence problems ?
See which optimization algorithm is used in your ML algorithms and is it possible to user second order optimization algorithm ?

SVM(SVC/SVR): Detail Lagrangian and KKT conditions for the constrained optimization problem.

min J(w)= 1/2 ||w||^2 s.t. l_k (w^T x_k + b) > 0  for k = 1,...,3

------------------------------------------
Logistic regression: Detail all steps including the maximization of the likelihood (Ex. 41)
To apply logistic regression here we need to transform the data in order for the outputs to be 0 and 1 (probability). 
Then one can fit the logit function to the data.
Interprete beta_i once their values found. 

-Use gradproj to develop a logistic regression algorithm (ex. 41). 
Application to DB_diabetes.csv databasis. 
use 50% of DB for learning and 50% for testing.

------------------------------------------

-KNN 
https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/


----------------------------------------------------


-Modify the steepest descent iterations implementing stochastic gradient. 

from:

xn+1 = xn - rho * grad_x J 

to

xn+1 = xn - rho * d   with  d=(partial J / partial xi) ei    i randomly chosen in {1,...,n}

compare convergence and angles between grad_x J  and d, why this is still a descent method ?
 
-----------------------------------------------------------





----------------------------------------------------
Useful youtube videos on ML
----------------------------------------------------
linear and polynomial fits of data
https://www.youtube.com/watch?v=ro5ftxuD6is

reading csv with csv
https://www.youtube.com/watch?v=OBPjFnyxoCc
scattering and plotting
https://www.youtube.com/watch?v=t9BbYPn2nyw

reading csv with pandas
https://www.youtube.com/watch?v=Lh1dxgxk7dw
regression with sklearn
https://www.youtube.com/watch?v=cBCVvND5i9o

linear regression with sklearn
https://www.youtube.com/watch?v=NUXdtN1W1FE

Good page on LM
https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b

KNN regressor
https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/


----------------------------------------------------

