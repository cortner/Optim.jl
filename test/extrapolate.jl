
import LineSearches
using JuLIP, JuLIP.Potentials, JuLIP.ASE
import JuLIP.Preconditioners.update!
using PyCall

let
   methods = [ LBFGS(), ConjugateGradient(),
               LBFGS(extrapolate=true, linesearch! = LineSearches.interpbacktrack!) ]
   msgs = ["LBFGS Default Options: ",  "CG Default Options: ",
          "LBFGS + Backtracking + Extrapolation: " ]

   println("--------------------")
   println("Rosenbrock Example: ")
   println("--------------------")
   rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
   for (method, msg) in zip(methods, msgs)
      results = Optim.optimize(rosenbrock, zeros(2), method=method)
      println(msg, "g_calls = ", results.g_calls, ", f_calls = ", results.f_calls)
   end

   println("--------------------------------------")
   println("p-Laplacian Example (preconditioned): ")
   println("--------------------------------------")
   plap(U; n=length(U)) = (n-1) * sum( (0.1 + diff(U).^2).^2 ) - sum(U) / (n-1)
   plap1(U; n=length(U), dU = diff(U), dW = 4 * (0.1 + dU.^2) .* dU) =
                           (n-1) * ([0.0; dW] - [dW; 0.0]) - ones(U) / (n-1)
   precond(x::Vector) = precond(length(x))
   precond(n::Number) = spdiagm( ( -ones(n-1), 2*ones(n), -ones(n-1) ),
                                 (-1,0,1), n, n) * (n+1)
   df = DifferentiableFunction( X->plap([0;X;0]),
                                (X, G)->copy!(G, (plap1([0;X;0]))[2:end-1]) )
   GRTOL = 1e-6
   N = 100
   initial_x = zeros(N)
   P = precond(initial_x)
   methods = [ LBFGS(P=P), ConjugateGradient(P=P),
         LBFGS(extrapolate=true, linesearch! = LineSearches.interpbacktrack!, P=P) ]

   for (method, msg) in zip(methods, msgs)
      results = Optim.optimize(df, copy(initial_x), method=method)
      println(msg, "g_calls = ", results.g_calls, ", f_calls = ", results.f_calls)
   end

   println("--------------------------------------")
   println("JuLIP Example (preconditioned): ")
   println("--------------------------------------")
   at = bulk("Si", cubic=true) * 5
   X = positions(at); n = length(at) ÷ 2
   at = extend!(at, Atoms( "Si", [0.5*(X[n]+X[n+1])] ))
   set_constraint!(at, FixedCell(at))
   set_calculator!(at, StillingerWeber())
   rattle!(at, 0.1)
   P = JuLIP.Preconditioners.Exp(at)
   methods = [ LBFGS(P=P, precondprep! = (P, x) -> update!(P, at, x)),
               ConjugateGradient(P=P, precondprep! = (P, x) -> update!(P, at, x)),
               LBFGS(P=P, precondprep! = (P, x) -> update!(P, at, x),
                     extrapolate=true, linesearch! = LineSearches.interpbacktrack!),
               ConjugateGradient(P=P, precondprep! = (P, x) -> update!(P, at, x),
                              linesearch! = LineSearches.interpbacktrack!),
                      ]
    msgs = ["LBFGS Default Options: ",  "CG Default Options: ",
           "LBFGS + Backtracking + Extrapolation: ",
           "CG + Backtracking:  " ]

   objective = DifferentiableFunction( x->energy(at, x),
                                       (x,g)->copy!(g, gradient(at, x)) )
   x0 = dofs(at)
   for (method, msg) in zip(methods, msgs)
      results = Optim.optimize(objective, copy(x0), method=method)
      println(msg, "g_calls = ", results.g_calls, ", f_calls = ", results.f_calls)
   end

   # println("--------------------------------------")
   # println("JuLIP Example with  Kumagai: ")
   # println("--------------------------------------")
   # si64 = bulk("Si", cubic=true) * 8
   # @pyimport atomistica
   # kumagai = ASECalculator(atomistica.Kumagai())
   # rattle!(si64, 0.1)
   # p = positions(si64)
   # int = ASEAtoms("Si", [(p[1] + p[2])/2])
   # si64 = extend!(si64, int)
   # set_calculator!(si64, kumagai)
   # set_constraint!(si64, FixedCell(si64))
   # JuLIP.Solve.minimise!(si64, precond=JuLIP.Preconditioners.Exp(si64))
end
