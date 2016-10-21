import LineSearches

let
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
   println("p-Laplacian: ")
   for extrapolate in (false, true)
      method = LBFGS(P=P, linesearch! = Optim.quadbt_linesearch!)
      results = Optim.optimize(df, copy(initial_x),
                              method=method,
                              f_tol = 1e-32, g_tol = GRTOL)
      println("  Extrapolate = ", extrapolate)
      println("     g_calls = ", results.g_calls,
              ", f_calls = ", results.f_calls)
   end


   println("Rosenbrock: ")
   rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
   for extrapolate in (false, true)
      if extrapolate
         method = LBFGS(linesearch! = Optim.quadbt_linesearch!)
      else
         method = ConjugateGradient()
      end
      results = Optim.optimize(rosenbrock, zeros(2), method=method)
      println("Extrapolate = ", extrapolate)
      println("     g_calls = ", results.g_calls,
              ", f_calls = ", results.f_calls)
   end

end
