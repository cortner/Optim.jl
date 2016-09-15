# wrapping backtracking_linesearch! to make it interpolate
quadbt_linesearch!{T}(d::Union{DifferentiableFunction,
                                              TwiceDifferentiableFunction},
                                     x::Vector{T},
                                     s::Vector,
                                     x_scratch::Vector,
                                     gr_scratch::Vector,
                                     lsr::LineSearchResults,
                                     alpha::Real = 1.0,
                                     mayterminate::Bool = false,
                                     c1::Real = 0.1,
                                     c2::Real = 0.9,
                                     rho=0.9,
                                     iterations::Integer = 1_000) =
    backtracking_linesearch!(d,
                                         x,
                                         s,
                                         x_scratch,
                                         gr_scratch,
                                         lsr,
                                         alpha,
                                         mayterminate,
                                         c1,
                                         c2,
                                         rho,
                                         iterations,
                                         true)


function backtracking_linesearch!{T}(d::Union{DifferentiableFunction,
                                              TwiceDifferentiableFunction},
                                     x::Vector{T},
                                     s::Vector,
                                     x_scratch::Vector,
                                     gr_scratch::Vector,
                                     lsr::LineSearchResults,
                                     alpha::Real = 1.0,
                                     mayterminate::Bool = false,
                                     c1::Real = 1e-4,
                                     c2::Real = 0.9,
                                     rho::Real = 0.9,
                                     iterations::Integer = 1_000,
                                     interp::Bool = false)

    # Check the input is valid
    if interp   # this means we are coming from quadbt_linesearch!
       backtrack_condition = 1.0 - 1.0/(2*rho) # want guaranteed backtrack factor
       if c1 >= backtrack_condition
           warning("""The Armijo constant c1 is too large; I am replacing it with
                      $(backtrack_condition)""")
           c1 = backtrack_condition
       end
    end

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls = 0
    g_calls = 0

    # Count number of parameters
    n = length(x)

    # Store f(x) in f_x
    f_x = d.fg!(x, gr_scratch)
    f_calls += 1
    g_calls += 1

    # Store angle between search direction and gradient
    gxp = vecdot(gr_scratch, s)

    # Tentatively move a distance of alpha in the direction of s
    @simd for i in 1:n
        @inbounds x_scratch[i] = x[i] + alpha * s[i]
    end

    # Backtrack until we satisfy sufficient decrease condition
    f_x_scratch = d.f(x_scratch)
    f_calls += 1
    while f_x_scratch > f_x + c1 * alpha * gxp
        # Increment the number of steps we've had to perform
        iteration += 1

        # Ensure termination
        if iteration > iterations
            error("Too many iterations in backtracking_linesearch!")
        end

        # Shrink proposed step-size:
        if !interp
           alpha *= rho
        else
           # This implementation interpolates the available data
           #    f(0), f'(0), f(α)
           # with a quadractic which is then minimised; this comes with a
           # guaranteed backtracking factor 0.5 * (1-c1)^{-1} which is < 1
           # provided that c1 < 1/2; the backtrack_condition at the beginning
           # of the function in fact guarantees a factor rho.
           alpha1 = - (gxp * alpha) / ( 2.0 * ((f_x_scratch - f_x)/alpha - gxp) )
           alpha = max(alpha1, alpha / 4.0)  # safe-guard to avoid miniscule steps
        end

        # Update proposed position
        @simd for i in 1:n
            @inbounds x_scratch[i] = x[i] + alpha * s[i]
        end

        # Evaluate f(x) at proposed position
        f_x_scratch = d.f(x_scratch)
        f_calls += 1
    end

    return alpha, f_calls, g_calls
end
