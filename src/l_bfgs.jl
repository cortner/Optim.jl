# Notational note
# JMW's dx_history <=> NW's S
# JMW's dg_history <=> NW's Y

# Here alpha is a cache that parallels betas
# It is not the step-size
# q is also a cache
function twoloop!(s::Vector,
                  g::Vector,
                  rho::Vector,
                  dx_history::Matrix,
                  dg_history::Matrix,
                  m::Integer,
                  pseudo_iteration::Integer,
                  alpha::Vector,
                  q::Vector)
    # Count number of parameters
    n = length(s)

    # Determine lower and upper bounds for loops
    lower = pseudo_iteration - m
    upper = pseudo_iteration - 1

    # Copy g into q for backward pass
    copy!(q, g)

    # Backward pass
    for index in upper:-1:lower
        if index < 1
            continue
        end
        i = mod1(index, m)
        @inbounds alpha[i] = rho[i] * vecdot(slice(dx_history, :, i), q)
        @simd for j in 1:n
            @inbounds q[j] -= alpha[i] * dg_history[j, i]
        end
    end

    # Copy q into s for forward pass
    copy!(s, q)

    # Forward pass
    for index in lower:1:upper
        if index < 1
            continue
        end
        i = mod1(index, m)
        @inbounds beta = rho[i] * vecdot(slice(dg_history, :, i), s)
        @simd for j in 1:n
            @inbounds s[j] += dx_history[j, i] * (alpha[i] - beta)
        end
    end

    # Negate search direction
    scale!(s, -1)

    return
end

macro lbfgstrace()
    quote
        if tracing
            dt = Dict()
            if o.extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(g)
                dt["Current step size"] = alpha
            end
            g_norm = norm(g, Inf)
            update!(tr,
                    iteration,
                    f_x,
                    g_norm,
                    dt,
                    o.store_trace,
                    o.show_trace,
                    o.show_every,
                    o.callback)
        end
    end
end

immutable LBFGS <: Optimizer
    m::Int
    linesearch!::Function
end

LBFGS(; m::Integer = 10, linesearch!::Function = hz_linesearch!) =
  LBFGS(Int(m), linesearch!)

function optimize{T}(d::DifferentiableFunction,
                     initial_x::Vector{T},
                     mo::LBFGS,
                     o::OptimizationOptions)
    # Print header if show_trace is set
    print_header(o)

    # Maintain current state in x and previous state in x_previous
    x, x_previous = copy(initial_x), copy(initial_x)

    # Count the total number of iterations
    iteration = 0
    pseudo_iteration = 0

    # Track calls to function and gradient
    f_calls, g_calls = 0, 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in g and previous gradient in g_previous
    g, g_previous = Array(T, n), Array(T, n)

    # Store a history of changes in position and gradient
    rho = Array(T, mo.m)
    dx_history, dg_history = Array(T, n, mo.m), Array(T, n, mo.m)

    # The current search direction
    s = Array(T, n)

    # Buffers for use in line search
    x_ls, g_ls = Array(T, n), Array(T, n)

    # Store f(x) in f_x
    f_x_previous, f_x = NaN, d.fg!(x, g)
    f_calls, g_calls = f_calls + 1, g_calls + 1
    copy!(g_previous, g)

    # Keep track of step-sizes
    alpha = alphainit(one(T), x, g, f_x)

    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Buffers for new entries of dx_history and dg_history
    dx, dg = Array(T, n), Array(T, n)

    # Buffers for use by twoloop!
    twoloop_q, twoloop_alpha = Array(T, n), Array(T, mo.m)

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = o.store_trace || o.show_trace || o.extended_trace || o.callback != nothing
    @lbfgstrace

    # Assess multiple types of convergence
    x_converged, f_converged, g_converged = false, false, false

    # Iterate until convergence
    converged = false
    while !converged && iteration < o.iterations
        # Increment the number of steps we've had to perform
        iteration += 1
        pseudo_iteration += 1

        # Determine the L-BFGS search direction
        twoloop!(s, g, rho, dx_history, dg_history, mo.m, pseudo_iteration,
                 twoloop_alpha, twoloop_q)

        # Refresh the line search cache
        dphi0 = vecdot(g, s)
        if dphi0 > 0.0
            pseudo_iteration = 1
            @simd for i in 1:n
                @inbounds s[i] = -g[i]
            end
            dphi0 = vecdot(g, s)
        end

        clear!(lsr)
        push!(lsr, zero(T), f_x, dphi0)

        # Determine the distance of movement along the search line
        alpha, f_update, g_update =
          mo.linesearch!(d, x, s, x_ls, g_ls, lsr, alpha, mayterminate)
        f_calls, g_calls = f_calls + f_update, g_calls + g_update

        # Maintain a record of previous position
        copy!(x_previous, x)

        # Update current position
        @simd for i in 1:n
            @inbounds dx[i] = alpha * s[i]
            @inbounds x[i] = x[i] + dx[i]
        end

        # Maintain a record of the previous gradient
        copy!(g_previous, g)

        # Update the function value and gradient
        f_x_previous = f_x
        f_x = d.fg!(x, g)
        f_calls, g_calls = f_calls + 1, g_calls + 1

        # Measure the change in the gradient
        @simd for i in 1:n
            @inbounds dg[i] = g[i] - g_previous[i]
        end

        # Update the L-BFGS history of positions and gradients
        rho_iteration = 1 / vecdot(dx, dg)
        if isinf(rho_iteration)
            # TODO: Introduce a formal error? There was a warning here previously
            break
        end
        dx_history[:, mod1(pseudo_iteration, mo.m)] = dx
        dg_history[:, mod1(pseudo_iteration, mo.m)] = dg
        rho[mod1(pseudo_iteration, mo.m)] = rho_iteration

        x_converged,
        f_converged,
        g_converged,
        converged = assess_convergence(x,
                                       x_previous,
                                       f_x,
                                       f_x_previous,
                                       g,
                                       o.x_tol,
                                       o.f_tol,
                                       o.g_tol)

        @lbfgstrace
    end

    return MultivariateOptimizationResults("L-BFGS",
                                           initial_x,
                                           x,
                                           Float64(f_x),
                                           iteration,
                                           iteration == o.iterations,
                                           x_converged,
                                           o.x_tol,
                                           f_converged,
                                           o.f_tol,
                                           g_converged,
                                           o.g_tol,
                                           tr,
                                           f_calls,
                                           g_calls)
end
