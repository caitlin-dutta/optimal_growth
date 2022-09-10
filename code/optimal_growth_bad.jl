####hello everybody!

using Parameters, Plots #read in necessary packages

#global variables instead of structs
@with_kw struct Primitives
    β = 0.99 #discount rate.
    θ = 0.36 #capital share
    δ = 0.025 #capital depreciation
    k_grid = collect(range(1.0, length = 1800, stop = 45.0)) #capital grid
    nk = length(k_grid) #number of capital elements
    markov::Array{Float64,2} = [0.977 0.023; 0.074 0.926] #markov
    z_grid::Array{Float64,1} = [1.25, 0.2] #productivity states
    nz::Int64 = length(z_grid)
end

mutable struct Results
    val_func::Array{Float64,2} #value function We have k and z
    pol_func::Array{Float64,2} #policy function
end

#function that solves model
function Solve_model()
    prim = Primitives()
    val_func = zeros(prim.nk, prim.nz)
    pol_func = zeros(prim.nk, prim.nz)
    res = Results(val_func,pol_func)
    V_iterate(prim, res) #value function iteration (function)
    prim, res
end

#Value iteration
function V_iterate(prim::Primitives, res::Results, tol::Float64 = 1e-3)
    error = 100
    n = 0
    while error>tol
        n+=1
        v_next = Bellman(prim,res)
        error = maximum(abs.(v_next - res.val_func)) # new error
        res.val_func = v_next
    end
    println("value function converged in", n , "iterations.")
end

#Bellman
function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, nz, nk, z_grid, k_grid, markov = prim
    v_next = zeros(nk, nz)

    for i_k = 1:nk, i_z = 1:nz #loop over k, z
        candidate_max = -1e10
        k, z = k_grid[i_k], z_grid[i_z]
        budget = z*k^k^θ  + (1-δ)*k

        for i_kp = 1:nk #loop over choice of k_prime
            kp = k_grid[i_kp]
            c = budget - kp # consumption
            if c>0
                val = log(c) + β * sum(res.val_func[i_kp,:].*markov[i_z,:]) #money
                if val>candidate_max
                    candidate_max = val
                    res.pol_func[i_k, i_z] = kp #update policy function
                end
            end
        end
        v_next[i_k, i_z] = candidate_max #update next guess of value function
    end
    v_next
end

prim, res = Solve_model()


#unpack our results and make some plots
@unpack val_func, pol_func = res
@unpack k_grid = prim

#plot value function
Plots.plot(k_grid,val_func, title="Value Functions", label = ["Good State" "Bad State"])

#plot policy functions
Plots.plot(k_grid, pol_func, title="Policy Functions", label = ["Good State" "Bad State"])
