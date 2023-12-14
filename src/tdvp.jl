import Pkg
using Pkg
Pkg.Registry.update()

Pkg.instantiate()
Pkg.add("ITensors")
Pkg.add("ITensorTDVP")
Pkg.add("Observers")
Pkg.add("Plots")
Pkg.add("LinearAlgebra")
Pkg.add("DataFrames")


using ITensors
using ITensorTDVP
using Plots
using Observers
using LinearAlgebra
using DataFrames


# Define mixed-field Ising Hamiltionian operator terms:

# H_op = OpSum()
# for i=1:N-1
#     H_op += 1.0,"Sz",i,"Sz",i+1 # ZZ terms
# end
# for i=1:N
#     H_op +=  hx,"Sx",i         # X terms
#     H_op +=  hz,"Sz",i         # Z terms
# end
# # Convert these terms to an MPO
# H = MPO(H_op,sites);



# Extract the raising, lowering and identity operators for the extended system:
function raise_lower(sitesext,N)
    Sp = ops(sitesext, [("S+", n) for n in 1:(2*N)]);  # Raising operators
    Sm = ops(sitesext, [("S-", n) for n in 1:(2*N)]);  # Lowering operators

    return Sp,Sm
end

function identity(N, sitesext)
    # ITensors doesn't include the identity operator as standard so construct it:
    Id = Vector{ITensor}(undef,2*N)
    for i =1:(2*N)
        iv = sitesext[i]
        ID = ITensor(iv', dag(iv));
        for j in 1:ITensors.dim(iv)
            ID[iv' => j, iv => j] = 1.0
        end
        Id[i] = ID
    end
    return Id
end


function commutator_H(N, hx, hz, J; H="MF", τ=0, K=0)
    # think? this is still mixed field Ising model......
    # Vectorisation approach used here is to stack matrix rows into a column vector.
    # This means that:
    # vec(AB) = A ⊗ I vec(B) =  I ⊗ B^T vec(A)
    # so |i><j| => |i> ⊗ |j>
    # vec(L A R) = L ⊗ R^T vec(A)

    # Define "Commutator" Hamiltonian operator terms:

    H_op = OpSum()
    if H=="MF" || H=="TF" 
        for i=1:2*(N-1)
            H_op += (-1)^(i-1) *  J,"Sz",i,"Sz",i+2
        end
        for i=1:2*N
            H_op += (-1)^(i-1) *  hx,"Sx",i
            H_op += (-1)^(i-1) *  hz,"Sz",i
        end

    elseif H=="K"

        # Ising interaction term with transverse and longitudinal fields
        # hx: transverse
        # hz: longitudinal
        for i=1:2*(N-1)
    
            H_op += (-1)^(i-1) * J,"Sz",i,"Sz",i+2

            H_op += (-1)^(i-1) * hz,"Sx",i
        end

        # periodic boundary conditions?
        # H_op += (-1)^(i-1) * "Sz",N,1 % N + 1,J
        # H_op += (-1)^(i-1) * "Sx",L,hz


        # External field term
        for i=1:2*N 
            
            H_op += (-1)^(i-1) *  hx,"Sx",i


        end

        # Kicked term
        for  i=1:2*N 
            H_op +=  (-1)^(i-1) * K,"Sz",i,"Sz",i,"t",τ
        end

    
    end 

    return H_op
end 

# Define function for computing entanglement entropy

function entanglement_entropy(ψ)
    # Compute the von Neumann entanglement entropy across each bond of the MPS
    N = length(ψ)
    SvN = zeros(N)
    psi = ψ
    for b=1:N
        psi = orthogonalize(psi, b)
        if b==1
            U,S,V = svd(psi[b] , siteind(psi, b))
        else
            U,S,V = svd(psi[b], (linkind(psi, b-1), siteind(psi, b)))
        end
        p = diag(S).^2               # Extract square of Schmidt coefficients
        p = p ./ sum(p)              # Normalise to a probability dist
        SvN[b] = -sum(p .* log2.(p)) # Compute Shannon entropy
    end
    return SvN
end;

   
# Define observer functions for TDVP:

function current_time(; current_time, bond, half_sweep)
    if bond == 1 && half_sweep == 2
        return real(-im*current_time)
    end
        
    return nothing
end
  
# function measure_SvN(; psi, bond, half_sweep)
#     if bond == 1 && half_sweep == 2
#     return entanglement_entropy(psi) - SvN_init
#     end
#     return nothing
# end
  
function measure_linkdim(; psi, bond, half_sweep)
    if bond == 1 && half_sweep == 2
        return maxlinkdim(psi)
    end
    return nothing
end;




function main(T=5.0, N=21; H="M", τ = 0.1, K=1.0)

    # N  Number of spins
    J  = 1.0    # ZZ interaction strength
    δt = 0.05   # Time-step for evolution
    # T  Total time
    χ  = 32;    # Max link dimension allowed


    if H=="M" # mixed field ising
        hx = 1.05   # X-field 
        hz = 0.5    # Z-field
    

    elseif H=="TF" # transverse field ising
        hx = 1.05   # X-field 
        hz = 0.0    # Z-field
    
    elseif H=="K" # kicked ising
        hx = 1.05   # X-field 
        hz = 0.0    # Z-field

    end 

    sitesext = siteinds("S=1/2",2*N)#; # Make 2N S=1/2 spin indices defining system + ancilla

    
  

    Sp, Sm = raise_lower(sitesext,N)

    Id = identity(N, sitesext)

    # Construct the identity vacuum state:
    Ivac = MPS(sitesext, "Up") # All up spins initial state
    gates = [(Id[n]*Id[n+1] + Sm[n]*Sm[n+1]) for n in 1:2:(2*N)]; # Maps |00> => |00> + |11>
    Ivac = apply(gates, Ivac; cutoff=1e-10); # Note we have no 1/sqrt(2) normalisation

    H_op = commutator_H(N, hx, hz,J; H=H, τ = τ, K=K)
    # HC = H ⊗ I - I ⊗ H, since H is real and hermitian H = H^T.
    # Convert these terms to an MPO
    HC = MPO(H_op,sitesext)#;

    # Define observable for scrambling:

    A_op = OpSum()
    A_op += 1.0,"Sx",2*floor(Int,N/2+1)-1  # Sx operator in the middle of the system
    A = MPO(A_op,sitesext);                # Build the MPO from these terms
    Avec = apply(A, Ivac; cutoff=1e-15);   # Compute |A> = A|I>

    SvN_init = entanglement_entropy(Avec)

    function measure_SvN(; psi, bond, half_sweep)
        if bond == 1 && half_sweep == 2
        return entanglement_entropy(psi) - SvN_init
        end
        return nothing
    end

    # Perform TDVP evolution of |A(t)>:
    obs = Observer("times" => current_time, "SvN" => measure_SvN, "chi" => measure_linkdim)
    #print(obs)

    #print("colmetadata(obs,:'times'):")
    #print(colmetadata(obs,:"times"))
    # d|A(t)>/dt = i HC |A(t)> so |A(t)> = exp(i t HC)|A(0)> 
    ψf = tdvp(HC, im * T, Avec; time_step = im * δt,normalize = false,maxdim = χ,cutoff = 1e-10,outputlevel=1,(observer!)=obs);
    # NamedTuple error: NamedTuple is bit after ;
    # trying to input SvN as a part of this NamedTuple 

    #print(size(obs))
    # Extract results from time-step observations
    times = obs.times
    SvN = obs.SvN
    chi = obs.chi
    
    
    # Plot the entanglement entropy of each bond for system + ancilla:
    gr()
    heat = heatmap(1:(2*N), times, reduce(vcat,transpose.(SvN)), c = :heat)
    savefig(heat,"heatmap_kicked.png")
    # Plot the entanglement entropy for bonds separating system + ancilla pairs:
    gr()
    S = reduce(vcat,transpose.(SvN))[:,2:2:(2*N)]
    heat1 = heatmap(1:N, times, S, c = :heat)
    savefig(heat1,"heatmap_bonds_sep_kicked.png")

    # Plot entanglement entropy of bonds between system + ancilla pairs:
    gr()
    S = reduce(vcat,transpose.(SvN))[:,1:2:(2*N)]
    heat2 = heatmap(1:N, times, S, c = :heat)
    savefig(heat2,"heatmap_bonds_between_kicked.png")

    # Plot the growth in the maximum link dimension with time:
    plot(times, chi, label=false)  
    scatter = scatter!(times, chi, label=false) 
    savefig(scatter,"scatter_kicked.png")


end


# get values from ARGS
T, N, H, τ,K = ARGS[1:end]


N = parse(Int64, N)
T = parse(Float64, T)
τ = parse(Float64, τ)
K = parse(Float64, K)

main(T,N; H, τ,K)


