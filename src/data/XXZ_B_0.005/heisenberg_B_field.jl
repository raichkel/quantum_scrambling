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
Pkg.add("LaTeXStrings")
Pkg.add("ColorSchemes")
Pkg.add("LsqFit")
Pkg.add("CSV")


using ITensors
using ITensorTDVP
using Plots
using Observers
using LinearAlgebra
using DataFrames
using Plots.PlotMeasures
using LaTeXStrings
using ColorSchemes
using LsqFit
using CSV


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


function commutator_H(N; Bx=0.01,Bz=0.01,Jx=1.0,Jy=0.75,Jz=1.25)
    # think? this is still mixed field Ising model......
    # Vectorisation approach used here is to stack matrix rows into a column vector.
    # This means that:
    # vec(AB) = A ⊗ I vec(B) =  I ⊗ B^T vec(A)
    # so |i><j| => |i> ⊗ |j>
    # vec(L A R) = L ⊗ R^T vec(A)

    # Define "Commutator" Hamiltonian operator terms:

    H_op = OpSum()
 

    E = 2*sqrt(Jx^2 + Jy^2 + Jz^2)


    for i=1:2*(N-1)
        # XXZ Heisenberg, use Jx=Jy

        H_op += (-1)^(i-1) * Jx/E,"Sx",i,"Sx",i+2
        H_op += (-1)^(i-1) * Jx/E,"Sy",i,"Sy",i+2
        H_op += (-1)^(i-1) * Jz/E,"Sz",i,"Sz",i+2
        
        # add external Z and optional X field
        H_op += (-1)^(i-1) * Bz,"Sz",i
        H_op += (-1)^(i-1) * Bx,"Sx",i
    
    
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

# Define function that calulates the commutator 
function compute_commutator(ψ,Sx_r_system,Sx_r_ancilla)
    Sx_a= apply(Sx_r_system,ψ;cutoff=1e-15)
    a_Sx= apply(Sx_r_ancilla,ψ;cutoff=1e-15)

    return Sx_a - a_Sx
end;


function local_op(N,sitesext;r) #function used to define local operator to track C(r,t)
    # require r < floor(Int, N/2) 
    site_index= 2*floor(Int,N/2+1)+2*r-1 #defining site index
    Sx_r_system= op("Sx", sitesext, site_index)
    Sx_r_ancilla= op("Sx", sitesext, site_index+1)
    
    return Sx_r_system, Sx_r_ancilla
end;
# Define observer functions for TDVP:

function current_time(; current_time, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return real(-im*current_time)
    end
      
    return nothing
end
  
function measure_SvN(; psi, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return entanglement_entropy(psi)-SvN_init
    end
    return nothing
end;
  
function measure_linkdim(; psi, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return maxlinkdim(psi)
    end
    return nothing
end;
  


  

function main(T=5.0, N=21; Bx=0.01,Bz=0.01, Jx=1.0, Jy=0.75, Jz=1.25 )

    # N  Number of spins
    J  = 1.0    # ZZ interaction strength
    δt = 0.005   # Time-step for evolution
    # T  Total time
    χ  = 32;    # Max link dimension allowed


    sitesext = siteinds("S=1/2",2*N)#; # Make 2N S=1/2 spin indices defining system + ancilla


    Sp, Sm = raise_lower(sitesext,N)

    Id = identity(N, sitesext)

    # Construct the identity vacuum state:
    Ivac = MPS(sitesext, "Up") # All up spins initial state
    gates = [(Id[n]*Id[n+1] + Sm[n]*Sm[n+1]) for n in 1:2:(2*N)]; # Maps |00> => |00> + |11>
    Ivac = apply(gates, Ivac; cutoff=1e-10); # Note we have no 1/sqrt(2) normalisation

    H_op = commutator_H(N; Bx=Bx, Bz=Bz, Jx=Jx,Jy=Jy,Jz=Jz)
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
    function measure_commutator(; psi, bond, half_sweep)
        Sx_5_system, Sx_5_ancilla= local_op(N,sitesext;r=5)
        Sx_10_system, Sx_10_ancilla= local_op(N,sitesext;r=10)
        Sx_20_system, Sx_20_ancilla= local_op(N,sitesext;r=15)
        Sx_30_system, Sx_30_ancilla= local_op(N,sitesext;r=20)
        Sx_40_system, Sx_40_ancilla= local_op(N,sitesext;r=25)
        Sx_50_system, Sx_50_ancilla= local_op(N,sitesext;r=17)
        Sx_60_system, Sx_60_ancilla= local_op(N,sitesext;r=19)
        Sx_70_system, Sx_70_ancilla= local_op(N,sitesext;r=22)
        Sx_80_system, Sx_80_ancilla= local_op(N,sitesext;r=23)
        Sx_90_system, Sx_90_ancilla= local_op(N,sitesext;r=8)
      
        if bond == 1 && half_sweep == 2
          commutator_5 = compute_commutator(psi, Sx_5_system, Sx_5_ancilla)
          commutator_10 = compute_commutator(psi, Sx_10_system, Sx_10_ancilla)  
          commutator_20 = compute_commutator(psi, Sx_20_system, Sx_20_ancilla)
          commutator_30 = compute_commutator(psi, Sx_30_system, Sx_30_ancilla)
          commutator_40 = compute_commutator(psi, Sx_40_system, Sx_40_ancilla)
          commutator_50 = compute_commutator(psi, Sx_50_system, Sx_50_ancilla)
          commutator_60 = compute_commutator(psi, Sx_60_system, Sx_60_ancilla)
          commutator_70 = compute_commutator(psi, Sx_70_system, Sx_70_ancilla)
          commutator_80 = compute_commutator(psi, Sx_80_system, Sx_80_ancilla)
          commutator_90 = compute_commutator(psi, Sx_90_system, Sx_90_ancilla)

        return [real.(inner(commutator_5, commutator_5)), real.(inner(commutator_10, commutator_10)),
                  real.(inner(commutator_20, commutator_20)), real.(inner(commutator_30, commutator_30)),
                  real.(inner(commutator_40, commutator_40)),  real.(inner(commutator_50, commutator_50)),
                  real.(inner(commutator_60, commutator_60)),  real.(inner(commutator_70, commutator_70)),
                  real.(inner(commutator_80, commutator_80)),  real.(inner(commutator_90, commutator_90))]
      
        end
        return nothing
      
    end;
    obs = Observer("times" => current_time, "SvN" => measure_SvN, "chi" => measure_linkdim,"Commutator"=>measure_commutator)



    # d|A(t)>/dt = i HC |A(t)> so |A(t)> = exp(i t HC)|A(0)> 
    ψf = tdvp(HC, im * T, Avec; 
            time_step = im * δt,
            normalize = true,
            maxdim = χ,
            cutoff = 1e-10,
            outputlevel=1,
            (observer!)=obs)
            

    # Extract results from time-step observations
    times=obs.times
    SvN=obs.SvN
    chi=obs.chi
    Commutator=obs.Commutator


    C_r_t_5 = []
    C_r_t_10 = []
    C_r_t_20 = []
    C_r_t_30 = []
    C_r_t_40 = []
    C_r_t_50 = []
    C_r_t_60 = []
    C_r_t_70 = []
    C_r_t_80 = []
    C_r_t_90 = []

    two = BigFloat(2)
    N_bf = BigFloat(N)

    for line in Commutator
        
        c_5 = 1/(two^N_bf)*line[1]
        c_10 = 1/(two^N_bf)*line[2] 
        c_20 = 1/(two^N_bf) * line[3]
        c_30 = 1/(two^N_bf) * line[4]
        c_40 = 1/(two^N_bf) * line[5]
        c_50 = 1/(two^N_bf) * line[6]
        c_60 = 1/(two^N_bf) * line[7]
        c_70 = 1/(two^N_bf) * line[8]
        c_80 = 1/(two^N_bf) * line[9]
        c_90 = 1/(two^N_bf) * line[10]

        push!(C_r_t_5, c_5)
        push!(C_r_t_10, c_10)
        push!(C_r_t_20, c_20)
        push!(C_r_t_30, c_30)
        push!(C_r_t_40, c_40)
        push!(C_r_t_50, c_50)
        push!(C_r_t_60, c_60)
        push!(C_r_t_70, c_70)
        push!(C_r_t_80, c_80)
        push!(C_r_t_90, c_90)
    end;
    # # Plot the entanglement entropy of each bond for system + ancilla:
    # gr()
    # heat = heatmap(1:(2*N), times, reduce(vcat,transpose.(SvN)), c = :seaborn_rocket_gradient,left_margin=40px,
    #     right_margin=40px, top_margin=40px,  bottom_margin = 40px, framestyle=:box, colorbar_title = "Entanglement Entropy")
    # ylabel!("Time t")
    # xlabel!("Site Index")
    # #title!("Entanglement Entropy system + ancilla")
    # savefig(heat, "heatmap.png")


    # # Plot the entanglement entropy for bonds separating system + ancilla pairs:
    # gr()
    # S = reduce(vcat,transpose.(SvN))[:,2:2:(2*N)]
    # heat1 = heatmap(1:N, times, S, c = :seaborn_rocket_gradient,left_margin=40px,
    #     right_margin=40px, top_margin=40px,  bottom_margin = 40px, framestyle=:box, colorbar_title = "Entanglement Entropy")
    # ylabel!("Time t")
    # xlabel!("Site Index")
    # #title!("Entanglement Entropy Separating System and Ancilla")
    # savefig(heat1,"heatmap_bonds_sep.png")

    # # Plot entanglement entropy of bonds between system + ancilla pairs:
    # gr()
    # S = reduce(vcat,transpose.(SvN))[:,1:2:(2*N)]
    # heat2 = heatmap(1:N, times, S,c = :seaborn_rocket_gradient,left_margin=40px,
    #     right_margin=40px, top_margin=40px,  bottom_margin = 40px, framestyle=:box, colorbar_title = "Entanglement Entropy")
    # ylabel!("Time t")
    # xlabel!("Site Index")
    # #title!("Entanglement Entropy Between System and Ancilla")
    # savefig(heat2,"heatmap_bonds_between.png")

    # # Plot the growth in the maximum link dimension with time:
    # plot(times, chi, label=false,left_margin=20px, right_margin=20px, top_margin=20px,  bottom_margin = 20px, framestyle=:box, linecolor=get(ColorSchemes.seaborn_rocket_gradient,0.1))  
    # xlabel!("Time")
    # ylabel!(L"$\chi$")
    # #scatter = scatter!(times, chi, label=false) 
    # savefig("scatter.png")



    # plot(times,log.(C_r_t_5), label="r=5", left_margin=20px,top_margin = 20px,
    #     right_margin=20px, bottom_margin = 20px, framestyle=:box, linecolor=get(ColorSchemes.seaborn_rocket_gradient,0.2))#, ylim=(min_y, 20)) #, ylim=(min_y, 20)
    # #scatter!(times,log.(C_r_t_2), ylim=(min_y, 20))
    # plot!(times,log.(C_r_t_10), label="r=10", linecolor=get(ColorSchemes.seaborn_rocket_gradient,0.3))#, ylim=(min_y, 20)) #!
    # #scatter!(times,log.(C_r_t_4), ylim=(min_y, 20))
    # plot!(times,log.(C_r_t_20), label="r=20", linecolor=get(ColorSchemes.seaborn_rocket_gradient, 0.4))#, ylim=(min_y, 20))
    # #scatter!(times,log.(C_r_t_6), ylim=(min_y, 20))
    # plot!(times,log.(C_r_t_30), label="r=30", linecolor=get(ColorSchemes.seaborn_rocket_gradient, 0.5))#, ylim=(min_y, 20))

    # plot!(times,log.(C_r_t_40), label="r=40", linecolor=get(ColorSchemes.seaborn_rocket_gradient,0.6))#, ylim=(min_y, 20))

    # plot!(times,log.(C_r_t_50), label="r=50", linecolor=get(ColorSchemes.seaborn_rocket_gradient, 0.65))#, ylim=(min_y, 20))

    # xlabel!("Time")
    # ylabel!("log(C(r,t))")

    # savefig("crt.png")


    C_r_t_array=[C_r_t_5, C_r_t_10, C_r_t_20, C_r_t_30, C_r_t_40, C_r_t_50]
    #logC_r_t_array=[log.(C) for C in C_r_t_array]

    r_array=[5,10,20,30,40,50]
    # v_B=0.67 #swingle values (waiting on ours to calcualte)
    # p0=[1.9,0.67] #λ_p,p,V_B

    #logC_array_confined,times_confined,test_param_array,logC_early_array=fit_growth_early(logC_array,r_array,times_array,v_B,p0)
    # logC_array_confined, times_confined = fit_growth_early(logC_r_t_array, r_array, times_array, p0,v_B)

    df_C = DataFrame(times = times, C_5 = C_r_t_5,C_10 = C_r_t_10, C_20 = C_r_t_20, C_30 = C_r_t_30,
    C_40 = C_r_t_40, C_50 = C_r_t_50, C_60 = C_r_t_60, C_70 = C_r_t_70, C_80 = C_r_t_80
    , C_90 = C_r_t_90, chi=chi)

    df_SvN = DataFrame(SvN, :auto)

    df_r = DataFrame(r = r_array)

    # Write DataFrame to CSV files
    CSV.write("C_array_MF.csv", df_C)
    CSV.write("SvN_array_MF.csv", df_SvN)




end


# get values from ARGS
T, N, Bx, Bz, Jx, Jy, Jz= ARGS[1:end]


N = parse(Int64, N)
T = parse(Float64, T)
Bx = parse(Float64, Bx)
Bz = parse(Float64, Bz)
Jx = parse(Float64, Jx)
Jy = parse(Float64, Jy)
Jz = parse(Float64, Jz)

main(T, N; Bx, Bz, Jx, Jy, Jz)

