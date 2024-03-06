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

function commutator_H(N, hx, hz, J; H="MF")
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
    

    elseif H=="H"

        E = 3*J
        for i=1:2*(N-1)

            H_op += (-1)^(i-1) * J/E,"Sx",i,"Sx",i+2
            H_op += (-1)^(i-1) * J/E,"Sy",i,"Sy",i+2
            H_op += (-1)^(i-1) * J/E,"Sz",i,"Sz",i+2
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
end

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
end
  
function measure_linkdim(; psi, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return maxlinkdim(psi)
    end
    return nothing
end


    
function model_C_early(r, t, v_B)
    return -(η[1] * ((r / η[3]) .- t) .^ (1 + η[2])) ./ (t .^ η[2])
end



function fit_growth_early(logC_r_t_array, r_array, times_array, p0,v_B)
    # logC_array: array of calculated logC(r,t) arrays
    # position_array: array of the positions r used in the calculation of C(r,t)
    # times_array: array of the time arrays corresponding to the logC(r,t) arrays
    # V_B: the butterfly velocity of the system
    # p0: [λ_p,p] initial guesses for these values used in the fitting of the early growth form

    # Initialize an empty array to store the indices for each sub-array
    indices_array = Vector{Vector{Int}}()

    # Iterate over each sub-array in logC_r_t_array
    for logC_r_t in logC_r_t_array
        # Find the indices where the values satisfy the condition
        indices = findall(x -> -50 < x < -10, logC_r_t)
        push!(indices_array, indices)
    end
    # Initialize arrays to store confined logC and times
    confined_logC_array = Vector{Vector{Float64}}()
    confined_times_array = Vector{Vector{Float64}}()

    # Iterate over each sub-array in logC_r_t_array
    for (logC_r_t, times, indices) in zip(logC_r_t_array, times_array, indices_array)
        # Extract the elements from logC_r_t_array and times_array corresponding to the indices
        confined_logC = logC_r_t[indices]
        confined_times = times[indices]

        # Store the confined sub-arrays in the respective arrays
        push!(confined_logC_array, confined_logC)
        push!(confined_times_array, confined_times)
    end

    #loop used to perform LsqFit on the confined spacetime region of logC
    # param_array=Vector{Vector{Float64}}() #initialise array to store fit parameters
    # for (r,logC,times) in zip(r_array,confined_logC_array,confined_times_array)
    #     @.model(t,p)=(-p[1]*((r/v_B)-t)^(1+p[2]))/(t^(p[2])) # model of logC_early
    #     xdata=times
    #     ydata=logC
    #     fit=curve_fit(model,xdata,ydata,p0) #lsq curve fit 
    #     params=coef(fit)
    #     push!(param_array,params)
        
    # end
    
    # #loop for calculating C_early with the calculated fitting parameters
    # logC_early_array= Vector{Vector{Float64}}()
    # for (r,times,params) in zip(r_array,confined_times_array,param_array)
    #     logC_early= model_C_early(r,times,params)
    
    #     push!(logC_early_array,logC_early) 
    # end  
      
    
     return confined_logC_array,confined_times_array#,param_array,logC_early_array 

end


######################################################################################################################################
function main(T=5.0, N=21; H="MF" )

    # N  Number of spins
    J  = 1.0    # ZZ interaction strength
    δt = 0.005   # Time-step for evolution
    # T  Total time
    χ  = 32;    # Max link dimension allowed


    if H=="MF" # mixed field ising
        hx = 1.05   # X-field 
        hz = 0.5    # Z-field
    

    elseif H=="TF" # transverse field ising
        hx = 1.05   # X-field 
        hz = 0.0    # Z-field
    
    elseif H=="H"
        hx = 0.0   # X-field 
        hz = 0.0    # Z-field
    
    end 
    sitesext = siteinds("S=1/2",2*N)#; # Make 2N S=1/2 spin indices defining system + ancilla

    
  
    Sp, Sm = raise_lower(sitesext,N)

    Id = identity(N, sitesext)

    # Construct the identity vacuum state:
    Ivac = MPS(sitesext, "Up") # All up spins initial state
    gates = [(Id[n]*Id[n+1] + Sm[n]*Sm[n+1]) for n in 1:2:(2*N)]; # Maps |00> => |00> + |11>
    Ivac = apply(gates, Ivac; cutoff=1e-10); # Note we have no 1/sqrt(2) normalisation

    H_op = commutator_H(N, hx, hz,J; H=H )
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
        # N = 201, require r < 100
        Sx_40_system, Sx_40_ancilla= local_op(N,sitesext;r=40)
        Sx_45_system, Sx_45_ancilla= local_op(N,sitesext;r=45)
        Sx_50_system, Sx_50_ancilla= local_op(N,sitesext;r=50)
        Sx_55_system, Sx_55_ancilla= local_op(N,sitesext;r=55)
        Sx_60_system, Sx_60_ancilla= local_op(N,sitesext;r=60)
        Sx_65_system, Sx_65_ancilla= local_op(N,sitesext;r=65)
        Sx_70_system, Sx_70_ancilla= local_op(N,sitesext;r=70)
        Sx_75_system, Sx_75_ancilla= local_op(N,sitesext;r=75)
        Sx_80_system, Sx_80_ancilla= local_op(N,sitesext;r=80)
        Sx_85_system, Sx_85_ancilla= local_op(N,sitesext;r=85)
        Sx_90_system, Sx_90_ancilla= local_op(N,sitesext;r=90)
      
      
        if bond == 1 && half_sweep == 2
          commutator_40 = compute_commutator(psi, Sx_40_system, Sx_40_ancilla)
          commutator_45 = compute_commutator(psi, Sx_45_system, Sx_45_ancilla)
          commutator_50 = compute_commutator(psi, Sx_50_system, Sx_50_ancilla)
          commutator_55 = compute_commutator(psi, Sx_55_system, Sx_55_ancilla)
          commutator_60 = compute_commutator(psi, Sx_60_system, Sx_60_ancilla)
          commutator_65 = compute_commutator(psi, Sx_65_system, Sx_65_ancilla)
          commutator_70 = compute_commutator(psi, Sx_70_system, Sx_70_ancilla)
          commutator_75 = compute_commutator(psi, Sx_75_system, Sx_75_ancilla)
          commutator_80 = compute_commutator(psi, Sx_80_system, Sx_80_ancilla)
          commutator_85 = compute_commutator(psi, Sx_85_system, Sx_85_ancilla)
          commutator_90 = compute_commutator(psi, Sx_90_system, Sx_90_ancilla)


          return [real.(inner(commutator_40, commutator_40)), real.(inner(commutator_45, commutator_45)) 
                  ,  real.(inner(commutator_50, commutator_50)), real.(inner(commutator_55, commutator_55))
                  ,  real.(inner(commutator_60, commutator_60)),  real.(inner(commutator_65, commutator_65))
                  ,  real.(inner(commutator_70, commutator_70)),  real.(inner(commutator_75, commutator_75))
                  ,  real.(inner(commutator_80, commutator_80)),  real.(inner(commutator_85, commutator_85))
                  , real.(inner(commutator_90, commutator_90))]
      
        end
        return nothing
      
    end;
    obs = Observer("times" => current_time, "SvN" => measure_SvN, "chi" => measure_linkdim,"Commutator"=>measure_commutator)



    # d|A(t)>/dt = i HC |A(t)> so |A(t)> = exp(i t HC)|A(0)> 
    ψf = tdvp(HC, im * T, Avec; 
            time_step = im * δt,
            normalize = false,
            maxdim = χ,
            cutoff = 1e-10,
            outputlevel=1,
            (observer!)=obs)

    # Extract results from time-step observations
    times=obs.times
    SvN=obs.SvN
    chi=obs.chi
    Commutator=obs.Commutator

  

    C_r_t_40 = []
    C_r_t_45 = []
    C_r_t_50 = []
    C_r_t_55 = []
    C_r_t_60 = []
    C_r_t_65 = []
    C_r_t_70 = []
    C_r_t_75 = []
    C_r_t_80 = []
    C_r_t_85 = []
    C_r_t_90 = []


    two = BigFloat(2)
    N_bf = BigFloat(N)

    for line in Commutator

  
        c_40 = 1/(two^N_bf) * line[1]
        c_45 = 1/(two^N_bf) * line[2]
        c_50 = 1/(two^N_bf) * line[3]
        c_55 = 1/(two^N_bf) * line[4]
        c_60 = 1/(two^N_bf) * line[5]
        c_65 = 1/(two^N_bf) * line[6]
        c_70 = 1/(two^N_bf) * line[7]
        c_75 = 1/(two^N_bf) * line[8]
        c_80 = 1/(two^N_bf) * line[9]
        c_85 = 1/(two^N_bf) * line[10]
        c_90 = 1/(two^N_bf) * line[11]

    
        push!(C_r_t_40, c_40)
        push!(C_r_t_45, c_45)
        push!(C_r_t_50, c_50)
        push!(C_r_t_55, c_55)
        push!(C_r_t_60, c_60)
        push!(C_r_t_65, c_65)
        push!(C_r_t_70, c_70)
        push!(C_r_t_75, c_75)
        push!(C_r_t_80, c_80)
        push!(C_r_t_85, c_85)
        push!(C_r_t_90, c_90)
    end


    C_r_t_array=[C_r_t_40,C_r_t_45,C_r_t_50,C_r_t_55,C_r_t_60,C_r_t_65,C_r_t_70,C_r_t_75,C_r_t_80,C_r_t_85,C_r_t_90]
    #logC_r_t_array=[log.(C) for C in C_r_t_array]

    times_array=[times,times,times,times,times,times,times,times,times,times,times]
    r_array=[40,45,50,55,60,65,70,75,80,85,90]
    # v_B=0.67 #swingle values (waiting on ours to calcualte)
    # p0=[1.9,0.67] #λ_p,p,V_B

    #logC_array_confined,times_confined,test_param_array,logC_early_array=fit_growth_early(logC_array,r_array,times_array,v_B,p0)
    # logC_array_confined, times_confined = fit_growth_early(logC_r_t_array, r_array, times_array, p0,v_B)

    # Write logC data to CSV file
    # Construct DataFrame
    df_logC = DataFrame(logC_40 = C_r_t_40,logC_45 = C_r_t_45, logC_50 = C_r_t_50,logC_55 = C_r_t_55, logC_60 = C_r_t_60,logC_65 = C_r_t_65, logC_70 = C_r_t_70,logC_75 = C_r_t_75, logC_80 = C_r_t_80,logC_85 = C_r_t_85, logC_90 = C_r_t_90)
    df_times = DataFrame(times_40 = times, times_45=times, times_50 = times, times_55=times, times_60 = times, times_65=times, times_70 = times, times_75=times, times_80 = times, times_85=times, times_90 = times)
    df_r = DataFrame(r = r_array)

    # Write DataFrame to CSV files
    CSV.write("C_array_MF.csv", df_logC)
    CSV.write("times_array_MF.csv", df_times)
    CSV.write("r_array_MF.csv", df_r)


    #S2 = reduce(vcat,transpose.(SvN))[:,2:2:(2*N)]
   
    # heatmap(1:N, times, S2, c=:seaborn_rocket_gradient,framestyle=:box,colorbar_title="S(r,t)",left_margin=40px,bottom_margin=20px,top_margin=20px,right_margin=40px)
    # xlabel!("r")
    # ylabel!("t")
    # #title!("Light Cone: Bonds seperating system and ancilla")
    # savefig("lightCone_XXX_Heisenberg.png")


end

  
# get values from ARGS
T, N, H = ARGS[1:end]


N = parse(Int64, N)
T = parse(Float64, T)
#Δ = parse(Float64, Δ)

main(T,N; H)

