#############
# MWE for weird savefig() BoundsError - doesn't actually produce error tho
##############


import Pkg
using Pkg
Pkg.Registry.update()

Pkg.instantiate()

using Plots
using Plots.PlotMeasures
using LaTeXStrings

function C_early(r,t; λ=1.9, p=0.67, v_B=0.67)
    
    bracket = BigFloat(-λ*(((r/v_B)-t)^(1+p))/(t^p))
    return BigFloat(exp(bracket))
end

T=10.0
δt=0.05
rstep = 10
rmin = 40
rmax =  100#floor(N/2)-rstep
tmin = 5.0
rindex = Int((rmax-rmin)/rstep) 
#tindex = Int((T-tmin)/δt)
c_arr = zeros(Float64,(rindex,13))  


r_arr = collect(range(rmin,rmax,step=rstep))
t_arr = collect(range(tmin,T,step =δt))

C_r_t_5 = [BigFloat(5),BigFloat(6),BigFloat(7),BigFloat(8),BigFloat(9),BigFloat(2),BigFloat(3),BigFloat(4),BigFloat(5),BigFloat(6),BigFloat(7),BigFloat(8),BigFloat(9)]
C_r_t_10 = [5,6,7,8,9,2,3,4,5,6,7,8,9]
C_r_t_20 = [5,6,7,8,9,2,3,4,5,6,7,8,9]
C_r_t_30 = [5,6,7,8,9,2,3,4,5,6,7,8,9]
C_r_t_40 = [5,6,7,8,9,2,3,4,5,6,7,8,9]
C_r_t_50 = [5,6,7,8,9,2,3,4,5,6,7,8,9]
C_r_t_60 = [5,6,7,8,9,2,3,4,5,6,7,8,9]
C_r_t_70 = [5,6,7,8,9,2,3,4,5,6,7,8,9]
C_r_t_80 = [5,6,7,8,9,2,3,4,5,6,7,8,9]
C_r_t_90 = [5,6,7,8,9,2,3,4,5,6,7,8,9]

for i in CartesianIndices(c_arr)
    j = i[2]
    ii = i[1]
    r = BigFloat(r_arr[ii])
    r = real(r)
    t = BigFloat(t_arr[j])
    t = real(t)
    c = C_early(r,t)
    c_arr[i]=c
    
end

print(size(c_arr))
# need to change this so C_r_t_00 are same length as c_arr because they use different timescales rn

plot(log.(c_arr[1,:]),log.(C_r_t_40), left_margin=20px,framestyle=:box)
plot!(log.(c_arr[2,:]),log.(C_r_t_50), left_margin=20px)
plot!(log.(c_arr[3,:]),log.(C_r_t_60), left_margin=20px)
plot!(log.(c_arr[4,:]),log.(C_r_t_70), left_margin=20px)
plot!(log.(c_arr[5,:]),log.(C_r_t_80), left_margin=20px)
plot!(log.(c_arr[4,:]),log.(C_r_t_90), left_margin=20px)


xlabel!(L"log($C_{early}(r,t)$)")
ylabel!(L"log($C(r,t)$)")

savefig("log_log.png")
