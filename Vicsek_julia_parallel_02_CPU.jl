#Vicsek model simulation after Chat� et al Modeling collective motion:
#Parallelization of code

#including more optimizations and benchmarking
clearconsole()
using Plots, Parameters, Statistics, DelimitedFiles
using BenchmarkTools, Profile
using Distributed
gr(size = (500,500))
addprocs(6)
nworkers()
@everywhere using SharedArrays

const Maxsteps = 5000
const rc = 1.0
const NParts = 10000
const v0 = 0.5
#const density = 4.0
const ethamax = 1.0
const dt = 1.0
const BoxL = 16.0


sppp(Maxsteps,rc,NParts,v0,BoxL,ethamax,dt)

# using Traceur
# @trace sppp(Maxsteps,rc,NParts,v0,BoxL,ethamax,dt)

foo("not")
function foo(word)
    clearconsole()
    #@everywhere GC.gc()
    #GR.redraw()
    N = (40,100,400,4000,10000)
    BoxL = (3.1,5.0,10.0,31.6,50.0)
    if word == "timeseries"
        x = collect(1000:500:Maxsteps)
        v = zeros(length(x))
        NParts = N[3]
        L = BoxL[3]
        for etha in ethamax:ethamax
            jj = 0
            for t in x
                jj +=1
                v[jj] = sppp(t,rc,NParts,v0,L,etha,dt)
                if t % 100 == 0 ; println("etha =",etha," timestep = ",
                    t," Va = ",v[jj]) end
            end
            p = plot!(x,v,xlims=(1000,Maxsteps),ylims=(0.0,1.0))
            display(p)
        end
    else
        f = open("order_parameter.txt","w")
        ruido = 0.0:0.2:ethamax
        v = zeros(length(ruido),length(N))
        jj = 0
        @inbounds for NParts in N
            ii = 0
            jj += 1
            L = BoxL[jj]
            @inbounds for etha in ruido
                ii += 1
                println("NParts = ",NParts," etha = ",etha)
                v[ii,jj] = sppp(Maxsteps,rc,NParts,v0,L,etha,dt)
            end
            p1=plot(ruido,v,markershapes = [:square :+ :x :utriangle :diamond],
                    xlims=(0.0,ethamax),ylims=(0.0,1.0),linewidth = 2)
            display(p1)
            png(p1,"test_fig2")
        end
        writedlm(f,[ruido v])
    end
    close(f)
end

function sppp(Maxsteps,rc,NParts,v0,L,etha,dt)
#sppp stands for self-propelling-particles-Parallelization
    #@everywhere GC.gc()
    Lhalf = L*0.5
    #initial positions on a square of side L
    xini = SharedArray(-Lhalf.+L*rand(NParts))
    yini = SharedArray(-Lhalf.+L*rand(NParts))
    LhalfTrunc = trunc(Lhalf,digits=1)
    #initial velocities directions
    thetaini = rand(NParts)*2*pi
    SinThetaini = SharedArray(sin.(thetaini))
    CosThetaini = SharedArray(cos.(thetaini))
    # vxini = SharedArray(v0.*CosThetaini)
    # vyini = SharedArray(v0.*SinThetaini)
    vxini = SharedArray(v0.*CosThetaini)
    vyini = SharedArray(v0.*SinThetaini)

    # p1 = quiver(xini[:],yini[:],quiver=(vxini[:],vyini[:]),
    #        lims = (-Lhalf,Lhalf))
    # display(p1)
    #plot_debug(xini,yini,vxini,vyini,L,NParts,rc,1,"quiver")
    #vector initialization
    vx = zeros(NParts)
    vy = zeros(NParts)
    x = zeros(NParts)
    y = zeros(NParts)
    AvgSinA = zeros(NParts)
    AvgCosA = zeros(NParts)
    theta = zeros(NParts)
    #theta = SharedArray(zeros(NParts))
    @inbounds for step =1:Maxsteps #loop over all timesteps
       #println(step)
       #Angle interaction calculations
       AvgSinA,AvgCosA = calc_interactions(NParts,xini,yini,L,SinThetaini,CosThetaini)
       noise = rand(NParts)*2*pi*etha
       xini,yini,vx,vy,SinThetaini,CosThetaini = evolution(AvgSinA,AvgCosA,NParts,xini,yini,L,vxini,vyini,noise)
       #calculating new velocities
       # vx = pmap(x->v0*cos.(x),theta)
       # vy = pmap(x->v0*sin.(x),theta)
       #calculating new positions using forward differences
       # x = @. xini + dt*vx
       # y = @. yini + dt*vy
       # #periodic boundary conditions
       # x = @. x - L*round(x/L)
       # y = @. y - L*round(y/L)
       #velocity vector field graph
       # quiver(x[:],y[:],quiver=(vx[:],vy[:]),
       #        lims = (-Lhalf,Lhalf))
       # scatter(xini,yini, series_annotations = text.(1:NParts,:bottom),
       #         lims = (-LhalfTrunc,LhalfTrunc),
       #         ticks = Ticks)
       #positions and velocities update
       # xini = x
       # yini = y
       # vxini = vx
       # vyini = vy
       #
       # thetaini = theta
       # SinThetaini = sin.(theta)
       # CosThetaini = cos.(theta)
    end #every 1
    va = sqrt(sum(vx)^2+sum(vy)^2)/v0/NParts;
    finalize(xini)
    finalize(yini)
    finalize(vxini)
    finalize(vyini)
    finalize(AvgSinA)
    finalize(AvgCosA)
    finalize(SinThetaini)
    finalize(CosThetaini)
    return va
end

function evolution(AvgSinA,AvgCosA,NParts,xini,yini,L,vxini,vyini,noise)
    @inbounds for i in 1:NParts
        if AvgSinA[i]>0.0 && AvgCosA[i]>0.0
            theta = atan(AvgSinA[i]/AvgCosA[i]) + noise[i]
        elseif (AvgSinA[i]>0.0 && AvgCosA[i]<0.0) ||  (AvgSinA[i]<0.0 && AvgCosA[i]<0.0)
            theta = atan(AvgSinA[i]/AvgCosA[i]) + pi + noise[i]
        else
            theta = atan(AvgSinA[i]/AvgCosA[i]) + 2*pi + noise[i]
        end
        vx = v0*cos(theta)
        vy = v0*sin(theta)
        x =  xini[i] + dt*vx
        y =  yini[i] + dt*vy
        #periodic boundary conditions
        x = x - L*round(x[i]/L)
        y = y - L*round(y[i]/L)
        xini[i] = x
        yini[i] = y
        vxini[i] = vx
        vyini[i] = vy
        #thetaini[i] = theta
        AvgSinA[i] = sin(theta)
        AvgCosA[i] = cos(theta)
    end
    xini, yini, vxini, vyini, AvgSinA, AvgCosA
end


function calc_interactions(NParts,xini,yini,L,SinThetaini,CosThetaini)
    @inbounds @sync @distributed for ip = 1:NParts
        SinThetaini[ip],CosThetaini[ip] = calc_averages(ip,NParts,rc,xini,yini,L,SinThetaini,CosThetaini)
    end
    return SinThetaini, CosThetaini
end

@everywhere function calc_averages(ip,NParts,rc,xini,yini,L,SinThetaini,CosThetaini)
    INcircle = 0;sintheta=0.0;costheta=0
    @inbounds for jp = 1:NParts
        dx = xini[ip]-xini[jp]
        dy = yini[ip]-yini[jp]
        #minimum image convention
        dx = dx-L*round(dx/L)
        dy = dy-L*round(dy/L)
        dr = sqrt(dx^2+dy^2)
        if (dr - rc) <= 0.0 #fill A with particles inside the ball
            INcircle +=1
            sintheta += SinThetaini[jp]
            costheta += CosThetaini[jp]
        end
    end
    return sintheta/INcircle, costheta/INcircle
end


#************************************************************************

function plot_debug(xini,yini,vxini,vyini,L,NParts,rc,point,string)
    coordx = [xini.+L xini.-L xini xini xini.+L xini.+L xini.-L xini.-L]
    coordy = [yini yini yini.+L yini.-L yini.+L yini.-L yini.+L yini.-L]
    Ticks = collect(-L:0.5*L:L)
    scale = 1.0
    vx = @. scale * vxini
    vy = @. scale * vyini
    if string == "scatter"
        p1=scatter(xini,yini, series_annotations = text.(1:NParts,:bottom),
                   legend=false,ms=2,lims=(-L*0.5,L*0.5),
                   ticks=Ticks)
        scatter!(coordx,coordy, series_annotations = text.(1:NParts,:bottom),
                legend=false,color=:green,ms=2,lims=(-L*0.5-L,L*0.5+L),
                ticks=Ticks)
    elseif string == "quiver"
        p1 = quiver(xini[:],yini[:],quiver=(vx[:],vy[:]),
                   lims = (-L*0.5,L*0.5),ticks=Ticks,legend=false,
                   color=:blue)
        quiver!(coordx[:],coordy[:],quiver=(vx[:],vy[:]),
                  lims = (-L*0.5-L,L*0.5+L),ticks=Ticks,legend=false,
                  color=:green)
    end
    scatter!(xini,yini, series_annotations = text.(1:NParts,:bottom),
               legend=false,ms=2,lims=(-L*0.5,L*0.5),
               ticks=Ticks,color=:blue)
    scatter!(coordx,coordy, series_annotations = text.(1:NParts,:bottom),
            legend=false,color=:green,ms=2,lims=(-L*0.5-L,L*0.5+L),
            ticks=Ticks)


    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    plot!(rectangle(L,L,-L*0.5,-L*0.5),color=:yellow, opacity=.5)

    t = range(0,stop=2*pi,length=50)
    ut = xini[point] .+ rc.*cos.(t)
    vt = yini[point] .+ rc.*sin.(t)
    plot!(ut,vt,color="red")
    display(p1)

end
