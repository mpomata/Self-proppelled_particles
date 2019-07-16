#Vicsek model simulation after Chat� et al Modeling collective motion:
#variations on the Vicsek model
#Cell-linked optimization
clearconsole()
using Plots, Parameters, BenchmarkTools, Statistics
gr()

Maxsteps = 2000
rc = 1.0
NParts = 1000
v0 = 0.3
density = 0.5
etha = 0.2
dt = 1.0

spp2(Maxsteps,rc,NParts,v0,density,etha,dt)

function spp2(Maxsteps,rc,NParts,v0,density,etha,dt)
    L = sqrt(NParts/density) #box lenght
    Lhalf = L*0.5
    m = Int(ceil(L/(1.1*rc)))
    dtick = trunc(L/m,digits=0)
    ϵ = 0.0
    rcsqr = rc^2
    #initial positions on a square of side L
    xini = rand(-Lhalf+ϵ:0.01:Lhalf-ϵ,NParts,1)
    yini = rand(-Lhalf+ϵ:0.01:Lhalf-ϵ,NParts,1)
    LhalfTrunc = trunc(Lhalf,digits=0)
    Ticks = collect(-LhalfTrunc:dtick:LhalfTrunc)
    #initial velocities directions
    thetaini = rand(-pi:0.01:pi,NParts,1)
    vxini = v0*cos.(thetaini)
    vyini = v0*sin.(thetaini)
    p1 = quiver(xini[:],yini[:],quiver=(vxini[:],vyini[:]),
                lims = (-Lhalf,Lhalf),ticks = Ticks)
    # p1=scatter(xini,yini, series_annotations = text.(1:NParts,:bottom),
    #         lims = (-LhalfTrunc,LhalfTrunc),
    #         ticks = Ticks)
    display(p1)
    #vector initialization
    vx = zeros(NParts,1)
    vy = zeros(NParts,1)
    x = zeros(NParts,1)
    y = zeros(NParts,1)
    ncell = m*m
    map = mapas(m)
    @gif for n =1:Maxsteps
       vxtot=0.0;vytot=0.0
    #loop over all cells
       head,list = linkos(ncell,m,rc,NParts,L,xini/L,yini/L)
       Angles = zeros(NParts,2)
       for icell = 1:ncell
           vxavg=0.0;vyavg=0.0;
           i = head[icell]
           #loop over particles in the cell
           PartsCell = 0
           while i>0
               rx = xini[i]
               ry = yini[i]
               #loop over all particles below i in the current cell
               j = list[i]
               Angles[i,1]=1
               Angles[i,2]=thetaini[i]
               while j>0;
                   dx = rx-xini[j]
                   dy = ry-yini[j]
                   #minimum image convention
                   dx = dx-round(dx)
                   dy = dy-round(dy)
                   drsqr = dx^2+dy^2
                   if drsqr<rcsqr #fill Angles[] with particles inside the ball
                       Angles[i,1] +=1
                       Angles[i,2] +=thetaini[j]
                   end
                   j = list[j]
               end
               #loop over neighboring cells
               jcell0 = 4*(icell-1)
               for nabor = 1:4
                   jcell = map[jcell0+nabor]
                   if jcell > 0
                   #loop over all particles in neighboring cells
                       j = head[jcell]
                       while j != 0
                           dx = rx-xini[j]
                           dy = ry-yini[j]
                           #minimum image convention
                           dx = dx-round(dx)
                           dy = dy-round(dy)
                           drsqr = dx^2+dy^2
                           if drsqr < rcsqr #fill Angles[] with particles inside the ball
                               Angles[i,1] +=1
                               Angles[i,2] +=thetaini[j]
                           end
                           j = list[j]
                       end
                   end
               end
               i = list[i]
           end
       end
       #After parsing all particles in all cells we calculate de average vel
       # using the average angles within the ball with random noise
       Angles_avg = Angles[:,2]./Angles[:,1] .+ rand(-0.5*etha:0.01:0.5*etha)
       #calculating new velocities
       vx = v0*cos.(Angles_avg)
       vy = v0*sin.(Angles_avg)
       #calculating new positions
       x = xini + dt.*vx
       y = yini + dt.*vy
       #periodic boundary conditions
       for idx in 1:NParts
           if x[idx] <  Lhalf; x[idx] += L end
           if x[idx] >= Lhalf; x[idx] -= L end
           if y[idx] <  Lhalf; y[idx] += L end
           if y[idx] >= Lhalf; y[idx] -= L end
       end
#       psi[n] = sqrt(vxtot^2+vytot^2)/v0/NParts;
       #velocity vector field graph
       quiver(x[:],y[:],quiver=(vx[:],vy[:]),
              lims = (-Lhalf,Lhalf), ticks = Ticks)
       #positions and velocities update
       xini = x;
       yini = y;
       vxini = vx;
       vyini = vy;
   end every 1
end


#************Functions definitions****************
function mapas(m)
    ncell = m*m
    mapsize = 4*ncell
    maps = zeros(Int64,mapsize,1);
#half of cells
    for iy=1:m
        for ix=1:m
            imap = (icell(ix,iy,m)-1)*4
            maps[imap+1]=icell(ix+1,iy  ,m)
            maps[imap+2]=icell(ix+1,iy+1,m)
            maps[imap+3]=icell(ix  ,iy+1,m)
            maps[imap+4]=icell(ix-1,iy+1,m)
        end
    end
    return maps
end
#***********************************************************************
function icell(ix,iy,m)
  1+rem(ix-1+m,m)+rem(iy-1+m,m)*m
end
#***********************************************************************
function linkos(ncell,m,rc,Nparts,L,rx,ry)
   head = zeros(Int64,ncell,1)
   list = zeros(Int64,Nparts,1)
   celli = m
   cell = 1/m
   if cell < rc/celli
      println("cell dimension is smaller than rc")
   end
#sort all particles
   for i = 1:Nparts
      iicell = 1+floor((rx[i]+0.5)*celli)+floor((ry[i]+0.5)*celli)*m
      icell = Int64(iicell)
      #scatter!(rx[i],ry[i])
      list[i] = head[icell]
      head[icell] = i
   end
   return head, list
end
#************************************************************************
