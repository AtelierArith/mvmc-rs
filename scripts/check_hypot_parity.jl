using Test
VERSION == v"1.13.1" || error("Hypot fixture requires Julia 1.13.1")
path=joinpath(@__DIR__,"..","tests","fixtures","hypot.txt")
hex(v)=join(string.(reinterpret.(UInt64,v);base=16,pad=16)," ")
io=IOBuffer()
for (x,y) in ((3.,4.),(1.3,2.7),(-4.2,.7),(2.425211183513878,3.1809355094624614),(1e300,2e299),(1e-300,2e-299),(1e-160,2e-160),(1.,1e-20),(0.,-0.),(Inf,1.))
    println(io,hex([x,y,hypot(x,y)]))
end
actual=String(take!(io))
if "--write" in ARGS;write(path,actual);else;@test actual==read(path,String);end
