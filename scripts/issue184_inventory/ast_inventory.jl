# Metadata-only parser: never eval/include package code, never build a reference.
using SHA, Test
function type_identity(node)
 node isa Expr && node.head in (:struct,:abstract,:primitive) || error("not a type declaration")
 string(node.args[node.head == :struct ? 2 : 1])
end
if ARGS == ["--self-test"]
 @testset "exact original AST type identities" begin
  @test type_identity(Meta.parse("struct Plain; x::Int; end")) == "Plain"
  @test type_identity(Meta.parse("mutable struct Mutable; x::Int; end")) == "Mutable"
  @test type_identity(Meta.parse("abstract type Abstract <: Number end")) == "Abstract <: Number"
  @test type_identity(Meta.parse("primitive type Primitive 8 end")) == "Primitive"
  @test_throws ErrorException type_identity(Meta.parse("x = 1"))
 end
 exit(0)
end
length(ARGS)==1 || error("usage: ast_inventory.jl exact-reference-git-root")
const ROOT=realpath(ARGS[1])
const PACKAGES=[("Julia-mVMC",ROOT,"8bb1b9e8ae47b1512c00b321be05664ddcac0fd1",""),
 ("PfaPack",joinpath(ROOT,"PfaPack.jl"),"0dcf52c15caec63516d0703f36bfc8a4bc0e58d0","PfaPack.jl/"),
 ("SFMT",joinpath(ROOT,"SFMT.jl"),"1526553009f318ae78338151460fda78beadddc2","SFMT.jl/")]
escape(value)=replace(string(value),'\t'=>"\\t",'\n'=>"\\n",'\r'=>"\\r")
println("package\trevision\tsource\tsha256\tline\tkind\tmodule\tidentity\texpression")
function emit(pkg,rev,path,sha,line,kind,modules,identity,expression)
 println(join(escape.((pkg,rev,path,sha,line,kind,join(modules,"."),identity,expression)),"\t"))
end
function walk(node,pkg,rev,path,sha,line,modules)
 node isa Expr || return
 node.head in (:error,:incomplete) && error("invalid parsed source $path: $node")
 if node.head == :module
  walk(node.args[3],pkg,rev,path,sha,line,[modules;string(node.args[2])]); return
 end
 if node.head == :export
  for symbol in node.args
   emit(pkg,rev,path,sha,line,"export",modules,string(symbol),repr(node))
  end
 elseif node.head == :macrocall && string(node.args[1]) in ("@testset","Test.@testset")
  loc=findfirst(arg->arg isa LineNumberNode,node.args)
  actual=loc===nothing ? line : node.args[loc].line
  label=findfirst(arg->arg isa String,node.args)
  identity=label===nothing ? "DYNAMIC_OR_UNNAMED_TESTSET" : node.args[label]
  emit(pkg,rev,path,sha,actual,"testset",modules,identity,repr(node))
 elseif node.head in (:using,:import)
  emit(pkg,rev,path,sha,line,"import_or_alias",modules,"",repr(node))
 elseif node.head == :function
  emit(pkg,rev,path,sha,line,"method",modules,repr(node.args[1]),repr(node.args[1]))
 elseif node.head in (:struct,:abstract,:primitive)
  emit(pkg,rev,path,sha,line,"type",modules,type_identity(node),repr(node))
 elseif node.head == :(=) && node.args[1] isa Expr && node.args[1].head in (:call,:where,:(::))
  emit(pkg,rev,path,sha,line,"method",modules,repr(node.args[1]),repr(node.args[1]))
 elseif node.head == :(=) && node.args[1] isa Symbol
  emit(pkg,rev,path,sha,line,"binding",modules,string(node.args[1]),repr(node))
 elseif node.head == :call && node.args[1] == :include
  emit(pkg,rev,path,sha,line,"include",modules,"",repr(node))
 end
 for arg in node.args
  if arg isa LineNumberNode
   line=arg.line
  else
   walk(arg,pkg,rev,path,sha,line,modules)
  end
 end
end
for (pkg,root,rev,prefix) in PACKAGES
 entries=split(read(`git -C $root ls-tree -r $rev`,String),'\n';keepempty=false)
 for entry in entries
  meta,name=split(entry,'\t';limit=2)
  startswith(meta,"160000 ") && continue
  endswith(name,".jl") || continue
  text=read(`git -C $root show $rev:$name`,String)
  path=prefix*name; sha=bytes2hex(sha256(text))
  emit(pkg,rev,path,sha,1,"file",String[],"","")
  walk(Meta.parseall(text;filename=path),pkg,rev,path,sha,1,String[])
 end
end
