"""Static check: names megatron.bridge imports from megatron.core that the installed core does not define."""
import ast, os, sys
SP = sys.argv[1]
roots = [os.path.join(SP, "megatron/bridge/training"), os.path.join(SP, "megatron/bridge/models"),
         os.path.join(SP, "megatron/bridge/data"), os.path.join(SP, "megatron/bridge/utils")]
def module_file(mod):
    base = os.path.join(SP, *mod.split("."))
    for p in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.exists(p): return p
def defined(path, name, seen=None):
    src = open(path).read(); tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name: return True
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets): return True
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name: return True
        if isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name.split(".")[0]) == name or a.name == "*" for a in n.names): return True
    if path.endswith("__init__.py") and os.path.exists(os.path.join(os.path.dirname(path), name + ".py")): return True
    if path.endswith("__init__.py") and os.path.isdir(os.path.join(os.path.dirname(path), name)): return True
    return False
missing = set()
for root in roots:
    for dp, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith(".py"): continue
            p = os.path.join(dp, f)
            for n in ast.walk(ast.parse(open(p).read())):
                if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("megatron.core"):
                    mf = module_file(n.module)
                    for a in n.names:
                        if mf is None: missing.add((n.module, "<module>", os.path.relpath(p, SP)))
                        elif a.name != "*" and not defined(mf, a.name): missing.add((n.module, a.name, os.path.relpath(p, SP)))
for m in sorted(missing): print(*m)
print(len(missing), "unresolved")
