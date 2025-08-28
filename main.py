import importlib, traceback

modules = [
    "qrl",
    "qrl.env",
    "qrl.env.core",
    "qrl.env.core.bloch_sphere",
    "qrl.env.core.expressibility",
    "qrl.env.core.error_channel",
    "qrl.env.core.probability",
    "qrl.env.core.compiler",
    "qrl.agents",
    "qrl.agents.base",
    "qrl.agents.agents",
]

for m in modules:
    print(f">>> importing {m}")
    try:
        importlib.import_module(m)
        print("OK:", m)
    except Exception as e:
        print("FAIL:", m, "-", e)
        traceback.print_exc()
        break
