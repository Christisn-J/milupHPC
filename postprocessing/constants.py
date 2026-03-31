si_prefixes = {
    "quetta": {"factor": 1e30, "abbr": "\mathrm{Q}"},
    "ronna":  {"factor": 1e27, "abbr": "\mathrm{R}"},
    "yotta":  {"factor": 1e24, "abbr": "\mathrm{Y}"},
    "zetta":  {"factor": 1e21, "abbr": "\mathrm{Z}"},
    "exa":    {"factor": 1e18, "abbr": "\mathrm{E}"},
    "peta":   {"factor": 1e15, "abbr": "\mathrm{P}"},
    "tera":   {"factor": 1e12, "abbr": "\mathrm{T}"},
    "giga":   {"factor": 1e9,  "abbr": "\mathrm{G}"},
    "mega":   {"factor": 1e6,  "abbr": "\mathrm{M}"},
    "kilo":   {"factor": 1e3,  "abbr": "\mathrm{k}"},
    "hecto":  {"factor": 1e2,  "abbr": "\mathrm{h}"},
    "deka":   {"factor": 1e1,  "abbr": "\mathrm{da}"},

    "none":   {"factor": 1e0,  "abbr": ""},

    "deci":   {"factor": 1e-1, "abbr": "\mathrm{d}"},
    "centi":  {"factor": 1e-2, "abbr": "\mathrm{c}"},
    "milli":  {"factor": 1e-3, "abbr": "\mathrm{m}"},
    "micro":  {"factor": 1e-6, "abbr": "\mathrm{\mu}"},
    "nano":   {"factor": 1e-9, "abbr": "\mathrm{n}"},
    "pico":   {"factor": 1e-12,"abbr": "\mathrm{p}"},
    "femto":  {"factor": 1e-15,"abbr": "\mathrm{f}"},
    "atto":   {"factor": 1e-18,"abbr": "\mathrm{a}"},
    "zepto":  {"factor": 1e-21,"abbr": "\mathrm{z}"},
    "yocto":  {"factor": 1e-24,"abbr": "\mathrm{y}"},
    "ronto":  {"factor": 1e-27,"abbr": "\mathrm{r}"},
    "quecto": {"factor": 1e-30,"abbr": "\mathrm{q}"},
}

# === Material definitions ===
MATERIALS = {
    "AL6061": {"density": 2700.0, "unit": f"{si_prefixes['kilo']['abbr']}g/m³", "name": "Aluminum 6061", "alias": "Al6061"},
    "STEEL": {"density": 7850.0, "unit": f"{si_prefixes['kilo']['abbr']}g/m³", "name": "Steel"},
    "COPPER": {"density": 8960.0, "unit": f"{si_prefixes['kilo']['abbr']}g/m³", "name": "Copper"},
    "ICE": {"density": 917.0, "unit": f"{si_prefixes['kilo']['abbr']}g/m³", "name": "Ice"},
    "BASALT": {"density": 917.0, "unit": f"{si_prefixes['kilo']['abbr']}g/m³", "name": "Basalt"},
}

# Metadata for each field: display name, LaTeX symbol, unit, and colormap
FIELD_META = {
    "t":        {"name": "Time",                "symbol": r"$t$",                   "unit": r"$\mathrm{s}$",                               "cmap": "gray_r"},
    "x":        {"name": "Position",            "symbol": r"$\pmb{r}$",             "unit": r"$\mathrm{m}$",                               "cmap": "gray_r"},
    "a":        {"name": "Acceleration",        "symbol": r"$\pmb{a}$",             "unit": r"$\frac{\mathrm{m}}{\mathrm{s}^2}$",          "cmap": "inferno"},
    "m":        {"name": "Mass",                "symbol": r"$m$",                   "unit": r"$\mathrm{kg}$",                              "cmap": "viridis"},
    "v":        {"name": "Velocity",            "symbol": r"$\pmb{v}$",             "unit": r"$\frac{\mathrm{m}}{\mathrm{s}}$",            "cmap": "plasma"},
    "rho":      {"name": "Density",             "symbol": r"$\varrho$",             "unit": r"$\frac{\mathrm{kg}}{\mathrm{m}^3}$",         "cmap": "viridis"},
    "p":        {"name": "Pressure",            "symbol": r"$P$",                   "unit": r"$\mathrm{Pa}$",                              "cmap": "plasma"},
    "e":        {"name": "specific Energy",     "symbol": r"$u$",                   "unit": r"$\frac{\mathrm{J}}{\mathrm{kg}}$",           "cmap": "inferno"},
    "E":        {"name": "Energy",              "symbol": r"$E$",                   "unit": r"$\mathrm{J}$",                               "cmap": "inferno"},
    "cs":       {"name": "Speed of Sound",      "symbol": r"$c_s$",                 "unit": r"$\frac{\mathrm{m}}{\mathrm{s}}$",            "cmap": "magma"},
    "proc":     {"name": "Process",             "symbol": r"$N_\mathrm{op}$",       "unit": r"$-$",                                        "cmap": "tab10"},
    "matId":    {"name": "Material ID",         "symbol": r"$N_\mathrm{mat, ID}$",  "unit": r"$-$",                                        "cmap": "tab10"},
    "sml":      {"name": "Smoothing Length",    "symbol": r"$h$",                   "unit": r"$\mathrm{m}$",                               "cmap": "viridis"},
    "noi":      {"name": "Number of Interactions","symbol": r"$N_\mathrm{oi}$",     "unit": r"$-$",                                        "cmap": "plasma"},
    "Sxx":      {"name": "Stress XX",           "symbol": r"$\sigma_{xx}$",         "unit": r"$\mathrm{Pa}$",                              "cmap": "coolwarm"},
    "Sxy":      {"name": "Stress XY",           "symbol": r"$\sigma_{xy}$",         "unit": r"$\mathrm{Pa}$",                              "cmap": "coolwarm"},
    "Sxz":      {"name": "Stress XZ",           "symbol": r"$\sigma_{xz}$",         "unit": r"$\mathrm{Pa}$",                              "cmap": "coolwarm"},
    "Syz":      {"name": "Stress YZ",           "symbol": r"$\sigma_{yz}$",         "unit": r"$\mathrm{Pa}$",                              "cmap": "coolwarm"},
    "drhodt":   {"name": "Density Rate",        "symbol": r"$\frac{d\rho}{dt}$",    "unit": r"$\frac{\mathrm{kg}}{\mathrm{m}^3\cdot\mathrm{s}}$", "cmap": "cividis"},
    "dedt":     {"name": "specific Energy Rate","symbol": r"$\frac{d\epsilon}{dt}$", "unit": r"$\frac{\mathrm{J}}{\mathrm{kg}\cdot\mathrm{s}}$", "cmap": "cividis"},
}