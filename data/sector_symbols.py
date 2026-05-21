"""Curated sector-based symbol universe for the opportunity scanner."""

SECTOR_SYMBOLS: dict = {
    "quantum": {
        "name": "Quantum Computing",
        "color": "#818cf8",
        "symbols": ["IONQ", "RGTI", "QUBT", "QBTS", "ARQQ", "IBM"],
    },
    "photonics": {
        "name": "Photonics & Optics",
        "color": "#60a5fa",
        "symbols": ["IPGP", "COHR", "LITE", "VIAV", "NPKI"],
    },
    "ai": {
        "name": "Artificial Intelligence",
        "color": "#4ade80",
        "symbols": ["NVDA", "AMD", "PLTR", "AI", "SOUN", "BBAI", "SMCI"],
    },
    "semiconductor": {
        "name": "Semiconductors",
        "color": "#f59e0b",
        "symbols": [
            "TSM", "QCOM", "AVGO", "MRVL", "AMAT", "LRCX", "KLAC",
            "MU", "ASML", "ON", "TXN", "MPWR", "NXPI", "INTC", "WOLF",
        ],
    },
    "industrial": {
        "name": "Industrials",
        "color": "#94a3b8",
        "symbols": ["GE", "HON", "CAT", "DE", "EMR", "ROK", "ETN", "PH", "ITW", "IR"],
    },
    "construction": {
        "name": "Construction & Materials",
        "color": "#fb923c",
        "symbols": ["VMC", "MLM", "NUE", "STLD", "URI", "PWR", "BLDR", "X", "ACM"],
    },
    "energy": {
        "name": "Energy",
        "color": "#fbbf24",
        "symbols": ["XOM", "CVX", "COP", "SLB", "OXY", "EOG", "MPC", "VLO", "HAL", "DVN"],
    },
    "energy_transition": {
        "name": "Clean Energy & Grid",
        "color": "#34d399",
        "symbols": ["NEE", "FSLR", "ENPH", "CEG", "VST", "RUN", "BE", "PLUG"],
    },
    "data_center": {
        "name": "Data Centers & Infra",
        "color": "#22d3ee",
        "symbols": ["EQIX", "DLR", "VRT", "IREN", "WDC", "NTAP", "DELL", "HPE", "AMT"],
    },
    "cybersecurity": {
        "name": "Cybersecurity",
        "color": "#f87171",
        "symbols": ["CRWD", "PANW", "ZS", "FTNT", "S", "OKTA", "CYBR", "QLYS"],
    },
    "cloud": {
        "name": "Cloud & SaaS",
        "color": "#a78bfa",
        "symbols": ["CRM", "SNOW", "DDOG", "MDB", "NET", "NOW", "WDAY", "HUBS"],
    },
    "fintech": {
        "name": "Fintech & Crypto Infra",
        "color": "#e879f9",
        "symbols": ["SQ", "PYPL", "COIN", "HOOD", "AFRM", "NU", "SOFI"],
    },
    "biotech": {
        "name": "Biotech & Life Sciences",
        "color": "#86efac",
        "symbols": ["MRNA", "BNTX", "REGN", "VRTX", "GILD", "BIIB", "INCY"],
    },
    "defense": {
        "name": "Defense & Aerospace",
        "color": "#cbd5e1",
        "symbols": ["LMT", "RTX", "NOC", "GD", "HII", "KTOS", "AVAV"],
    },
}

ALL_HANDPICK_SYMBOLS: list = sorted({
    sym
    for sector_data in SECTOR_SYMBOLS.values()
    for sym in sector_data["symbols"]
})

SYMBOL_SECTORS: dict = {
    sym: [k for k, v in SECTOR_SYMBOLS.items() if sym in v["symbols"]]
    for sym in ALL_HANDPICK_SYMBOLS
}
