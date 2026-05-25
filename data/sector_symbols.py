"""Expanded sector universe — 500+ symbols across 30+ industries."""

SECTOR_SYMBOLS: dict = {
    "ai": {
        "name": "Artificial Intelligence",
        "color": "#4ade80",
        "symbols": [
            "NVDA", "AMD", "PLTR", "AI", "SOUN", "BBAI", "SMCI", "PATH",
            "ORCL", "IBM", "BIDU", "SNOW", "DDOG", "MDB", "ESTC",
        ],
    },
    "quantum": {
        "name": "Quantum Computing",
        "color": "#818cf8",
        "symbols": ["IONQ", "RGTI", "QUBT", "QBTS", "ARQQ"],
    },
    "semiconductor": {
        "name": "Semiconductors",
        "color": "#f59e0b",
        "symbols": [
            "TSM", "QCOM", "AVGO", "MRVL", "AMAT", "LRCX", "KLAC",
            "MU", "ASML", "ON", "TXN", "MPWR", "NXPI", "INTC", "WOLF",
            "ADI", "MCHP", "SWKS", "QRVO", "COHU", "BRKS", "ONTO", "FORM",
            "COHR", "IPGP", "LITE", "VIAV", "CEVA", "ACMR",
        ],
    },
    "big_tech": {
        "name": "Big Tech",
        "color": "#6366f1",
        "symbols": [
            "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NFLX",
            "DELL", "HPE", "HPQ", "WDC", "STX", "NTAP", "LOGI",
        ],
    },
    "cloud_saas": {
        "name": "Cloud & SaaS",
        "color": "#a78bfa",
        "symbols": [
            "CRM", "NOW", "WDAY", "HUBS", "PAYC", "PCTY", "VEEV",
            "ZM", "DOCU", "BOX", "TWLO", "RNG", "FIVN", "APPN",
            "SMAR", "MNDY", "GTLB", "CFLT", "DCBO", "SPT",
        ],
    },
    "cybersecurity": {
        "name": "Cybersecurity",
        "color": "#f87171",
        "symbols": [
            "CRWD", "PANW", "ZS", "FTNT", "S", "OKTA", "CYBR",
            "QLYS", "TENB", "RPD", "VRNT", "SAIL", "CHKP",
        ],
    },
    "data_center": {
        "name": "Data Centers & Infra",
        "color": "#22d3ee",
        "symbols": ["EQIX", "DLR", "VRT", "AMT", "CCI", "SBAC", "IRON", "IREN"],
    },
    "fintech": {
        "name": "Fintech",
        "color": "#e879f9",
        "symbols": [
            "SQ", "PYPL", "COIN", "HOOD", "AFRM", "NU", "SOFI",
            "UPST", "LC", "IBKR", "SCHO",
        ],
    },
    "payments": {
        "name": "Payments",
        "color": "#c084fc",
        "symbols": ["V", "MA", "FI", "FIS", "GPN", "FLUT", "WEX", "EVTC"],
    },
    "banking": {
        "name": "Banking & Finance",
        "color": "#93c5fd",
        "symbols": [
            "JPM", "GS", "MS", "BAC", "WFC", "C", "USB", "PNC",
            "TFC", "FITB", "CFG", "RF", "HBAN", "KEY", "MTB",
        ],
    },
    "asset_mgmt": {
        "name": "Asset Management",
        "color": "#7dd3fc",
        "symbols": ["BLK", "BX", "KKR", "APO", "ARES", "BAM", "OWL"],
    },
    "insurance": {
        "name": "Insurance",
        "color": "#bae6fd",
        "symbols": ["PGR", "TRV", "AFL", "MET", "PRU", "CB", "ALL", "HIG", "AIG", "RLI"],
    },
    "pharma": {
        "name": "Pharma (Large Cap)",
        "color": "#6ee7b7",
        "symbols": [
            "LLY", "JNJ", "ABBV", "BMY", "MRK", "PFE", "AMGN",
            "AZN", "NVO", "RHHBY",
        ],
    },
    "biotech": {
        "name": "Biotech & Life Sciences",
        "color": "#86efac",
        "symbols": [
            "MRNA", "BNTX", "REGN", "VRTX", "GILD", "BIIB", "INCY",
            "ILMN", "EXAS", "NTRA", "PACB", "TWST", "RXRX", "ROIV",
        ],
    },
    "gene_editing": {
        "name": "Gene Editing",
        "color": "#4ade80",
        "symbols": ["CRSP", "NTLA", "BEAM", "EDIT", "FATE", "ACMR"],
    },
    "medical_devices": {
        "name": "Medical Devices",
        "color": "#34d399",
        "symbols": [
            "ISRG", "SYK", "BSX", "MDT", "EW", "PODD", "DXCM",
            "RMD", "BDX", "ZBH", "INSP", "MMSI", "NVST",
        ],
    },
    "healthcare_services": {
        "name": "Healthcare Services",
        "color": "#2dd4bf",
        "symbols": ["UNH", "CVS", "CI", "HUM", "MCK", "ABC", "CAH", "DGX", "LH"],
    },
    "healthtech": {
        "name": "Health Tech",
        "color": "#67e8f9",
        "symbols": ["TDOC", "HIMS", "DOCS", "PHR", "IQVIA", "ACCD"],
    },
    "ev": {
        "name": "Electric Vehicles",
        "color": "#a3e635",
        "symbols": ["TSLA", "RIVN", "NIO", "LI", "XPEV", "LCID", "CHPT", "BLNK", "EVGO"],
    },
    "clean_energy": {
        "name": "Clean Energy",
        "color": "#34d399",
        "symbols": [
            "NEE", "FSLR", "ENPH", "CEG", "VST", "RUN", "BE", "PLUG",
            "SEDG", "ARRY", "NOVA", "FCEL", "HASI", "CWEN", "FLNC",
        ],
    },
    "energy": {
        "name": "Oil & Gas",
        "color": "#fbbf24",
        "symbols": [
            "XOM", "CVX", "COP", "SLB", "OXY", "EOG", "MPC", "VLO",
            "HAL", "DVN", "PSX", "FANG", "HES", "MRO", "APA",
        ],
    },
    "defense": {
        "name": "Defense & Aerospace",
        "color": "#cbd5e1",
        "symbols": [
            "LMT", "RTX", "NOC", "GD", "BA", "HII", "KTOS", "AVAV",
            "LDOS", "SAIC", "BAH", "CACI", "AXON", "BWXT",
        ],
    },
    "space": {
        "name": "Space",
        "color": "#c7d2fe",
        "symbols": ["RKLB", "ASTS", "LUNR", "SPCE", "MNTS", "GSAT"],
    },
    "industrial": {
        "name": "Industrials",
        "color": "#94a3b8",
        "symbols": [
            "GE", "HON", "CAT", "DE", "EMR", "ROK", "ETN", "PH",
            "ITW", "IR", "MMM", "CGNX", "IRBT", "FORM",
        ],
    },
    "construction": {
        "name": "Construction & Materials",
        "color": "#fb923c",
        "symbols": [
            "VMC", "MLM", "NUE", "STLD", "URI", "PWR", "BLDR",
            "X", "ACM", "CLF", "CMC", "RS", "MAS", "BECN",
        ],
    },
    "logistics": {
        "name": "Logistics & Supply Chain",
        "color": "#fdba74",
        "symbols": ["UPS", "FDX", "XPO", "ODFL", "SAIA", "JBHT", "GXO", "CHRW", "EXPD", "ARCB"],
    },
    "ecommerce": {
        "name": "E-Commerce",
        "color": "#fcd34d",
        "symbols": [
            "SHOP", "MELI", "ETSY", "EBAY", "W", "PDD",
            "BABA", "JD", "CPNG", "GLBE",
        ],
    },
    "consumer_disc": {
        "name": "Consumer Discretionary",
        "color": "#fde68a",
        "symbols": [
            "AMZN", "HD", "LOW", "NKE", "SBUX", "MCD", "CMG", "YUM",
            "TJX", "ROST", "BURL", "DG", "DLTR", "LULU", "UAA",
        ],
    },
    "consumer_staples": {
        "name": "Consumer Staples",
        "color": "#d9f99d",
        "symbols": ["WMT", "COST", "KO", "PEP", "PG", "MDLZ", "GIS", "CL", "KMB"],
    },
    "streaming_media": {
        "name": "Streaming & Media",
        "color": "#f9a8d4",
        "symbols": ["NFLX", "DIS", "SPOT", "ROKU", "WBD", "PARA", "FUBO", "FOXA"],
    },
    "social_media": {
        "name": "Social Media & Internet",
        "color": "#f472b6",
        "symbols": ["META", "SNAP", "PINS", "RDDT", "MTCH", "BMBL", "YELP"],
    },
    "gaming": {
        "name": "Gaming & Entertainment",
        "color": "#e879f9",
        "symbols": ["EA", "TTWO", "RBLX", "U", "DKNG", "PENN", "LVS", "MGM", "WYNN", "CZR"],
    },
    "travel": {
        "name": "Travel & Hospitality",
        "color": "#d8b4fe",
        "symbols": ["ABNB", "BKNG", "EXPE", "MAR", "HLT", "H", "UBER", "LYFT", "DAL", "UAL", "AAL"],
    },
    "real_estate": {
        "name": "Real Estate & REITs",
        "color": "#fca5a5",
        "symbols": [
            "PLD", "PSA", "O", "WELL", "EXR", "INVH", "VICI",
            "GLPI", "ARE", "DLR", "REXR", "COLD", "EQR", "AVB",
        ],
    },
    "telecom": {
        "name": "Telecom",
        "color": "#6b7280",
        "symbols": ["T", "VZ", "TMUS", "LUMN", "GSAT"],
    },
    "materials_mining": {
        "name": "Materials & Mining",
        "color": "#a8a29e",
        "symbols": [
            "FCX", "NEM", "GOLD", "AEM", "ALB", "MP", "SQM",
            "LAC", "LTHM", "AA", "VALE",
        ],
    },
    "crypto_mining": {
        "name": "Crypto & Bitcoin Mining",
        "color": "#fbbf24",
        "symbols": ["MSTR", "COIN", "HOOD", "RIOT", "MARA", "HUT", "CLSK", "BITF", "WULF", "BTBT"],
    },
    "autonomous": {
        "name": "Autonomous & Robotics",
        "color": "#7dd3fc",
        "symbols": ["TSLA", "MBLY", "LAZR", "OUST", "CGNX", "NVDA", "GOOGL"],
    },
    "agtech": {
        "name": "Agriculture & Food Tech",
        "color": "#bbf7d0",
        "symbols": ["CTVA", "CF", "MOS", "AGCO", "FMC", "ICL", "INGR"],
    },
    "utilities": {
        "name": "Utilities",
        "color": "#e2e8f0",
        "symbols": ["NEE", "D", "SO", "DUK", "EXC", "AES", "NRG", "ETR", "PEG", "ES", "AWK"],
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
