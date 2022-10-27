"""
Mapping of textual and numerical labels for Snopes and PolitiFact.
"""
LABEL_DICTS = {
    'pomt': {
        'pants on fire!': 0,
        'false': 1,
        'mostly false': 2,
        'half-true': 3,
        'mostly true': 4,
        'true': 5
    },
    'snes': {
        'false': 0,
        'mostly false': 1,
        'mixture': 2,
        'mostly true': 3,
        'true': 4
    }
}
