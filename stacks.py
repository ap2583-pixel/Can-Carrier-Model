# Hand-drawn regions for the four stacks in shelf_carriers.png (374x418 px).
# top/bottom = first/last row of carrier plastic; slot = columns through the
# side openings; solid = columns of plain carrier wall used for colour.
STACKS = {
    'top-left':     dict(slot=[93, 94, 95, 96, 97, 98], solid=list(range(76, 90)), top=41, bottom=202),
    'top-right':    dict(slot=[217, 218, 219, 220, 221, 247, 248, 249], solid=list(range(198, 214)), top=49, bottom=205),
    'bottom-left':  dict(slot=[78, 79, 80, 119, 120, 121], solid=list(range(86, 108)), top=279, bottom=377),
    'bottom-right': dict(slot=[198, 199, 200, 201, 253, 254], solid=list(range(206, 222)), top=250, bottom=381),
}
