# Hand reference for shelf_carriers.png, one colour per carrier, top to bottom.
# Made by inspecting 8x zooms of the photo. At ~4-5 px per carrier the
# reference itself carries roughly +/-1-2 carriers of uncertainty per stack
# (bottom-left, mostly black, is the least certain: about +/-3).
# 'slot' is the carrier's bottom-edge slot row where a detection matched it.
REFERENCE = {
    'top-left': [
        (45, 'white'), (50, 'white'), (55, 'black'), (59, 'black'), (65, 'black'), (69, 'black'),
        (72, 'silver'), (77, 'silver'), (82, 'green'), (86, 'green'), (91, 'orange'), (95, 'silver'),
        (100, 'silver'), (105, 'tan'), (110, 'white'), (115, 'orange'), (120, 'orange'),
        (124, 'dark green'), (129, 'dark green'), (134, 'black'), (139, 'white'), (144, 'black'),
        (149, 'pink'), (153, 'black'), (158, 'black'), (163, 'white'), (168, 'orange'),
        (173, 'yellow'), (178, 'yellow'), (183, 'blue'), (188, 'blue'), (192, 'white'),
        (197, 'white'), (202, 'white')],
    'top-right': [
        (53, 'black'), (57, 'black'), (63, 'white'), (68, 'white'), (73, 'white'), (78, 'amber'),
        (82, 'black'), (87, 'yellow'), (92, 'white'), (97, 'white'), (102, 'white'), (107, 'black'),
        (112, 'green'), (116, 'black'), (121, 'black'), (126, 'silver'), (132, 'amber'),
        (138, 'blue'), (143, 'light green'), (149, 'black'), (154, 'orange'), (158, 'black'),
        (164, 'white'), (168, 'black'), (173, 'navy'), (179, 'navy'), (184, 'black'),
        (189, 'black'), (194, 'pink'), (199, 'tan'), (204, 'amber')],
    'bottom-left': [
        (282, 'black'), (286, 'black'), (290, 'white'), (294, 'black'), (298, 'black'),
        (301, 'black'), (304, 'black'), (308, 'black'), (312, 'black'), (316, 'dark green'),
        (320, 'white'), (324, 'black'), (328, 'black'), (332, 'black'), (336, 'black'),
        (340, 'black'), (344, 'black'), (348, 'white'), (352, 'silver'), (356, 'white'),
        (360, 'silver'), (364, 'black'), (368, 'black'), (371, 'black'), (374, 'red')],
    'bottom-right': [
        (253, 'white'), (258, 'white'), (261, 'white'), (265, 'orange'), (270, 'yellow'),
        (274, 'yellow'), (278, 'pink'), (282, 'pink'), (286, 'tan'), (290, 'blue'), (294, 'blue'),
        (298, 'black'), (301, 'black'), (305, 'red'), (309, 'black'), (314, 'white'),
        (318, 'black'), (322, 'amber'), (325, 'black'), (331, 'white'), (335, 'black'),
        (339, 'pink'), (343, 'orange'), (348, 'yellow'), (353, 'orange'), (357, 'pink'),
        (363, 'black'), (367, 'pink'), (371, 'tan'), (376, 'pink'), (380, 'pink')],
}
