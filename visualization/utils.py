def orientation2camera(ori):
    dic = {
        "front": "CAM_FRONT",
        "front_right": "CAM_FRONT_RIGHT",
        "front_left": "CAM_FRONT_LEFT",
        "back_right": "CAM_BACK_RIGHT",
        "back_left": "CAM_BACK_LEFT",
        "back": "CAM_BACK"
    }
    return dic[ori]
