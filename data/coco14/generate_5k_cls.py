# import os
#
# VAL_CLS_FILE = 'val_cls.txt'
# VAL5K_FILE = 'val_5k.txt'
# VAL5K_CLS_FILE = 'val_5k_cls.txt'
#
# val_list = tuple(open(VAL_CLS_FILE, "r"))
# val_name_list, val_label_list = [], []
# for dat in val_list:
#     dat = dat.strip().split(" ")
#     val_name_list.append(dat[0])
#     val_label_list.append(dat[1:])
#
# val5k_name_list = open(VAL5K_FILE, "r")
# val5k_name_list = [id_.rstrip() for id_ in val5k_name_list]
#
# with open(VAL5K_CLS_FILE, "w") as f:
#     for dat in val5k_name_list:
#         try:
#             idx = val_name_list.index(dat)
#             line = dat + "".join([" " + i for i in val_label_list[idx]])
#             f.writelines(line + "\n")
#         except ValueError:
#             print(dat)


val_list = tuple(open("train_cls_sel.txt", "r"))
val_name_list = []
for dat in val_list:
    dat = dat.strip().split(" ")
    val_name_list.append(dat[0])

with open("train_sel.txt", "w") as f:
    for dat in val_name_list:
        f.writelines(dat + "\n")
