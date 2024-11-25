import argparse


def add_base_args(parser):
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="../outputs")
    parser.add_argument(
        "--attention_layers_to_use",
        nargs="+",
        type=str,
        default=[
            'down_blocks[0].attentions[0].transformer_blocks[0].attn2',
            'down_blocks[1].attentions[1].transformer_blocks[0].attn2',
            'down_blocks[2].attentions[0].transformer_blocks[0].attn2',  ##########5
            'down_blocks[2].attentions[1].transformer_blocks[0].attn2',  ##########6
            "up_blocks[1].attentions[0].transformer_blocks[0].attn2",  ########## +8
            "up_blocks[1].attentions[1].transformer_blocks[0].attn2",  ########## +9
            "up_blocks[1].attentions[2].transformer_blocks[0].attn2",  ########## +10
            "up_blocks[2].attentions[0].transformer_blocks[0].attn2",  # +11
            "up_blocks[2].attentions[1].transformer_blocks[0].attn2",  # +12
            "up_blocks[3].attentions[1].transformer_blocks[0].attn1",  #############3
            'up_blocks[3].attentions[2].transformer_blocks[0].attn2',
            'mid_block.attentions[0].transformer_blocks[0].attn1',
            'mid_block.attentions[0].transformer_blocks[0].attn2'
        ],
    )
    return parser


def add_voc_dataset_args(parser):
    parser.add_argument("--dataset_name", type=str, default="voc2012")
    parser.add_argument("--root_datasets_path", type=str, default="/home/sunl/workspace/datasets/VOCdevkit/VOC2012/")
    parser.add_argument("--num_class", type=int, default=20)
    parser.add_argument("--cam_bg_thr", type=float, default=0)
    parser.add_argument("--train_data_dir", type=str, default='../data/voc/train_aug.txt')
    parser.add_argument("--val_data_dir", type=str, default='../data/voc/val.txt')
    parser.add_argument("--test_data_dir", type=str, default='../data/voc/val.txt')

    # use_json means use diffsegmenter to get image label, if false, use tagclip to get image labels
    parser.add_argument("--use_json", type=bool, default=False)
    parser.add_argument("--json_path", type=str, default='../data/voc/predict_syn_txt_0.8_img_0.97.json')

    parser.add_argument("--iter", type=int, default=8)
    parser.add_argument("--enhanced", type=float, default=1.6)
    parser.add_argument("--ent", type=float, default=0.015)
    parser.add_argument("--att_mean", type=bool, default=False)

    return parser


def add_voc_context_dataset_args(parser):
    parser.add_argument("--dataset_name", type=str, default="voc_context")
    parser.add_argument("--root_datasets_path", type=str, default="/home/sunl/workspace/datasets/VOCdevkit/VOC2010/")
    parser.add_argument("--num_class", type=int, default=59)
    parser.add_argument("--cam_bg_thr", type=float, default=0)
    parser.add_argument("--train_data_dir", type=str, default='../data/voc/train_aug.txt')
    parser.add_argument("--val_data_dir", type=str, default='../data/voc/val.txt')
    parser.add_argument("--test_data_dir", type=str, default='../data/voc_context/val.txt')

    # use_json means use diffsegmenter to get image label, if false, use tagclip to get image labels
    parser.add_argument("--use_json", type=bool, default=False)
    parser.add_argument("--json_path", type=str, default='../data/voc_context/predict_syn_txt_0.8_img_0.97.json')

    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--enhanced", type=float, default=1.6)
    parser.add_argument("--ent", type=float, default=0.002)
    parser.add_argument("--att_mean", type=bool, default=True)

    return parser


def add_coco_object_dataset_args(parser):
    parser.add_argument("--dataset_name", type=str, default="coco_object")
    parser.add_argument("--root_datasets_path", type=str, default="/home/sunl/workspace/datasets/coco2014")
    parser.add_argument("--num_class", type=int, default=80)
    parser.add_argument("--cam_bg_thr", type=float, default=0.45)
    parser.add_argument("--train_data_dir", type=str, default='../data/coco14/train.txt')
    parser.add_argument("--val_data_dir", type=str, default='../data/coco14/val_5k.txt')
    parser.add_argument("--test_data_dir", type=str, default='../data/coco14/val_5k.txt')

    # use_json means use diffsegmenter to get image label, if false, use tagclip to get image labels
    parser.add_argument("--use_json", type=bool, default=False)
    parser.add_argument("--json_path", type=str, default='../data/coco14/predict_syn_txt_0.8_img_0.97.json')

    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--enhanced", type=float, default=1.6)
    parser.add_argument("--ent", type=float, default=0.2)
    parser.add_argument("--att_mean", type=bool, default=True)

    return parser


def add_train_args(parser):
    parser.add_argument("--train_mask_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    return parser


def add_test_args(parser):
    parser.add_argument("--num_patchs_per_side", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--test_t", nargs="+", type=int, default=[100])
    parser.add_argument("--test_mask_size", type=int, default=512)
    return parser


def add_classifier_args(parser):
    parser.add_argument('--model_path', type=str, default='ViT-B/16')
    parser.add_argument('--clip_patch_size', type=int, default=16)
    parser.add_argument('--classified_result_dir', type=str, default="output/classifier_result")
    return parser


def add_shell_args(parser):
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--rand_seed", type=int, default=1101)
    parser.add_argument("--no_use_self_ers", default=True, action="store_false")
    parser.add_argument("--no_use_cross_enh", default=True, action="store_false")
    parser.add_argument("--no_use_cluster", default=True, action="store_false")
    return parser


def init_args():
    parser = argparse.ArgumentParser()
    parser = add_base_args(parser)
    # parser = add_voc_dataset_args(parser)
    parser = add_voc_context_dataset_args(parser)
    # parser = add_coco_object_dataset_args(parser)
    parser = add_train_args(parser)
    parser = add_test_args(parser)
    parser = add_classifier_args(parser)
    parser = add_shell_args(parser)
    args = parser.parse_known_args()[0]
    return args
