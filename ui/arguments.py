import argparse


def add_base_args(parser):
    parser.add_argument(
        "--attention_layers_to_use",
        nargs="+",
        type=str,
        default=[
            'down_blocks[2].attentions[0].transformer_blocks[0].attn2',  ##########5
            'down_blocks[2].attentions[1].transformer_blocks[0].attn2',  ##########6

            'mid_block.attentions[0].transformer_blocks[0].attn2',
            'up_blocks[2].attentions[1].transformer_blocks[0].attn1',
            "up_blocks[3].attentions[1].transformer_blocks[0].attn1",  #############3

            "up_blocks[1].attentions[0].transformer_blocks[0].attn2",  ########## +8
            "up_blocks[1].attentions[1].transformer_blocks[0].attn2",  ########## +9
            "up_blocks[1].attentions[2].transformer_blocks[0].attn2",  ########## +10
            "up_blocks[2].attentions[0].transformer_blocks[0].attn2",  # +11
            "up_blocks[2].attentions[1].transformer_blocks[0].attn2",  # +12
        ],
    )
    return parser


def add_test_args(parser):
    parser.add_argument("--test_t", nargs="+", type=int, default=[100])
    parser.add_argument("--test_mask_size", type=int, default=512)
    return parser


def add_shell_args(parser):
    parser.add_argument("--rand_seed", type=int, default=1101)
    parser.add_argument("--enhanced", type=float, default=1.6)
    return parser


def add_data_args(parser):
    parser.add_argument("--text_captions", type=str, default=None)
    parser.add_argument("--num_class", type=int, default=0)
    return parser


def init_args():
    parser = argparse.ArgumentParser()
    parser = add_base_args(parser)
    parser = add_test_args(parser)
    parser = add_shell_args(parser)
    parser = add_data_args(parser)
    args = parser.parse_known_args()[0]
    return args
