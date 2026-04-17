import argparse

def initParams():
    parser = argparse.ArgumentParser(description="Configuration for the project")

    parser.add_argument('--seed', type=int, help="Random number seed for reproducibility", default=688)

    # Train & Dev Data folder prepare
    parser.add_argument("--atadd_t1_train_audio", type=str, help="Path to the training audio for ATADD T1 dataset",
                        default='yourpath/atadd/T1/train')
    parser.add_argument("--atadd_t1_train_label", type=str, help="Path to the training label for ATADD T1 dataset",
                        default="yourpath/atadd/T1/label/train.csv")
    parser.add_argument("--atadd_t1_dev_audio", type=str, help="Path to the development audio for ATADD T1 dataset",
                        default='yourpath/atadd/T1/dev')
    parser.add_argument("--atadd_t1_dev_label", type=str, help="Path to the development label for ATADD T1 dataset",
                        default="yourpath/atadd/T1/label/dev.csv")
    parser.add_argument("--atadd_t1_eval_audio", type=str, help="Path to the evaluation audio for ATADD T1 dataset",
                        default='yourpath/atadd/T1/eval')

    parser.add_argument("--atadd_t2_train_audio", type=str, help="Path to the training audio for ATADD T2 dataset",
                        default='/data/liulan/workspace/dataset/at_add_track2/train')
    parser.add_argument("--atadd_t2_train_label", type=str, help="Path to the training label for ATADD T2 dataset",
                        default="/data/liulan/workspace/dataset/at_add_track2/labels/train.csv")
    parser.add_argument("--atadd_t2_dev_audio", type=str, help="Path to the development audio for ATADD T2 dataset",
                        default='/data/liulan/workspace/dataset/at_add_track2/dev')
    parser.add_argument("--atadd_t2_dev_label", type=str, help="Path to the development label for ATADD T2 dataset",
                        default="/data/liulan/workspace/dataset/at_add_track2/labels/dev.csv")
    parser.add_argument("--atadd_t2_eval_audio", type=str, help="Path to the evaluation audio for ATADD T2 dataset",
                        default='/data/liulan/workspace/dataset/at_add_track2/eval')


    # SSL folder prepare
    parser.add_argument("--xlsr", default="/data/liulan/workspace/huggingface/wav2vec2-xls-r-300m")
    parser.add_argument("--wavlm", default="/data/liulan/workspace/huggingface/wavlm-large")
    parser.add_argument("--mert", default="/data/liulan/workspace/huggingface/MERT-v1-330M")
    parser.add_argument("--beats", default="/data/liulan/workspace/huggingface/BEATs_inter3")
    parser.add_argument("--clap", default="/data/liulan/workspace/huggingface/larger_clap_music_and_speech")

    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models/try/')

    # countermeasure
    parser.add_argument("--audio_len", type=int, help="raw waveform length", default=64600)
    parser.add_argument('-m', '--model', help='Model arch', default='pt-w2v2aasist',
                        # choices=['specresnet', 'aasist', 'ft-w2v2aasist', 'fr-wavlmaasist', 'fr-mertaasist',
                        #          'fr-w2v2aasist', 'ft-wavlmaasist', 'ft-mertaasist',
                        #          'pt-w2v2aasist', 'wpt-w2v2aasist',
                        #          'pt-wavlmaasist', 'wpt-wavlmaasist',
                        #          'pt-mertaasist', 'wpt-mertaasist']
                        )

    # dual-SSL feature fusion
    parser.add_argument(
        "--fusion",
        type=str,
        default="cat_linear",
        choices=["cat_linear", "gated", "cross_attn", "film", "type_aware",
                 "proj512_cat", "add"],
        help=(
            "Feature fusion method for dual-SSL models (e.g. ft-xlsrmertaasist). "
            "cat_linear: concat+Linear (default); "
            "gated: soft-gate interpolation; "
            "cross_attn: bidirectional cross-attention; "
            "film: Feature-wise Linear Modulation; "
            "type_aware: dynamic per-type weights with auxiliary type-clf loss; "
            "proj512_cat: each stream projected to 512 then cat (no joint linear); "
            "add: element-wise sum, zero extra parameters."
        ),
    )
    parser.add_argument(
        "--type_loss_weight",
        type=float,
        default=0.1,
        help=(
            "Weight of the auxiliary audio-type classification loss "
            "when --fusion type_aware is used. Set to 0 to disable."
        ),
    )

    # pt
    parser.add_argument("--prompt_dim", type=int, help="prompt dim", default=1024)
    parser.add_argument("--num_prompt_tokens", type=int, help="audio dim", default=10)
    parser.add_argument("--pt_dropout", type=float, help="dropout", default=0.1)

    # wpt
    parser.add_argument("--num_wavelet_tokens", type=int, help="wavelet token", default=4)

    return parser
