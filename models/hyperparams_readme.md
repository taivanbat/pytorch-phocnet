--learning_rate_step: 'A dictionary-like string indicating the learning rate for up to the number of iterations. '
                             'E.g. the default \'70000:1e-4,80000:1e-5\' means learning rate 1e-4 up to step 70000 and 1e-5 till 80000.'
    parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.9,
                        help='The momentum for SGD training (or beta1 for Adam). Default: 0.9')
    parser.add_argument('--momentum2', '-mom2', action='store', type=float, default=0.999,
                        help='Beta2 if solver is Adam. Default: 0.999')
    parser.add_argument('--delta', action='store', type=float, default=1e-8,
                        help='Epsilon if solver is Adam. Default: 1e-8')
    parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                        help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
    parser.add_argument('--display', action='store', type=int, default=500,
                        help='The number of iterations after which to display the loss values. Default: 100')
    parser.add_argument('--test_interval', action='store', type=int, default=2000,
                        help='The number of iterations after which to periodically evaluate the PHOCNet. Default: 500')
    parser.add_argument('--iter_size', '-is', action='store', type=int, default=10,
                        help='The batch size after which the gradient is computed. Default: 10')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=1,
                        help='The batch size after which the gradient is computed. Default: 1')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.00005,
                        help='The weight decay for SGD training. Default: 0.00005')
    parser.add_argument('--min_image_width_height', '-miwh', action='store', type=int, default=26,
                        help='The minimum width or height of the images that are being fed to the AttributeCNN. Default: 26')
    parser.add_argument('--phoc_unigram_levels', '-pul',
                        action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='1,2,4,8',
                        help='The comma seperated list of PHOC unigram levels. Default: 1,2,4,8')
    parser.add_argument('--embedding_type', '-et', action='store',
                        choices=['phoc', 'spoc', 'dctow', 'phoc-ppmi', 'phoc-pruned'],
                        default='phoc',
                        help='The label embedding type to be used. Possible: phoc, spoc, phoc-ppmi, phoc-pruned. Default: phoc')
    parser.add_argument('--fixed_image_size', '-fim', action='store',
                        type=lambda str_tuple: tuple([int(elem) for elem in str_tuple.split(',')]),
                        default=None ,
                        help='Specifies the images to be resized to a fixed size when presented to the CNN. Argument must be two comma seperated numbers.')
    parser.add_argument('--dataset', '-ds', required=True, choices=['gw','iam', 'wiener'], default= 'wiener',
                        help='The dataset to be trained on')