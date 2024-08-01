
import argparse


def args_parser(dataset):
    parser = argparse.ArgumentParser()

    ################### PAMAP ###################################
    if dataset=='pamap':
        # dataset information
        parser.add_argument('--dataset',type=str,default='pamap',help='harth, pamap')

        # model parameters
        parser.add_argument('--epochs', type=int, default=50,
                            help="number of rounds of training")
        
        parser.add_argument('--input_dim', type=int,default=18)
        
        parser.add_argument('--hidden_dim', type=int, default=8)

        parser.add_argument('--class_num', type=int, default=12)
        
        parser.add_argument('--batch_size', type=int, default=32 )

        parser.add_argument('--batchsum', type=int, default=2500 )

        parser.add_argument('--num_users',type=int, default=4) # activities from 4 users 

        parser.add_argument('--test_users_num',type=int,default=5)

        parser.add_argument('--header',type=bool, default=False)

        parser.add_argument('--e',type=float,default=0.4,help='lower bound confidence')
    

    
    ################### harth ###################################
    if dataset=='harth':
        # dataset information
        parser.add_argument('--dataset',type=str,default='harth',help='harth, pamap')

        # model parameters
        parser.add_argument('--epochs', type=int, default=50,
                            help="number of rounds of training")
        
        parser.add_argument('--input_dim', type=int,default=6)
        
        parser.add_argument('--hidden_dim', type=int, default=8)

        parser.add_argument('--class_num', type=int, default=15)
        
        parser.add_argument('--batch_size', type=int, default=32 )

        parser.add_argument('--batchsum', type=int, default=2500 )

        parser.add_argument('--num_users',type=int, default=15)

        parser.add_argument('--test_users_num',type=int,default=22)

        parser.add_argument('--header',type=bool, default=True)

        parser.add_argument('--e',type=float,default=0.3,help='lower bound confidence')
    
    if dataset=='harthsp':
        # dataset information
        parser.add_argument('--dataset',type=str,default='harthsp',help='harth, pamap')

        # model parameters
        parser.add_argument('--epochs', type=int, default=50,
                            help="number of rounds of training")
        
        parser.add_argument('--input_dim', type=int,default=6)
        
        parser.add_argument('--hidden_dim', type=int, default=8)

        parser.add_argument('--class_num', type=int, default=15)
        
        parser.add_argument('--batch_size', type=int, default=32 )

        parser.add_argument('--batchsum', type=int, default=2500 )

        parser.add_argument('--num_users',type=int, default=15)

        parser.add_argument('--test_users_num',type=int,default=18)

        parser.add_argument('--header',type=bool, default=True)

        parser.add_argument('--e',type=float,default=0.4,help='lower bound confidence')
    
    ########### general model parameters
    parser.add_argument('--device',type=str,default='cuda',help='cuda,cpu')
    parser.add_argument('--model_name',type=str,default='ditto',help='fedavg, fedprox, ditto')
    parser.add_argument('--whole_change',type=bool,default=True)
    parser.add_argument('--whole_update',type=bool,default=True)
    
    parser.add_argument('--local_ep',type=int,default=1 )

    parser.add_argument('--beta',type=float,default=0.1)

    parser.add_argument('--th',type=int,default=5,help='number to trig human feedback')

    
    
    parser.add_argument('--window_size',type=int,default=200)
    
    parser.add_argument('--lr', type=float, default=0.001,
                            help='learning rate')
        
    parser.add_argument('--momentum', type=float, default=0.9,
                            help='SGD momentum (default: 0.5)')
        
    parser.add_argument('--wd', type=float, default=5e-4,
                            help='weight decay')
        
    parser.add_argument('--optimizer',type=str, default='adam',help='sgd, adam')
        
    parser.add_argument('--lossfn', type=str, default='CrossEntropyLoss',
                            help='NLLLoss or CrossEntropyLoss')
    args = parser.parse_args()
    return args