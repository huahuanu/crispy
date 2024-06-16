import argparse

from src.utils import TrainInterface

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')  #
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--lr', type=float, default=1e-3 , help='learning rate')  #
    # parser.add_argument('--lr_min', type=float, default=3e-4 , help='learning rate')  #
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--save_name', type=str, default='jaad', help='weight path')
    parser.add_argument('--data_path', type=str, default='../data/JAAD_dataset/', help='dataset path')
    parser.add_argument('--set_path', type=str, default='../data/JAAD/', help='dataset set path')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose device')

    # Set style
    parser.add_argument('--train', type=bool, default=True, help='train or not.')
    parser.add_argument('--test', type=bool, default=True, help='test or not.')
    parser.add_argument('--monitor_acc', type=bool, default=True, help='default monitor mode to val_acc[Default] or val_loss.')
    parser.add_argument('--early_stop', type=bool, default=True, help='early stop.')
    parser.add_argument('--threshold', type=int, default=15, help='threshold for early stop.')

    # Set model
    parser.add_argument('--d_model', type=int, default=64, help='model dimension') #
    parser.add_argument('--dff', type=int, default=128, help='hidden dimension') #
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')  #
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads') 
    parser.add_argument('--drop_rate', type=float, default=0.5, help='dropout rate') #
    parser.add_argument('--num_class', type=int, default=2, help='number of class') 
    # Set dataset
    parser.add_argument('--time_scale', type=int, default=24, help='time scale')
    parser.add_argument('--bbox_size', type=int, default=4, help='bbox size')
    parser.add_argument('--vel_size', type=int, default=2, help='vel size')
    parser.add_argument('--time_crop', type=bool, default=False, help='time crop or not.')
    parser.add_argument('--time_crop_scale', type=int, default=8, help='time crop size.')
    # Others
    parser.add_argument('--debug', type=bool, default=False, help='debug or not.')
    args = parser.parse_args()

    # Call TrainInterface
    call_cell = TrainInterface(dataset='JAAD',args=args)
    call_cell.run(args=args)
    print("Finished!")

if __name__ == "__main__":
    main()
