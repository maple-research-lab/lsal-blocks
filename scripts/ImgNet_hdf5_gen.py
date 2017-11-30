__author__ = 'newmoon'
from tflearn.data_utils_mahtab import build_hdf5_image_dataset

if __name__ == "__main__":
    target_path='/home/newmoon/GAN/bigan-master/data/imagenet/train.txt'
    base_img_path='/home/newmoon/GAN/bigan-master/data/imagenet64/train'
    target_path_valid= '/home/newmoon/GAN/bigan-master/data/imagenet/val.txt'
    base_img_path_valid='/home/newmoon/GAN/bigan-master/data/imagenet64/val'
    image_shape=(64,64)
    output_path='/home/newmoon/GAN/bigan-master/data/imagenet64/ImgNet64_train_unnormalize.hdf5'

    #(target_path, base_img_path, image_shape, output_path='dataset.h5',
                      #       mode='file', categorical_labels=True,
                       #      normalize=True, grayscale=False,
                        #     files_extension=None, chunks=False,)
    build_hdf5_image_dataset(target_path, base_img_path,target_path_valid,base_img_path_valid,image_shape,output_path, 'file', True, False,False,
                             None, False)

    '''
    def build_hdf5_image_dataset(target_path, base_img_path, target_path_valid,base_img_path_valid, image_shape, output_path='dataset.h5',
                             mode='file', categorical_labels=True,
                             normalize=True, grayscale=False,
                             files_extension=None, chunks=False)
    '''
    print('done')

    #build_hdf5_image_dataset(target_path=target_path, base_img_path=base_img_path,image_shape, image_shape,output_path=output_path, mode='file',categorical_labels= True, normalize=True,grayscale=False)

